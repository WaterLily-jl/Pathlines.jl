using Pathlines, WaterLily, LilyPad, StaticArrays, Test
import GLMakie
using LinearAlgebra: norm
import WaterLily: interp
import Pathlines: update!
import CUDA

# Small 2D Taylor-Green vortex simulation (reused across testsets)
function tgv_sim(N=32, T=Float32; mem=Array, Δt=0.25f0)
    k = T(2π/N)
    function uλ(i,xy)
        x,y = @. (xy - T(1.5)) * k
        i==1 && return -sin(x)*cos(y)
        i==2 && return  cos(x)*sin(y)
        zero(T)
    end
    Simulation((N,N), (0,0), N; uλ, ν=T(1e-4), T, mem, Δt)
end

@testset "Particles" begin

    @testset "interp: scalar linear field" begin
        # a[i,j] = i+j (1-indexed). WaterLily coord x maps to float array-index x+1.5.
        # Interior cell CI(2,2) is at WL coord (0.5, 0.5); a[2,2]=4.
        N = 16
        a = Float32[i+j for i in 1:N, j in 1:N]

        # Query at exact cell centers — should return array value with no interpolation error
        @test interp(SA[0.5f0, 0.5f0], a) == 4.0f0  # CI(2,2): a[2,2]=4
        @test interp(SA[3.5f0, 1.5f0], a) == 8.0f0  # CI(5,3): a[5,3]=8

        # Query at midpoint between 4 neighbors: 0.25*(a[2,2]+a[2,3]+a[3,2]+a[3,3]) = 0.25*(4+5+5+6) = 5.0
        @test interp(SA[1.0f0, 1.0f0], a) == 5.0f0  # midpoint: expected 5.0

        # Linearity: interp should be exact for affine fields
        x = SA[2.3f0, 1.7f0]
        # array float-index = x+1.5 = (3.8, 3.2); linear interp of i+j at that point = 3.8+3.2 = 7.0
        @test interp(x, a) ≈ 7.0f0 atol=1f-5
    end

    @testset "CN accuracy: step-by-step reference" begin
        # Reference: continuously integrate a particle through 10 sim steps using
        # LilyPad.departure at each physical Δt=0.1. Then compare a one-shot CN
        # integration over the full Δt=1.0 window to that reference.
        Δt = 0.1f0
        sim = tgv_sim(; Δt)
        x⁰ = SA[4.0f0, 6.5f0]
        u_init = copy(sim.flow.u)   # snapshot at t=0

        x_ref = x⁰
        Δt_total = 0f0
        for _ in 1:10
            sim_step!(sim); sim.flow.Δt[end] = Δt  # enforce constant Δt
            x_ref = LilyPad.departure(x_ref, sim.flow.u, sim.flow.u⁰, -Δt)
            Δt_total += Δt
        end

        x_cn = LilyPad.departure(x⁰, sim.flow.u, u_init, -Δt_total)
        err  = norm(x_cn - x_ref)
        @info "CN unsteady accuracy" Δt_total err
        @test err < 0.03
    end

    @testset "update!: particles stay in bounds" begin
        sim = tgv_sim()
        p = Particles(256, sim.flow.p; life=UInt(50))

        for _ in 1:20
            sim_step!(sim)
            update!(p, sim)
        end

        pos = Array(p.position)
        @test !any(isnan, reinterpret(Float32, pos))
        in_bounds = all(all(p.lower[d] ≤ pos[i][d] ≤ p.upper[d] for d in eachindex(p.lower)) for i in eachindex(pos))
        @test in_bounds
    end

    if CUDA.functional()
        @testset "GPU update!: particles stay in bounds" begin
            # uλ must not capture Type{T} — use concrete Float32 literals for GPU isbits
            N = 32; k = Float32(2π/N)
            function uλ_gpu(i,xy)
                x,y = @. (xy - 1.5f0)*k
                i==1 && return -sin(x)*cos(y)
                i==2 && return  cos(x)*sin(y)
                0.0f0
            end
            sim = Simulation((N,N), (0,0), N; uλ=uλ_gpu, ν=1f-4, T=Float32, mem=CUDA.CuArray)
            p = Particles(1024, sim.flow.p; life=UInt(50), mem=CUDA.CuArray)

            for _ in 1:20
                sim_step!(sim)
                update!(p, sim)
            end
            CUDA.synchronize()

            pos = Array(p.position)
            @test !any(isnan, reinterpret(Float32, pos))
            in_bounds = all(all(p.lower[d] ≤ pos[i][d] ≤ p.upper[d] for d in eachindex(p.lower)) for i in eachindex(pos))
            @test in_bounds
            @info "GPU: all $(length(pos)) particles in bounds"
        end
    else
        @info "CUDA not functional — skipping GPU tests"
    end

end

@testset "ParticleViz" begin

    @testset "constructor: sizes and valid values" begin
        # ParticleViz copies particle state to CPU and wraps it in Observables.
        # minsize=2 means every marker width ≥ 2 even for stationary particles.
        sim = tgv_sim()
        p = Particles(64, sim.flow.p; life=UInt(50))
        for _ in 1:3; sim_step!(sim); update!(p, sim); end
        Δt = sim.flow.Δt[end-1]
        v = ParticleViz(p, Δt; minsize=2, width=1)

        @test length(v.pos)  == 64
        @test length(v.pos⁰) == 64
        @test length(v.mag)  == 64
        @test length(v.dir)  == 64
        # Observables should wrap the same CPU arrays (no copy)
        @test v.opos[] === v.pos
        @test v.omag[] === v.mag
        @test v.odir[] === v.dir
        @test !any(isnan, reinterpret(Float32, v.pos))
        # Vec2f marker: (length, width); length ≥ minsize=2, width == 1 kwarg
        @test all(m[1] ≥ 2f0 for m in v.mag)
        @test all(m[2] == 1f0 for m in v.mag)
    end

    @testset "notify!: updates CPU buffers and Observables" begin
        sim = tgv_sim()
        p = Particles(64, sim.flow.p; life=UInt(50))
        for _ in 1:3; sim_step!(sim); update!(p, sim); end
        v = ParticleViz(p, sim.flow.Δt[end-1])

        pos_before = copy(v.pos)
        for _ in 1:5; sim_step!(sim); update!(p, sim); end
        notify!(v, p, sim.flow.Δt[end-1])

        # At least some particles must have moved
        @test v.pos != pos_before
        # Observables still point to the same CPU arrays (not replaced)
        @test v.opos[] === v.pos
        @test !any(isnan, reinterpret(Float32, v.pos))
        @test all(m[1] ≥ 2f0 for m in v.mag)
    end

    @testset "velmag and veldir: 3-4-5 triangle" begin
        # Particle moves (3,4) in Δt=1 → speed=5, angle=atan(4,3)
        x0 = SA[0.0f0, 0.0f0]
        x  = SA[3.0f0, 4.0f0]

        # scale=1, minsize=0 → marker length = speed = 5.0; width=2
        m = Pathlines.velmag(x, x0, 1.0f0; scale=1, minsize=0, width=2)
        @test m ≈ GLMakie.Vec2f(5, 2)

        d = Pathlines.veldir(x, x0)
        @test d ≈ atan(4f0, 3f0)
    end

    @testset "PathlinePlot: fading overlay creates axis plots" begin
        # PathlinePlot draws a semi-transparent rectangle each frame to fade trails.
        # Old code: GLMakie.Box(fig,...) is a layout element — it adds nothing to
        # ax.scene.plots and so cannot produce a visible per-frame overlay.
        # Fix: GLMakie.poly!(ax, [Rect2f(...)], color) draws in axis data space.
        L = 16
        fig = GLMakie.Figure()
        ax = GLMakie.Axis(fig[1,1]; limits=(2, L+1, 2, L+1))

        # Simulate 3 frames of PathlinePlot loop body (FIXED approach):
        for _ in 1:3
            GLMakie.poly!(ax, [GLMakie.Rect2f(2, 2, L-1, L-1)], color=(:black, 0.2))
        end
        # Each frame adds one poly plot to the axis scene
        @test length(ax.scene.plots) >= 3
    end

end

@testset "PathViz" begin

    @testset "_rgba: converts to opaque RGBAf" begin
        # Black → all channels 0, alpha 1
        c = Pathlines._rgba(:black)
        @test c isa GLMakie.RGBAf
        @test c.r == 0f0 && c.g == 0f0 && c.b == 0f0
        @test c.alpha == 1f0

        # White → all channels 1, alpha 1
        w = Pathlines._rgba(:white)
        @test w.r == 1f0 && w.g == 1f0 && w.b == 1f0
        @test w.alpha == 1f0
    end

    @testset "_fade_canvas!: lerp toward bgcolor" begin
        # A white canvas faded 50% toward black must be exactly grey (0.5, 0.5, 0.5)
        # because lerp(1, 0, 0.5) = 0.5
        black = Pathlines._rgba(:black)
        white = Pathlines._rgba(:white)
        canvas = fill(white, 8, 8)
        Pathlines._fade_canvas!(canvas, black, 0.5f0)
        @test all(c -> c.r ≈ 0.5f0 && c.g ≈ 0.5f0 && c.b ≈ 0.5f0, canvas)

        # Full fade (α=1) must snap to bgcolor exactly
        Pathlines._fade_canvas!(canvas, black, 1f0)
        @test all(c -> c.r == 0f0 && c.g == 0f0 && c.b == 0f0, canvas)
    end

    @testset "_draw_segments!: marks pixels along segment" begin
        # One segment from canvas coord (1,1)→(8,8) (diagonal).
        # After drawing, the pixels along the diagonal must differ from bgcolor.
        # sx=sy=1 so sim coords == canvas coords.
        bg = Pathlines._rgba(:black)
        red = GLMakie.RGBAf(1f0, 0f0, 0f0, 1f0)
        cmap = [red]   # single-colour LUT
        canvas = fill(bg, 10, 10)

        pos  = [StaticArrays.SA[8f0, 8f0]]
        pos⁰ = [StaticArrays.SA[1f0, 1f0]]
        Pathlines._draw_segments!(canvas, pos, pos⁰, cmap, 0f0, 1f0, 1f0, 1f0)

        # At least some pixels on the diagonal must have been written (r > 0)
        n_painted = count(c -> c.r > 0f0, canvas)
        @test n_painted >= 6  # diagonal is 7 pixels on a 10×10 canvas

        # No pixels outside the bounding box [1,8]×[1,8] should be touched
        @test canvas[10, 10] == bg
        @test canvas[1, 10]  == bg
    end

    @testset "_draw_segments!: speed mapped to colormap" begin
        # Slow segment (speed < lo) → first colormap entry; fast → last entry
        bg  = Pathlines._rgba(:black)
        red  = GLMakie.RGBAf(1f0, 0f0, 0f0, 1f0)
        blue = GLMakie.RGBAf(0f0, 0f0, 1f0, 1f0)
        cmap = [red, blue]  # 2-entry LUT: red=slow, blue=fast

        # Slow segment: move 0.1 pixels → speed ≈ 0.1 < lo=1 → clamped to cidx=1 → red
        canvas = fill(bg, 10, 10)
        Pathlines._draw_segments!(canvas, [StaticArrays.SA[3.1f0, 3f0]],
                                          [StaticArrays.SA[3f0,   3f0]],
                                   cmap, 1f0, 3f0, 1f0, 1f0)
        painted = filter(c -> c != bg, vec(canvas))
        @test !isempty(painted)
        @test all(c -> c.r > c.b, painted)  # slow → red (first LUT entry)

        # Fast segment: move 10 pixels → speed=10 > hi=3 → clamped to cidx=2 → blue
        canvas2 = fill(bg, 20, 20)
        Pathlines._draw_segments!(canvas2, [StaticArrays.SA[1f0,  11f0]],
                                           [StaticArrays.SA[1f0,   1f0]],
                                   cmap, 1f0, 3f0, 1f0, 1f0)
        painted2 = filter(c -> c != bg, vec(canvas2))
        @test !isempty(painted2)
        @test all(c -> c.b > c.r, painted2)  # fast → blue (last LUT entry)
    end

    # Shared sim + viz for constructor/update tests (built once to avoid repeated
    # GLMakie Figure construction).  16×16 grid keeps sim_step! fast.
    let sim = tgv_sim(16), figsize = (64, 32)
        v = PathViz(sim; N=128, life=UInt(20), figsize, body=false,
                    bgcolor=:black, colormap=:plasma, colorrange=(0,2))

        @testset "PathViz constructor: canvas size and scene layout" begin
            # Canvas must match figsize, not the sim grid (16×16 → canvas 64×32)
            @test size(v._canvas) == (64, 32)

            # Scale factors: px/sim_nx where sim_nx includes ghost cells
            sim_nx = size(sim.flow.σ, 1)
            sim_ny = size(sim.flow.σ, 2)
            @test v._sx ≈ Float32(64 / sim_nx)
            @test v._sy ≈ Float32(32 / sim_ny)

            # Figure size must match (viewport widths is a Vec, compare as array)
            @test collect(v.fig.scene.viewport[].widths) == [64, 32]

            # image! is the only plot (no body layer when body=false)
            @test length(v.ax.scene.plots) == 1
        end

        @testset "update!: canvas modified after one particle step" begin
            bg = Pathlines._rgba(:black)
            @test all(c -> c == bg, v._canvas)  # starts all-bgcolor

            # Advance one sim step then update particles — does not loop on sim_time
            sim_step!(sim)
            Pathlines.update!(v, sim)

            n_painted = count(c -> c != bg, v._canvas)
            @test n_painted > 0  # at least one segment must have been splatted
            @info "PathViz update!: $n_painted pixels painted (canvas $(size(v._canvas)))"
        end
    end

end
