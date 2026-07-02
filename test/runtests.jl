using Pathlines, WaterLily, LilyPad, StaticArrays, Test
import Makie
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

@testset "PathlineCanvas" begin

    @testset "_rgba: converts to opaque RGBAf" begin
        c = Pathlines._rgba(:black)
        @test c isa Makie.RGBAf
        @test c.r == 0f0 && c.g == 0f0 && c.b == 0f0
        @test c.alpha == 1f0

        w = Pathlines._rgba(:white)
        @test w.r == 1f0 && w.g == 1f0 && w.b == 1f0
        @test w.alpha == 1f0
    end

    @testset "constructor: canvas size and scale factors" begin
        pc = PathlineCanvas(16, 8; bgcolor=:black, figsize=(64, 32))
        @test size(pc.canvas) == (64, 32)
        @test pc.sx ≈ Float32(64 / 16)
        @test pc.sy ≈ Float32(32 / 8)
        @test pc.figsize == (64, 32)
    end

    @testset "fade!: lerp canvas toward bgcolor" begin
        # White canvas faded 50% toward black must be exactly grey (0.5, 0.5, 0.5)
        pc = PathlineCanvas(4, 4; bgcolor=:black, fadealpha=0.5, colorrange=(0,1))
        fill!(pc.canvas, Pathlines._rgba(:white))
        fade!(pc)
        @test all(c -> c.r ≈ 0.5f0 && c.g ≈ 0.5f0 && c.b ≈ 0.5f0, pc.canvas)

        # Full fade (α=1) must snap to bgcolor exactly
        fill!(pc.canvas, Pathlines._rgba(:white))
        pc2 = PathlineCanvas(4, 4; bgcolor=:black, fadealpha=1.0, colorrange=(0,1))
        fill!(pc2.canvas, Pathlines._rgba(:white))
        fade!(pc2)
        @test all(c -> c.r == 0f0 && c.g == 0f0 && c.b == 0f0, pc2.canvas)
    end

    @testset "draw!: marks pixels along a diagonal segment" begin
        # 10×10 canvas with resolution matching sim coords (sx=sy=1)
        pc = PathlineCanvas(10, 10; bgcolor=:black, colorrange=(0,1),
                            figsize=(10,10), resolution=(10,10))
        bg  = pc.bgcolor
        pos  = [SA[8f0, 8f0]]
        pos⁰ = [SA[1f0, 1f0]]
        draw!(pc, pos, pos⁰, 1f0)
        n_painted = count(c -> c != bg, pc.canvas)
        @test n_painted >= 6   # diagonal across a 10×10 canvas is ~7 pixels
        # Pixels outside the segment bounding box must be untouched
        @test pc.canvas[10, 10] == bg
        @test pc.canvas[1, 10]  == bg
    end

    @testset "_draw_segments!: speed mapped to colormap" begin
        # 2-entry LUT: red=slow, blue=fast
        bg   = Pathlines._rgba(:black)
        red  = Makie.RGBAf(1f0, 0f0, 0f0, 1f0)
        blue = Makie.RGBAf(0f0, 0f0, 1f0, 1f0)
        cmap = [red, blue]

        # Slow segment: move 0.1 px in dt=1 → speed=0.1 < lo=1 → clamped to red
        canvas = fill(bg, 10, 10)
        Pathlines._draw_segments!(canvas, [SA[3.1f0, 3f0]], [SA[3f0, 3f0]],
                                  cmap, 1f0, 3f0, 1f0, 1f0, 1f0)
        painted = filter(c -> c != bg, vec(canvas))
        @test !isempty(painted)
        @test all(c -> c.r > c.b, painted)

        # Fast segment: move 10 px in dt=1 → speed=10 > hi=3 → clamped to blue
        canvas2 = fill(bg, 20, 20)
        Pathlines._draw_segments!(canvas2, [SA[1f0, 11f0]], [SA[1f0, 1f0]],
                                  cmap, 1f0, 3f0, 1f0, 1f0, 1f0)
        painted2 = filter(c -> c != bg, vec(canvas2))
        @test !isempty(painted2)
        @test all(c -> c.b > c.r, painted2)
    end

end
