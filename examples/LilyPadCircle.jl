using WaterLily, LilyPad, StaticArrays, BiotSavartBCs, CUDA, Pathlines, GLMakie
import ColorSchemes: colorschemes

function make_lilypad_circle(; N=2_000, life=UInt(200), Δt=1.5, T=Float32,
                               mem=CUDA.functional() ? CUDA.CuArray : Array,
                               colormap=:starrynight, p=4)
    n, m = 2^(p+4), 9*2^p
    radius, center, ω = T(2^p), SVector{2,T}(m/2, m/2), zero(T)

    body = AutoBody((x, t) -> √(x'x) - radius, RigidMap(center, ω; ω))
    sim  = LilyBiotSim((n, m), (1, 0), 2radius; ν=0, Δt, body, mem, T)

    bgcolor   = colorschemes[colormap].colors[1]
    bodycolor = colorschemes[colormap].colors[end]
    fig, ax = viz!(sim; N, life, mem, remeasure=false, body_color=bodycolor, bgcolor,
                   figsize=(1280, 720), colormap, colorrange=(0, 2))

    function points(θ) # endpoints of a spoke at angle θ
        r = radius * SVector{2,T}(cos(θ), sin(θ))
        c = center .+ T(0.5)
        [Point2f((c+r)...), Point2f((c-r)...)]
    end
    return sim, fig, ax, points
end

begin
    # Create the simulation and visualization, add a spoke overlay
    sim, fig, ax, points = make_lilypad_circle()
    θ = Ref(0f0); dt = 0.1f0
    spoke = Observable(points(θ[]))
    lines!(ax, spoke, color=:black, linewidth=4)
end

begin
    # Control rotation with arrow keys; space to stop
    ω = Ref(0f0); dω = Float32(sim.U/sim.L)
    on(events(fig).keyboardbutton) do event
        event.action == Keyboard.press || return
        if     event.key == Keyboard.up;    ω[] += dω
        elseif event.key == Keyboard.down;  ω[] -= dω
        elseif event.key == Keyboard.space; ω[]  = 0f0
        else; return; end
        sim.body = setmap(sim.body; ω=ω[])
        measure!(sim)
    end

    # Advance sim + pathlines, update spoke, until window closes
    while events(fig).window_open[]
        viz_step!(fig, sim_time(sim) + dt)
        θ[] += ω[]*dt*sim.L/sim.U
        spoke[] = points(θ[])
        yield()   # hand control back to GLMakie event loop
    end
end