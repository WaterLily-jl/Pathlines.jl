using WaterLily, LilyPad, StaticArrays, BiotSavartBCs, CUDA, Pathlines, GLMakie
import ColorSchemes: colorschemes

function make_lilypad_circle(; p=4, Δt=1.5, T=Float32, mem=Array)
    n, m = 2^(p+4), 9*2^p
    radius, center, ω = T(2^p), SVector{2,T}(m/2, m/2), zero(T)

    body = AutoBody((x, t) -> √(x'x) - radius, RigidMap(center, ω; ω))
    sim  = LilyBiotSim((n, m), (1, 0), 2radius; ν=0, Δt, body, mem, T)

    function points(θ) # endpoints of a spoke at angle θ
        r = radius * SVector{2,T}(cos(θ), sin(θ))
        c = center .+ T(0.5)
        [Point2f((c+r)...), Point2f((c-r)...)]
    end
    return sim, points
end

begin
    # Create the simulation and a spoke overlay
    mem = CUDA.functional() ? CUDA.CuArray : Array
    sim, points = make_lilypad_circle(;mem)
    θ = Ref(0f0); dt = 0.1f0
    spoke = Observable(points(θ[]))

    # Create the visualization with the spoke
    # Requires WaterLily#pathline-viz!
    N=5_000; life=UInt(200); fadetau=0.5
    colormap=:starrynight
    bgcolor   = colorschemes[colormap].colors[1]
    body_color = colorschemes[colormap].colors[end]
    fig, ax = viz!(sim; N, life, mem, remeasure=false, body_color, bgcolor, fadetau,
                   figsize=(1280, 720), colormap, colorrange=(0, 2))
    lines!(ax, spoke, color=:black, linewidth=4)
end

# begin
#     # Create a video with variable rotation rate
#     ω = Ref(0f0)
#     @time GLMakie.record(fig,"LilyPadCircle.mp4",dt:dt:100) do t
#         ω = 20<t<80 ? Float32((2-t/10)*sim.U/sim.L) : 0f0
#         sim.body = setmap(sim.body; ω); measure!(sim)
#         viz_step!(fig, sim, t; remeasure = false)
#         θ[] += ω*dt*sim.L/sim.U
#         spoke[] = points(θ[])
#         yield()   # hand control back to GLMakie event loop
#     end
# end

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

    # Advance viz & update spoke until window closes
    while events(fig).window_open[]
        viz_step!(fig, sim_time(sim) + dt)
        θ[] += ω[]*dt*sim.L/sim.U
        spoke[] = points(θ[])
        yield()   # hand control back to GLMakie event loop
    end
end