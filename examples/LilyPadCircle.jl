using WaterLily, LilyPad, StaticArrays, BiotSavartBCs, CUDA, Pathlines, GLMakie
import ColorSchemes: colorschemes

function make_lilypad_circle(; N=2_000, life=UInt(200), Δt=1.5, T=Float32,
                               mem=CUDA.functional() ? CUDA.CuArray : Array,
                               colormap=:starrynight, p=7)
    n, m = 2^(p+1), 2^p
    radius, center, ω = T(m / 8), SVector{2,T}(m/2-1, m/2-1), zero(T)

    body=AutoBody((x, t) -> √(x'x) - radius,
                  RigidMap(center,ω; ω))

    sim = LilyBiotSim((n, m), (1, 0), 2radius; ν=0, Δt, body, mem, T)

    bgcolor = colorschemes[colormap].colors[1]
    bodycolor = colorschemes[colormap].colors[end]
    viz = PathViz(sim; N, life, mem, body=true, bgcolor, figsize=(1024, 512),
                  colormap, colorrange=(0, 2), bodycolor)

    function points(θ) # endpoints of a spoke at angle θ
        r = radius*SVector{2, T}(cos(θ), sin(θ))
        c = center .+ T(0.5)
        [Point2f((c+r)...),Point2f((c-r)...)]
    end
    return sim,viz,points
end

begin
    # Create the simulation and visualization objects, and display the figure
    sim,viz,points = make_lilypad_circle()
    Pathlines.step!(viz, sim, 0.1; remeasure=false)
    θ = Ref(0f0); dt = 0.1f0
    spoke = Observable(points(θ[]))
    lines!(viz.ax, spoke, color=:black, linewidth=4)    
    display(viz.fig)
end

 begin
    # Control the rotation-rate of the circle with the up/down arrow keys, and space to stop
    ω = Ref(0f0); dω = Float32(sim.U/sim.L)
    on(events(viz.fig).keyboardbutton) do event
        event.action == Keyboard.press || return
        if     event.key == Keyboard.up;    ω[] += dω
        elseif event.key == Keyboard.down;  ω[] -= dω
        elseif event.key == Keyboard.space; ω[]  = 0f0
        else; return; end
        sim.body = setmap(sim.body; ω=ω[])
        measure!(sim)
    end

    # Update the sim and spoke endpoints in a loop until the Makie window is closed
    while events(viz.fig).window_open[]
        t = sim_time(sim)+dt
        Pathlines.step!(viz, sim, t; remeasure = false)
        θ[] += ω[]*dt*sim.L/sim.U
        spoke[] = points(θ[])
        yield()   # hand control back to GLMakie event loop
    end
end 