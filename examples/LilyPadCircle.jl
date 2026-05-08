using WaterLily, LilyPad, StaticArrays, BiotSavartBCs, CUDA, Pathlines, GLMakie
import ColorSchemes: colorschemes

function make_lilypad_circle(; N=Int(2e4), life=UInt(200), Δt=1.5, T=Float32,
                               mem=CUDA.functional() ? CUDA.CuArray : Array)
    n, m = 2^8, 2^7   # 256×128
    radius, center, ω = T(m / 8), SVector{2,T}(m/2-1, m/2-1), zero(T)

    body=AutoBody((x, t) -> √(x'x) - radius,
                  RigidMap(center,ω; ω))

    sim = LiliBiotSim((n, m), (1, 0), 2radius; ν=0, Δt, body, mem, T)

    bgcolor = colorschemes[:starrynight].colors[1]
    bodycolor = colorschemes[:starrynight].colors[end]
    viz = PathViz(sim; N, life, mem, body=true, bgcolor, figsize=(1024, 512),
                  colormap=:starrynight, colorrange=(0, 2), bodycolor)

    function points(θ) 
        r = radius*SVector{2, T}(cos(θ), sin(θ))
        c = center .+ 0.5
        [Point2f((c+r)...),Point2f((c-r)...)]
    end
    return sim,viz,points
end

begin
    sim,viz,points = make_lilypad_circle(N=2_000);
    Pathlines.step!(viz, sim, 0.1; remeasure=false)
    θ = Ref(0f0); dt = 0.1f0
    spoke = Observable(points(θ[]))
    lines!(viz.ax, spoke, color=:black, linewidth=4)    
    display(viz.fig)
end

begin
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

    while events(viz.fig).window_open[]
        t = sim_time(sim)+dt
        Pathlines.step!(viz, sim, t; remeasure = false)
        θ[] += ω[]*dt*sim.L/sim.U
        spoke[] = points(θ[])
        yield()   # hand control back to GLMakie event loop
    end
end