using WaterLily, LilyPad, StaticArrays, BiotSavartBCs, CUDA, Pathlines, GLMakie

function make_lilypad_circle(; N=Int(2e4), life=UInt(200), name="lilypad_circle.mp4",
                               mem=CUDA.functional() ? CUDA.CuArray : Array,
                               T=Float32)
    n, m = 2^8, 2^7   # 256×128
    radius, center, ω = T(m / 8), SVector{2,T}(m/2-1, m/2-1), zero(T)

    body=AutoBody((x, t) -> √(x'x) - radius,
                  RigidMap(center,ω; ω))

    sim = LilyBiotSim((n, m), (1, 0), 2radius; ν=0, body, mem, T)

    viz = PathViz(sim; N, life, mem, body=true, bgcolor=:black, figsize=(1024, 512),
                  colormap=:starrynight, colorrange=(0, 2), bodycolor=:white)

    function points(θ) 
        r = radius*SVector{2, T}(cos(θ), sin(θ))
        [Point2f((center+r)...),Point2f((center-r)...)]
    end
    return sim,viz,points
end

sim,viz,points = make_lilypad_circle(N=2_000);
Pathlines.step!(viz, sim, 0.1; remeasure=false)
θ = Ref(0f0); dt = 0.1f0
spoke = Observable(points(θ[]))
lines!(viz.ax, spoke, color=:red, linewidth=2)
display(viz.fig)

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
