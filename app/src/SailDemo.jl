module SailDemo

using WaterLily, BiotSavartBCs, GLMakie, StaticArrays, ParametricBodies, LilyPad, Pathlines
import ParametricBodies: tangent, hat, AbstractParametricBody, curve_props, perp

function sail(; p=6, Δt=2, T=Float32, mem=Array, β=0.)
    m = 2^p; n = 2m; β = T(β)
    length, edge = T(4m/5), SVector{2,T}(m/4, 4m/7)
    function new_body(θ,c1,c2)
        R = SMatrix{2,2,T}(cos(θ), -sin(θ), sin(θ), cos(θ))
        pnts = length*R*SMatrix{2,4,T}(0, 0, 0.25, c1, 0.75, c2, 1, 0) .+ edge
        spline = BSplineCurve(pnts,degree=3)
        dotS(u,t) = β*hat(tangent(spline,u,t))
        return SailBody(ParametricBody(spline;dotS,thk=1.5,boundary=false))
    end
    return LilyBiotSim((n, m), (1, 0), length; ν=0, Δt, body=new_body(0, 0, 0), mem, T, ϵ=0.5), new_body
end

struct SailBody{B<:AbstractParametricBody} <: AbstractParametricBody
    inner::B
end
Base.getproperty(s::SailBody, f::Symbol) = f === :inner ? getfield(s,:inner) : getproperty(getfield(s,:inner), f)
function WaterLily.measure(body::SailBody, x, t; fastd²=Inf)
    d,n,dotS = curve_props(body.inner, x, t; fastd²)
    d^2 > fastd² && return d, zero(x), zero(x)
    dξdt = n[2]>0 ? dotS : zero(x)  # one-sided: sail only pushes on the leeward side
    return (d, n, dξdt)
end

normal(curve,u,t=0) = perp(hat(tangent(curve,u,t)))

function segment_force(p, curve, u0, u1, δ=2)
    ds,u = √sum(abs2,curve(u1)-curve(u0)), (u1+u0)/2
    x,n = curve(u), normal(curve,u)
    -(WaterLily.interp(x+δ*n,p)-WaterLily.interp(x-δ*n,p)) * n * ds
end

function project_modes(p::Array{T}, curve, θ, u) where T
    ey = SA{T}[sin(θ), cos(θ)]
    f1 = f2 = zero(T)
    for i in 1:length(u)-1
        F = segment_force(p, curve, u[i], u[i+1])
        dp_ds = F'ey
        umid = (u[i]+u[i+1])/2
        f1 += dp_ds * 3umid*(1-umid)^2
        f2 += dp_ds * 3umid^2*(1-umid)
    end
    f1, f2
end

function julia_main()::Cint
    mem = Array
    sim, new_body = sail(; mem, β=1.0)
    p = sim.flow.p |> Array
    fig, ax = viz!(sim; remeasure=true, verbose=false, N=1_000, fadetau=0.5, mem,
                   colormap=:vik, colorrange=(0.25, 1.75))

    Narrows = 25
    u = range(0f0, 1f0, length=Narrows+1)
    x = [sim.body.curve((u[i]+u[i+1])/2) for i in 1:Narrows] |> Observable
    df = zeros(SVector{2,Float32}, Narrows) |> Observable
    arrows2d!(ax, x, df, color=:white)
    tail = sim.body.curve(1f0) |> Observable
    vec = (tail[] - sim.body.curve(0f0)) |> Observable
    arrows2d!(ax, tail, vec, color=:red)

    AoA, tension = Ref(0f0), Ref(10f0)
    f1, f2 = Ref(0f0), Ref(0f0)
    on(events(fig).keyboardbutton) do event
        event.action == Keyboard.press || return
        if     event.key == Keyboard.up;    AoA[] -= π/360
        elseif event.key == Keyboard.down;  AoA[] += π/360
        elseif event.key == Keyboard.left;  tension[] *= 0.9f0
        elseif event.key == Keyboard.right; tension[] *= 1.1f0
        else; return; end
    end

    while events(fig).window_open[]
        viz_step!(fig, sim)
        copyto!(p, sim.flow.p)
        f1r, f2r = project_modes(p, sim.body.curve, AoA[], u)
        f1[] += (f1r - f1[])/8; f2[] += (f2r - f2[])/8
        c1, c2 = SA[0.889f0 -0.222f0; -0.222f0 0.889f0] * SA[f1[], f2[]] / (8f0 * tension[])
        sim.body = new_body(AoA[], c1, c2)
        for i in 1:Narrows
            x.val[i] = sim.body.curve((u[i]+u[i+1])/2)
            df.val[i] = 0.75f0*df.val[i] + segment_force(p, sim.body.curve, u[i], u[i+1])
        end
        tail.val = sim.body.curve(1f0)
        vec.val = 2tension[] * hat(tail.val - sim.body.curve(0f0))
        notify(x); notify(df); notify(tail); notify(vec)
    end
    return 0
end

end # module SailDemo
