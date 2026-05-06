"""
LilyPadCircle.jl

Pathline-style visualization of a 2D circle in external flow using
LilyBiotSim + Pathlines particle advection.

Run with:
    julia --project=. examples/LilyPadCircle.jl
"""

using WaterLily, LilyPad, BiotSavartBCs, CUDA, Pathlines, GLMakie

function make_lilypad_circle(; N=Int(2e4), life=UInt(200), name="lilypad_circle.mp4",
                               mem=CUDA.functional() ? CUDA.CuArray : Array,
                               T=Float32)
    n, m = 2^8, 2^7   # 256×128
    radius, center = T(m / 8), T(m / 2 - 1)
    sdf(x, t) = sqrt(sum(abs2, x .- center)) - radius

    sim = LilyBiotSim((n, m), (1, 0), 2radius; ν=0, body=AutoBody(sdf), mem, T)

    viz = PathViz(sim; N, life, mem, body=true, bgcolor=:black, figsize=(1024, 512),
                  colormap=:plasma, colorrange=(0, 2), bodycolor=:white)

    @time GLMakie.record(viz.fig, name, 0.1:0.1:30.0) do t
        Pathlines.step!(viz, sim, t; remeasure=false)
    end
end

make_lilypad_circle(N=2_000)
