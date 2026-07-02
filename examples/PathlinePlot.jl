"""
Smoke test for the Pathlines.jl + WaterLily.viz! integration.

Sets up a temporary Julia environment with dev versions of WaterLily and
Pathlines.jl from the workspace, runs a 2D circle flow, and saves pathline
frames as PNGs to the current directory.
"""

import Pkg

# ── persistent project environment (avoids recompiling every run) ─────────────
env_dir = joinpath(@__DIR__, ".test_pathlines_env")
Pkg.activate(env_dir)

repo_root = dirname(@__DIR__)   # WaterLily-Examples/../  = .../GitHub
Pkg.develop(path=joinpath(repo_root, "WaterLily"))
Pkg.develop(path=joinpath(repo_root, "LilyPad.jl"))   # unregistered dep of Pathlines
Pkg.develop(path=joinpath(repo_root, "Pathlines.jl"))
# GLMakie and StaticArrays come in as Pathlines deps; add extras just in case
Pkg.add(["GLMakie", "StaticArrays"])

# ── simulation ────────────────────────────────────────────────────────────────
using WaterLily, GLMakie, Pathlines, StaticArrays

function circle(; L=2^6, Re=250, U=1)
    c = SA[3L÷2, L]          # centre at (1.5L, L)
    body = AutoBody((x,t) -> √sum(abs2, x .- c) - L÷4)
    Simulation((3L, 2L), (U,0), L÷4; ν=U*(L÷4)/Re, body)
end

println("\n=== hook status ===")
println("_pathlines_viz_hook set: ", !isnothing(WaterLily._pathlines_viz_hook[]))

sim = circle()

println("\n=== running viz! (5 frames saved as frame_0001..0005.png) ===")
viz!(sim;
    duration = 5,
    step     = 1.0,
    N        = 5_000,      # fewer particles for faster test
    img      = "frame_%04d.png",   # per-frame PNG, renders offscreen (no display needed)
    verbose  = true,
)

println("\nDone — check frame_0001.png ... frame_0005.png in $(pwd())")
