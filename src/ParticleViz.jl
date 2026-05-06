import GLMakie

struct ParticleViz
    pos;  opos   # CPU position buffer + Observable
    pos⁰         # CPU previous-position buffer
    mag;  omag   # CPU marker-size (Vec2f) buffer + Observable
    dir;  odir   # CPU rotation-angle buffer + Observable
    velmag::Function
end

"""
    ParticleViz(p::Particles, Δt; scale=20, minsize=2, width=2)

Build a CPU-side Makie visualization wrapper for a `Particles` instance.
Creates `GLMakie.Observable` arrays for position, velocity magnitude
(as `Vec2f` marker size), and direction (rotation angle), suitable for
use with `GLMakie.scatter!`.
"""
function ParticleViz(p::Particles, Δt; scale=20, minsize=2, width=2)
    vizvelmag(x, x0, Δt) = velmag(x, x0, Δt; scale, minsize, width)
    pos  = Array(p.position)
    pos⁰ = Array(p.position⁰)
    mag  = vizvelmag.(pos, pos⁰, Δt)
    dir  = veldir.(pos, pos⁰)
    ParticleViz(pos,  GLMakie.Observable(pos),
                pos⁰,
                mag,  GLMakie.Observable(mag),
                dir,  GLMakie.Observable(dir),
                vizvelmag)
end

"""
    notify!(v::ParticleViz, p::Particles, Δt)

Copy updated particle state from `p` into the CPU buffers of `v`,
recompute velocity magnitude and direction, and notify all Observables
to trigger a plot update.
"""
function notify!(v::ParticleViz, p::Particles, Δt)
    copyto!(v.pos,  p.position)
    copyto!(v.pos⁰, p.position⁰)
    v.mag .= v.velmag.(v.pos, v.pos⁰, Δt)
    v.dir .= veldir.(v.pos, v.pos⁰)
    GLMakie.notify(v.opos)
    GLMakie.notify(v.omag)
    GLMakie.notify(v.odir)
end

velmag(x, x0, Δt; scale=20, minsize=2, width=2) =
    (dx = x - x0; GLMakie.Vec2f(minsize + scale*√(dx'*dx)/Δt, width))
veldir(x, x0) = (dx = x - x0; atan(dx[2], dx[1]))
