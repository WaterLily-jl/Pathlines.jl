import GLMakie
struct ParticleViz
    pos; opos
    pmag; mag; omag
    pdir; dir; odir
    velmag::Function
end

function ParticleViz(p::Particles, Δt; scale=20, minsize=2, width=2)
    pos = p.position |> Array; opos = GLMakie.Observable(pos)
    vizvelmag(x,x0,Δt) = velmag(x,x0,Δt; scale, minsize, width)
    pmag = vizvelmag.(p.position, p.position⁰, Δt)
    mag = pmag |> Array; omag = GLMakie.Observable(mag)
    pdir = veldir.(p.position, p.position⁰)
    dir = pdir |> Array; odir = GLMakie.Observable(dir)
    ParticleViz(pos, opos, pmag, mag, omag, pdir, dir, odir, vizvelmag)
end

function notify!(v::ParticleViz, p::Particles, Δt)
    copyto!(v.pos, p.position); notify(v.opos)
    v.pmag .= v.velmag.(p.position, p.position⁰, Δt)
    copyto!(v.mag, v.pmag); notify(v.omag)
    v.pdir .= veldir.(p.position, p.position⁰)
    copyto!(v.dir, v.pdir); notify(v.odir)
end

velmag(x, x0, Δt; scale=20, minsize=2, width=2) = (dx = x - x0; GLMakie.Vec2f(minsize+scale*√(dx'*dx)/Δt,width))
veldir(x, x0) = (dx = x - x0; atan(dx[2], dx[1]))
