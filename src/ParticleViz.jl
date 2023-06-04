import GLMakie
struct ParticleViz
    pos; opos
    pmag; mag; omag
    pdir; dir; odir
end
function ParticleViz(p::Particles,Δt)
    pos = p.position |> Array; opos = GLMakie.Observable(pos)
    pmag = velmag.(p.position,p.position⁰,Δt)
    mag = pmag |> Array; omag = GLMakie.Observable(mag)
    pdir = veldir.(p.position,p.position⁰)
    dir = pdir |> Array; odir = GLMakie.Observable(dir)
    ParticleViz(pos,opos,pmag,mag,omag,pdir,dir,odir) 
end
function notify!(v::ParticleViz,p::Particles,Δt)
    copyto!(v.pos,p.position); notify(v.opos)
    v.pmag .= velmag.(p.position,p.position⁰,Δt)
    copyto!(v.mag,v.pmag); notify(v.omag)
    v.pdir .= veldir.(p.position,p.position⁰)
    copyto!(v.dir,v.pdir); notify(v.odir)
end
velmag(x,x0,Δt) = (dx=x-x0; GLMakie.Vec2f(2+20√(dx'*dx)/Δt,2))
veldir(x,x0) = (dx=x-x0; atan(dx[2],dx[1]))
