using CUDA, StaticArrays
import GLMakie: Box,linesegments!,Figure,Axis,hidedecorations!

function addlayer!(fig,dat)
    Box(fig,width=1600,height=1600,color=(:green,0.1))
    linesegments!(ax,dat,markersize=4,color=:white)
    return fig
end

function TGV(L; Re=1e5, T=Float32, mem=Array)
    # Define vortex size, velocity, viscosity
    U = 1; ν = U*L/Re
    # Taylor-Green-Vortex initial velocity field
    function uλ(i,xy)
        x,y = @. (xy-1.5)*π/L           # scaled coordinates
        i==1 && return -U*sin(x)*cos(y) # u_x
        i==2 && return  U*cos(x)*sin(y) # u_y
    end
    # Initialize simulation
    return Simulation((L, L), (0, 0), L; U, uλ, ν, T, mem)
end

L = 32; sim = TGV(L,mem=CuArray); sim_step!(sim,0.01);

fig = Figure(resolution=(800,800));
ax = Axis(fig[1, 1]; limits=(2, L+1, 2, L+1),autolimitaspect = 1)
Box(fig,width=1600,height=1600,color=:green)
hidedecorations!(ax)
fig

lower = SVector{2,Float32}(2,2)
upper = SVector{2,Float32}(L+1,L+1)
p = Particles(1024,lower,upper,mem=CuArray)
update!(p,sim)
dat = tuple.(p.position⁰,p.position) |> Array
addlayer!(fig,dat)

run(it) = for _ in 1:it
    update!(p,sim)
    copyto!(dat,tuple.(p.position⁰,p.position))
    addlayer!(fig,dat)
    display(fig)
end
run(20)