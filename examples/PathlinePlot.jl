using WaterLily,CUDA,Pathlines,GLMakie

function make_pathplot(L=64,T=Float32, U=1,mem=Array)
    k = T(π/L)
    function uλ(i,xy)
        x,y = @. (xy-1.5f0)*k            # scaled coordinates
        i==1 && return -U*sin(x)*cos(y)  # u_x
        i==2 && return  U*cos(x)*sin(y)  # u_y
        zero(k)
    end
    
    # Initialize simulation
    sim = Simulation((L, L), (0, 0), L; U, uλ, ν=U*L/1e3, T, mem)

    # Initialize plot
    color = :black
    fig = GLMakie.Figure(size=(800,800));
    ax = GLMakie.Axis(fig[1, 1]; limits=(2, L+1, 2, L+1),autolimitaspect = 1)
    GLMakie.Box(fig,width=1600,height=1600;color)
    GLMakie.hidedecorations!(ax)

    # Set up Particles
    p = Particles(1024,sim.flow.σ;mem) 
    dat = tuple.(p.position⁰,p.position) |> Array   

    @time GLMakie.record(fig,"pathlineplot.mp4",0.01:0.05:2.0) do t
        Box(fig,width=1600,height=1600,color=(color,0.2))
        while sim_time(sim)<t
            sim_step!(sim)
            Pathlines.update!(p,sim)
            copyto!(dat,tuple.(p.position⁰,p.position))
            GLMakie.linesegments!(ax,dat,linewidth=4,color=:white)
        end
    end
end

make_pathplot()