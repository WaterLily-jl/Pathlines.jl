using WaterLily, CUDA, Pathlines, GLMakie

function make_particleplot(; N=Int(1e4), life=UInt(255), name="particleplot.mp4",
                             U=1, L=64, T=Float32,
                             mem=CUDA.functional() ? CUDA.CuArray : Array)
    k = T(π/L)
    function u0(i,xy)
        x,y = @. (xy-1.5f0)*k            # scaled coordinates
        i==1 && return -U*sin(x)*cos(y)  # u_x
        i==2 && return  U*cos(x)*sin(y)  # u_y
        zero(k)
    end
    sim = Simulation((L,L),(0, 0), L; U, u0, ν=L*U/1e3, T, mem)

    # Set up Particles and CPU visualization arrays
    p = Particles(N,sim.flow.σ;mem,life)
    v = ParticleViz(p,sim.flow.Δt[1],width=1)

    #Set up figure
    fig = GLMakie.Figure()
    ax = GLMakie.Axis(fig[1, 1]; autolimitaspect=1)
    GLMakie.scatter!(ax, v.opos, markersize=v.omag, rotation=v.odir, marker=:circle)

    # Record video
    @time GLMakie.record(fig,name,0.01:0.05:2.0) do t
        while sim_time(sim)<t
            sim_step!(sim)
            Pathlines.update!(p,sim)
        end
        notify!(v,p,sim.flow.Δt[end-1])
    end
end

make_particleplot()