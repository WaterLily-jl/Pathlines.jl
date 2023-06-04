using Pathlines,BenchmarkTools,WaterLily
import CUDA
function TGV(L; Re=1e5, T=Float32, mem=Array)
    # Define vortex size, velocity, viscosity
    U = 1; ν = U*L/Re
    # Taylor-Green-Vortex initial velocity field
    function uλ(i,xyz)
        x,y,z = @. (xyz-1.5)*π/L                # scaled coordinates
        i==1 && return -U*sin(x)*cos(y)*cos(z) # u_x
        i==2 && return  U*cos(x)*sin(y)*cos(z) # u_y
        return 0.                              # u_z
    end
    # Initialize simulation
    return Simulation((L, L, L), (0, 0, 0), L; U, uλ, ν, T, mem)
end
function testParticles(mem=Array,N=96,T=Float32)
    CUDA.allowscalar(false)
    vortex = TGV(N;T,mem);
    WaterLily.mom_step!(vortex.flow,vortex.pois);
    p = Particles(Int(1e6),vortex.flow.p;mem)
    update!(p,vortex);
    @btime update!($p,$vortex) samples=1 evals=10;
end