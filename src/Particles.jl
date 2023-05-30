using WaterLily,StaticArrays,CUDA,EllipsisNotation,KernelAbstractions

struct Particles{D,V<:AbstractArray,S<:AbstractArray}
    position::V
    position⁰::V
    age::S
    lower::SVector{D}
    upper::SVector{D}
    life::UInt
end
function Particles(N::Int,lower::SVector{D,T},upper::SVector{D,T};life=UInt(255),mem=Array) where {D,T}
    position = [spawn(lower,upper) for _ in 1:N] |> mem
    age = rand(UInt,N) .% life |> mem
    Particles{D,typeof(position),typeof(age)}(position, copy(position), age, lower, upper, life)
end
Base.show(io::IO, ::MIME"text/plain", z::Particles) = show(io,MIME("text/plain"),z.position)

spawn(lower::SVector{D,T},upper::SVector{D,T}) where {D,T} = rand(SVector{D,T}).*(upper-lower)+lower

"""
    interp(x::SVector, arr::AbstractArray)

    Linear interpolation from array `arr` at index-coordinate `x`.
    Note: This routine works for any number of dimensions.
"""
function interp(x::SVector{D,T}, arr::AbstractArray{T,D}) where {D,T}
    # Index below the interpolation coordinate and the difference
    i = floor.(Int,x); y = x.-i
    
    # CartesianIndices around x 
    I = CartesianIndex(i...); R = I:I+oneunit(I)

    # Linearly weighted sum over arr[R] (in serial)
    s = zero(T)
    @fastmath @inbounds @simd for J in R
        weight = prod(@. ifelse(J.I==I.I,1-y,y))
        s += arr[J]*weight
    end
    return s
end

function interp(x::SVector{D,T}, varr::AbstractArray{T}) where {D,T}
    # Shift to align with each staggered grid component and interpolate
    @inline shift(i) = SVector{D,T}(ifelse(i==j,0.5,0.) for j in 1:D)
    return SVector{D,T}(interp(x-shift(i),@view(varr[..,i])) for i in 1:D)
end

function ∫uΔt(x⁰, u⁰, u, Δt)
    v₁ = interp(x⁰,u⁰)
    v₂ = interp(x⁰+Δt*v₁,u)
    return Δt*(v₁+v₂)/2 # RK2
    # v₂ = (interp(x⁰+Δt*v₁/2,u⁰)+interp(x⁰+Δt*v₁/2,u))/2
    # v₃ = (interp(x⁰+Δt*v₂/2,u⁰)+interp(x⁰+Δt*v₂/2,u))/2
    # v₄ = interp(x⁰+Δt*v₃,u)
    # return Δt*(v₁+2v₂+2v₃+v₄)/6 # RK4
end

@kernel function _update!(age,x,x⁰,@Const(u⁰),@Const(u),@Const(Δt),@Const(life),@Const(lower),@Const(upper))
    i = @index(Global)
    # Use sim to integrate to new position and update other states
    x⁰[i] = x[i]
    x[i] += ∫uΔt(x⁰[i],u⁰,u,Δt)
    age[i] += one(eltype(age))

    # Enforce bounds
    if(age[i]==life || x[i]<lower || x[i]>upper)
        age[i] = zero(eltype(age))
        x[i] = spawn(lower,upper)
        x⁰[i] = x[i]
    end
end

"""
    update!(p:Particles,sim:Simulation)

    Update the state of each particle in `p` using a given flow `sim`.
    Uses KernelAbstractions to run multi-threaded on CPUs and GPUs.
"""
function update!(p::Particles,sim::Simulation)
    # KernelAbstractions loop through all the Particles
    Δt = sim.flow.Δt[end-1]
    _update!(get_backend(p.age),64)(p.age,p.position,p.position⁰,
        sim.flow.u⁰,sim.flow.u,Δt,p.life,p.lower,p.upper,ndrange=length(p.age))
    return p
end

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

function test(mem=Array)
    CUDA.allowscalar(false)
    D,N,T = 3,32,Float32
    vortex = TGV(N;T,mem);
    WaterLily.mom_step!(vortex.flow,vortex.pois);

    lower = SVector{D,T}(1.5 for _ in 1:D)
    upper = SVector{D,T}(N-0.5 for _ in 1:D)
    p = Particles(Int(1e6),lower,upper;mem)
    update!(p,vortex)
    @time update!(p,vortex)
end
