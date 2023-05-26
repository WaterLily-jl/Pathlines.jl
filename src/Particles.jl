using WaterLily,StaticArrays,CUDA,EllipsisNotation

struct Particles{D,V<:AbstractArray,S<:AbstractArray}
    position::V
    age::S
    lower::SVector{D}
    upper::SVector{D}
    life::UInt8
end
function Particles(N::Int,lower::SVector{D,T},upper::SVector{D,T};life::UInt8=0xff,mem=Array) where {D,T}
    position = [spawn(lower,upper) for _ in 1:N] |> mem
    age = rand(UInt8,N) .% life |> mem
    Particles{D,typeof(position),typeof(age)}(position, age, lower, upper, life)
end
Base.show(io::IO, ::MIME"text/plain", z::Particles) = show(io,MIME("text/plain"),z.position)

spawn(lower::SVector{D,T},upper::SVector{D,T}) where {D,T} = rand(SVector{D,T}).*(upper-lower)+lower

function interp(x::SVector{D,T}, arr::AbstractArray{T,D}) where {D,T}
    """
    Linear interpolation from array `arr` at coordinate `x`.
    Note: This routine works for any number of dimensions.
    """
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

function ∫uΔt(x, u⁰, u, Δt)
    v⁰ = interp(x,u⁰);  dx = Δt*v⁰  # predict
    v = interp(x+dx,u); Δt*(v+v⁰)/2 # correct
end
bound(age,x,life,lower,upper) = ifelse(age==life || x<lower || x>upper, spawn(lower,upper), x)

function update!(p::Particles,sim::Simulation)
    # update position using the simulation and age by one step
    Δt = last(sim.flow.Δt)
    p.position .+= ∫uΔt.(p.position,Ref(sim.flow.u⁰),Ref(sim.flow.u),Ref(Δt))
    Δage = one(eltype(p.age))
    p.age .+= Ref(Δage)

    # Enforce bounds
    p.position .= bound.(p.age,p.position,Ref(p.life),Ref(p.lower),Ref(p.upper))
    p.age .= p.age .% Ref(p.life)
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

function test()
    CUDA.allowscalar(false)
    D,N,T = 3,32,Float32
    vortex = TGV(N;T,mem=CuArray);
    WaterLily.mom_step!(vortex.flow,vortex.pois);

    lower = SVector{D,T}(1.5 for _ in 1:D)
    upper = SVector{D,T}(N-0.5 for _ in 1:D)
    p = Particles(Int(1e6),lower,upper,mem=CuArray)
    update!(p,vortex)
    @time update!(p,vortex)
end
