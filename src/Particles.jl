using WaterLily,StaticArrays,EllipsisNotation,KernelAbstractions

struct Particles{D,V<:AbstractArray,S<:AbstractArray}
    position::V
    position⁰::V
    age::S
    lower::SVector{D}
    upper::SVector{D}
    life::UInt
end
Base.show(io::IO, ::MIME"text/plain", z::Particles) = show(io,MIME("text/plain"),z.position)

function Particles(N::Int,lower::SVector{D,T},upper::SVector{D,T};life=UInt(255),mem=Array) where {D,T}
    position = [spawn(lower,upper) for _ in 1:N] |> mem
    age = rand(UInt,N) .% life |> mem
    Particles{D,typeof(position),typeof(age)}(position, copy(position), age, lower, upper, life)
end
Particles(N::Int,A::AbstractArray{T,D};life=UInt(255),mem=Array) where {D,T} = 
    Particles(N,SVector{D,T}(1.5 for i in 1:D),SVector{D,T}(size(A,i)-0.5 for i in 1:D);life,mem)

spawn(lower::SVector{D,T},upper::SVector{D,T}) where {D,T} = rand(SVector{D,T}).*(upper-lower)+lower

"""
    update!(p:Particles,sim:Simulation)

    Update the state of each particle in `p` using a given flow `sim`.
    Uses KernelAbstractions to run multi-threaded on CPUs and GPUs.
"""
update!(p::Particles,sim::Simulation) = _update!(get_backend(p.age),64)(p.age,p.position,p.position⁰,
    sim.flow.u⁰,sim.flow.u,sim.flow.Δt[end-1],p.life,p.lower,p.upper,ndrange=length(p.age))
@kernel function _update!(age,x,x⁰,@Const(u⁰),@Const(u),@Const(Δt),@Const(life),@Const(lower),@Const(upper))
    i = @index(Global)
    # Use sim to integrate to new position and update other states
    x⁰[i] = x[i]
    x[i] += ∫uΔt(x⁰[i],u⁰,u,Δt)
    age[i] += one(eltype(age))

    # Enforce bounds
    if(age[i]==life || x[i]<=lower || x[i]>=upper)
        age[i] = zero(eltype(age))
        x[i] = spawn(lower,upper)
        x⁰[i] = x[i]
    end
end

function ∫uΔt(x⁰, u⁰, u, Δt)
    v₁ = interp(x⁰,u⁰)
    v₂ = (interp(x⁰+Δt*v₁/2,u⁰)+interp(x⁰+Δt*v₁/2,u))/2
    v₃ = (interp(x⁰+Δt*v₂/2,u⁰)+interp(x⁰+Δt*v₂/2,u))/2
    v₄ = interp(x⁰+Δt*v₃,u)
    return Δt*(v₁+2v₂+2v₃+v₄)/6 # RK4
end

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
    return SVector{D,T}(interp(x+shift(i),@view(varr[..,i])) for i in 1:D)
end
