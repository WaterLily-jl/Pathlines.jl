using WaterLily, LilyPad, StaticArrays, KernelAbstractions

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
# Bounds in WaterLily coordinates: interior cells span [0.5, size-2.5] in each dimension
Particles(N::Int,A::AbstractArray{T,D};life=UInt(255),mem=Array) where {D,T} =
    Particles(N,SVector{D,T}(T(0.5) for _ in 1:D),SVector{D,T}(T(size(A,i))-T(2.5) for i in 1:D);life,mem)

spawn(lower::SVector{D,T},upper::SVector{D,T}) where {D,T} = rand(SVector{D,T}).*(upper-lower)+lower

"""
    update!(p::Particles, sim::Simulation)

Update the state of each particle in `p` using a given flow `sim`.
Uses a 2nd-order Crank-Nicolson integrator (via `LilyPad.departure`) and
KernelAbstractions to run multi-threaded on CPUs and GPUs.
"""
update!(p::Particles,sim::Simulation) = _update!(get_backend(p.age),64)(p.age,p.position,p.position⁰,
    sim.flow.u⁰,sim.flow.u,sim.flow.Δt[end-1],p.life,p.lower,p.upper,ndrange=length(p.age))

@kernel function _update!(age,x,x⁰,@Const(u⁰),@Const(u),@Const(Δt),@Const(life),@Const(lower),@Const(upper))
    i = @index(Global)
    x⁰[i] = x[i]
    x[i] = LilyPad.departure(x⁰[i], u, u⁰, -Δt)
    age[i] += one(eltype(age))
    if(age[i]==life || any(x[i] .< lower) || any(x[i] .> upper))
        age[i] = zero(eltype(age))
        x[i] = spawn(lower,upper)
        x⁰[i] = x[i]
    end
end
