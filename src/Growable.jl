# =========================================================================
# GROWABLE PER-PARTICLE BUFFERS
#
# Scratch arrays owned by the time stepper and the interpolators (stage states, neighbour gathers,
# MOOD masks, cached gradients and moment matrices) grow with the particle count and must live
# where the grid lives. A `Growable` holds such an array behind a mutable binding, so it can be
# reallocated when the capacity is exceeded whatever the array type (`resize!` only exists for
# `Vector`). Kernel bodies never see the wrapper: `Adapt.adapt` (and therefore every
# `foreach_particle` launch, see Backend.jl) replaces it by the array it holds.
# =========================================================================
using Adapt

export Growable, CapacityError

"""
    Growable{T, V <: AbstractVector{T}} <: AbstractVector{T}
    Growable(v::AbstractVector)
    Growable{T}(ArrayType)

A per-particle buffer that can be reallocated to a larger capacity on any backend. `parent(g)` is the
array currently held (`V`, e.g. a `Vector` or a `ROCArray`); `Growable{T}(ArrayType)` starts empty
with storage of type `ArrayType{T}`. The wrapper indexes like its parent on the host and disappears
under `Adapt.adapt`, so kernel bodies receive the raw array. `reserve!` fixes the capacity: a fixed
buffer throws a `CapacityError` instead of growing.
"""
mutable struct Growable{T, V <: AbstractVector{T}} <: AbstractVector{T}
    data::V
    fixed::Bool
end
Growable(v::V) where {T, V <: AbstractVector{T}} = Growable{T, V}(v, false)
Growable{T}(::Type{AT}) where {T, AT <: AbstractArray} = Growable(AT{T}(undef, 0))

"""
    CapacityError <: Exception

Thrown by `ensure_capacity!` when a buffer whose capacity was fixed by `reserve!` would have to grow:
the run has outgrown its memory budget (more particles, or more neighbours per particle, than reserved).
"""
struct CapacityError <: Exception
    eltype::DataType
    capacity::Int
    requested::Int
end
Base.showerror(io::IO, e::CapacityError) = print(io, "CapacityError: a reserved buffer of ", e.capacity, " × ",
    e.eltype, " would have to grow to ", e.requested, " elements; the run has outgrown its memory budget (see `reserve!`)")

Base.parent(g::Growable) = g.data
Base.size(g::Growable) = size(g.data)
Base.IndexStyle(::Type{Growable{T, V}}) where {T, V} = IndexStyle(V)
Base.@propagate_inbounds Base.getindex(g::Growable, i::Int) = g.data[i]
Base.@propagate_inbounds Base.setindex!(g::Growable, v, i::Int) = (g.data[i] = v)
Base.view(g::Growable, I...) = view(g.data, I...)
Base.fill!(g::Growable, v) = (fill!(g.data, v); g)
Base.similar(g::Growable, ::Type{S}, dims::Dims) where {S} = similar(g.data, S, dims)
# Whole-array operations go to the held array, so they work for device storage (no scalar indexing).
Base.copy(g::Growable) = copy(g.data)
Base.collect(g::Growable) = collect(g.data)
Base.Array(g::Growable) = Array(g.data)
Base.Array{S}(g::Growable) where {S} = Array{S}(g.data)
Base.Array{S, 1}(g::Growable) where {S} = Array{S, 1}(g.data)
Base.copyto!(dest::Growable, doffs::Integer, src::AbstractArray, soffs::Integer, n::Integer) = (copyto!(dest.data, doffs, src, soffs, n); dest)
Base.copyto!(dest::AbstractArray, doffs::Integer, src::Growable, soffs::Integer, n::Integer) = copyto!(dest, doffs, src.data, soffs, n)
Base.copyto!(dest::Growable, doffs::Integer, src::Growable, soffs::Integer, n::Integer) = (copyto!(dest.data, doffs, src.data, soffs, n); dest)
Base.copyto!(dest::Growable, src::AbstractArray) = (copyto!(dest.data, src); dest)
Base.copyto!(dest::AbstractArray, src::Growable) = copyto!(dest, src.data)
Base.copyto!(dest::Growable, src::Growable) = (copyto!(dest.data, src.data); dest)
Base.show(io::IO, g::Growable) = (print(io, "Growable("); show(io, g.data); print(io, ")"))
Base.show(io::IO, m::MIME"text/plain", g::Growable) = (print(io, "Growable wrapping "); show(io, m, g.data))

Adapt.adapt_structure(to, g::Growable) = adapt(to, g.data)
KA.get_backend(g::Growable) = KA.get_backend(g.data)

"""
    ensure_capacity!(g::Growable, n)

Makes room for `n` elements (growing by 25 % headroom like the `Vector` buffers). A `Vector` is resized in
place; any other array is reallocated with `similar` and its current contents copied over. The elements
beyond the old length are undefined, as after `resize!`. A buffer whose capacity was fixed by `reserve!`
throws a `CapacityError` instead of growing.
"""
@inline function ensure_capacity!(g::Growable, req_capacity::Int)
    if length(g.data) < req_capacity
        g.fixed && throw(CapacityError(eltype(g.data), length(g.data), req_capacity))
        g.data = _grown(g.data, ceil(Int, req_capacity * 1.25))
    end
    return nothing
end

_grown(v::Vector, cap::Int) = resize!(v, cap)
function _grown(v::AbstractVector, cap::Int)
    w = similar(v, cap)
    isempty(v) || copyto!(w, 1, v, 1, length(v))
    return w
end
