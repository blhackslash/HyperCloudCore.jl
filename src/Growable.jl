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

export Growable

"""
    Growable{T, V <: AbstractVector{T}} <: AbstractVector{T}
    Growable(v::AbstractVector)
    Growable{T}(ArrayType)

A per-particle buffer that can be reallocated to a larger capacity on any backend. `parent(g)` is the
array currently held (`V`, e.g. a `Vector` or a `ROCArray`); `Growable{T}(ArrayType)` starts empty
with storage of type `ArrayType{T}`. The wrapper indexes like its parent on the host and disappears
under `Adapt.adapt`, so kernel bodies receive the raw array.
"""
mutable struct Growable{T, V <: AbstractVector{T}} <: AbstractVector{T}
    data::V
end
Growable{T}(::Type{AT}) where {T, AT <: AbstractArray} = Growable(AT{T}(undef, 0))

Base.parent(g::Growable) = g.data
Base.size(g::Growable) = size(g.data)
Base.IndexStyle(::Type{Growable{T, V}}) where {T, V} = IndexStyle(V)
Base.@propagate_inbounds Base.getindex(g::Growable, i::Int) = g.data[i]
Base.@propagate_inbounds Base.setindex!(g::Growable, v, i::Int) = (g.data[i] = v)
Base.view(g::Growable, I...) = view(g.data, I...)
Base.fill!(g::Growable, v) = (fill!(g.data, v); g)
Base.similar(g::Growable, ::Type{S}, dims::Dims) where {S} = similar(g.data, S, dims)
Base.show(io::IO, g::Growable) = (print(io, "Growable("); show(io, g.data); print(io, ")"))
Base.show(io::IO, m::MIME"text/plain", g::Growable) = (print(io, "Growable wrapping "); show(io, m, g.data))

Adapt.adapt_structure(to, g::Growable) = adapt(to, g.data)
KA.get_backend(g::Growable) = KA.get_backend(g.data)

"""
    ensure_capacity!(g::Growable, n)

Makes room for `n` elements (growing by 25 % headroom like the `Vector` buffers). A `Vector` is resized in
place; any other array is reallocated with `similar` and its current contents copied over. The elements
beyond the old length are undefined, as after `resize!`.
"""
@inline function ensure_capacity!(g::Growable, req_capacity::Int)
    if length(g.data) < req_capacity
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
