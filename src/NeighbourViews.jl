# =========================================================================
# LAZY PER-SLOT VIEWS OF A STATE VECTOR
#
# Schemes read a neighbour's state through the slot index `k` of the neighbour lists. Instead of
# gathering those states into per-slot buffers before the flux pass (a write and a read of every
# neighbour state per stage), these views index the state vector on the fly: `s[k] == U[indices[k]]`.
# They are bits types, so kernel bodies build them from their arguments at no cost, and every MLS
# solve, limiter and MOOD criterion that takes a slot-indexed vector of states accepts them unchanged.
# =========================================================================
export neighbour_states, neighbour_differences

"""
    SlotStates(U, indices) <: AbstractVector
    neighbour_states(U, pg)

The states of the neighbours by slot: `s[k] == U[indices[k]]`, with `indices = pg.neighbor.indices`.
"""
struct SlotStates{S, V <: AbstractVector{S}, I <: AbstractVector{Int}} <: AbstractVector{S}
    U::V
    indices::I
end
Base.size(s::SlotStates) = size(s.indices)
Base.IndexStyle(::Type{<:SlotStates}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(s::SlotStates, k::Int) = s.U[s.indices[k]]

"""
    SlotDifferences(U, indices, f_i) <: AbstractVector
    neighbour_differences(U, pg, f_i)

The differences to the centre particle by slot: `d[k] == U[indices[k]] - f_i`.
"""
struct SlotDifferences{S, V <: AbstractVector{S}, I <: AbstractVector{Int}} <: AbstractVector{S}
    U::V
    indices::I
    f_i::S
end
Base.size(d::SlotDifferences) = size(d.indices)
Base.IndexStyle(::Type{<:SlotDifferences}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(d::SlotDifferences, k::Int) = d.U[d.indices[k]] - d.f_i

@inline neighbour_states(U::AbstractVector{S}, pg) where {S} = SlotStates{S, typeof(U), typeof(pg.neighbor.indices)}(U, pg.neighbor.indices)
@inline neighbour_differences(U::AbstractVector{S}, pg, f_i::S) where {S} =
    SlotDifferences{S, typeof(U), typeof(pg.neighbor.indices)}(U, pg.neighbor.indices, f_i)
