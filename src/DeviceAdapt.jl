# =========================================================================
# KERNEL ARGUMENT CONVERSION (Adapt.jl)
#
# KernelAbstractions calls `Adapt.adapt(backend, arg)` on every kernel argument. These methods turn the
# host structures into bits types: arrays become device arrays (or device views), mutable metadata an
# immutable snapshot, `Ref`s plain values, and host-only parts (geometry closures and boundary map,
# scratch buffers, reordering state) are dropped. Nothing is copied on a shared-memory backend.
#
# The same methods make a grid *resident* on a device when called with an array type, e.g.
# `Adapt.adapt(CuArray, pg)` (then the metadata snapshot is replaced by a fresh mutable copy by
# `device_grid`, see below).
# =========================================================================
using Adapt

Adapt.adapt_structure(to, m::GridMetadata) = GridMetadataView(m)

Adapt.adapt_structure(to, c::ParticleGridCore{D, T}) where {D, T} = ParticleGridCore{D, T}(
    adapt(to, c.positions), adapt(to, c.is_boundary), adapt(to, c.volumes), adapt(to, c.tags), adapt(to, c.particle_orders))

Adapt.adapt_structure(to, nd::NeighborData{D, T, WF}) where {D, T, WF} = NeighborData{D, T, WF}(
    nd.weight_func, adapt(to, nd.ranges), adapt(to, nd.indices), adapt(to, nd.weights), adapt(to, nd.distances),
    adapt(to, nd.counts), adapt(to, nd.offsets))

Adapt.adapt_structure(to, b::GlobalBins{D, T, N_OFF}) where {D, T, N_OFF} = GlobalBins{D, T, N_OFF}(
    b.mins, b.maxs, b.coarse_size, b.coarse_dims, adapt(to, b.head), adapt(to, b.next), adapt(to, b.bin_neighbors))

Adapt.adapt_structure(to, sb::SharedBuffers{D, M, T}) where {D, M, T} = SharedBuffers{D, M, T}(
    adapt(to, sb.rho_buffer), adapt(to, sb.pos_buffer), adapt(to, sb.float_buffer), adapt(to, sb.float_buffer2),
    adapt(to, sb.bit_buffer), adapt(to, sb.int_buffer))

# Kernel view of a grid: geometry (closures, boundary map), scratch buffers and reordering stay on the host.
Adapt.adapt_structure(to, pg::ParticleGrid) = ParticleGrid(
    adapt(to, pg.meta), nothing, pg.domain, adapt(to, pg.core), nothing, adapt(to, pg.neighbor), nothing,
    adapt(to, pg.bins), pg.mover, adapt(to, pg.rhos), adapt(to, pg.curvatures))

Adapt.adapt_structure(to, ib::InteractionBuffer{D, M, T}) where {D, M, T} = InteractionBuffer{D, M, T}(
    adapt(to, ib.f), adapt(to, ib.df), adapt(to, ib.df_flux), adapt(to, ib.df_scratch), adapt(to, ib.mask))

Adapt.adapt_structure(to, w::MLSWeights{B, IO, T, L}) where {B, IO, T, L} =
    MLSWeightsView{B, IO, T, L, typeof(adapt(to, w.inv_moment)), typeof(adapt(to, w.ok))}(
        adapt(to, w.inv_moment), adapt(to, w.ok), w.version[], w.invL[], w.scales[])

Adapt.adapt_structure(to, m::MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER}) where {D, M, T, B_LEN, MAX_ORDER, DIV_ORDER} =
    MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER}(m.interpolators, m.limiter, m.flux, adapt(to, m.gradients),
                                                adapt(to, m.grad_weights), adapt(to, m.div_weights))

function Adapt.adapt_structure(to, bt::IMEXButcherTableau{T}) where {T}
    s = num_stages(bt)
    return IMEXButcherTableau(SMatrix{s, s, T}(bt.a), SMatrix{s, s, T}(bt.a_t), SVector{s, T}(bt.c),
                              SVector{s, T}(bt.c_t), SVector{s, T}(bt.b), SVector{s, T}(bt.b_t))
end

"""
    device_grid(ArrayType, pg::ParticleGrid) -> ParticleGrid

A copy of `pg` whose component arrays are `ArrayType`s (e.g. `CuArray`), with its own mutable metadata.
The geometry, scratch buffers and reordering state are not transferred (they are host-only), so the
resulting grid serves the kernel passes of the time stepper; neighbour searches, reordering, boundary
conditions and adaptivity still run on the host grid (see docs/src/gpu_roadmap.md).
"""
function device_grid(::Type{AT}, pg::ParticleGrid{D, M, T}) where {AT <: AbstractArray, D, M, T}
    m = pg.meta
    meta = GridMetadata{D, T}(m.N, m.N_interior, m.N_ghost, m.R, m.max_dx, m.dx, m.interp_range_factor, m.max_nb, m.geometry_version)
    return ParticleGrid(meta, pg.geometry, pg.domain, adapt(AT, pg.core), adapt(AT, pg.shared), adapt(AT, pg.neighbor),
                        pg.reorder, adapt(AT, pg.bins), pg.mover, adapt(AT, pg.rhos), adapt(AT, pg.curvatures))
end
