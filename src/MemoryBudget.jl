# =========================================================================
# MEMORY BUDGET
#
# Every growable buffer of a run (grid, time stepper, interpolator) is either *per particle* (length ≥ N)
# or *per neighbour slot* (length ≥ the slot capacity of the neighbour lists, see `_slot_capacity`), and
# the only arrays that are neither are the cell list of the bins. `memory_estimate` therefore prices a run
# from the element types of those buffers alone, without allocating anything, and `reserve!` sizes them
# once, to the largest particle capacity that fits a byte limit, and freezes them: a later
# `ensure_capacity!` beyond the reserved capacity throws a `CapacityError` instead of reallocating. A
# reallocation holds the old and the new array at the same time, which is what a device that is short
# of memory must avoid; with a reservation the run either fits from the start or stops with a clear error
# when it outgrows the budget.
#
# The host grid is always built first (also for a device run, see `device_grid`), so its neighbour search
# is the "CPU pretest" that tells how many neighbour slots a particle needs (`slots_per_particle`).
# =========================================================================
export memory_estimate, reserve!, slots_per_particle

# --- the growable buffers of each object ---------------------------------------------------------
# `particle_buffers(x, moving)` and `slot_buffers(x)` list the `Growable`s of `x`; `fixed_arrays(x)` the
# arrays that depend on neither count. Host-only state (the reordering permutations) is left out, and so
# is `neighbor.ranges`, which holds N + 1 entries and is handled on its own.
particle_buffers(::Any, ::Bool) = ()
slot_buffers(::Any) = ()
fixed_arrays(::Any) = ()

particle_buffers(pg::ParticleGrid, ::Bool) = (
    pg.rhos, pg.curvatures,
    pg.core.positions, pg.core.is_boundary, pg.core.volumes, pg.core.tags, pg.core.particle_orders,
    pg.shared.rho_buffer, pg.shared.pos_buffer, pg.shared.float_buffer, pg.shared.float_buffer2,
    pg.shared.bit_buffer, pg.shared.int_buffer,
    pg.neighbor.counts, pg.neighbor.offsets, pg.bins.cell_particles, pg.bins.bin_of)
slot_buffers(pg::ParticleGrid) = (pg.neighbor.indices, pg.neighbor.weights, pg.neighbor.distances)
fixed_arrays(pg::ParticleGrid) = (pg.bins.cell_start, pg.bins.cell_count, pg.bins.bin_neighbors)

particle_buffers(ts::GeneralRKTimeStepper, moving::Bool) = (
    ts.rho_n, ts.rho_cand, ts.Y_stages..., ts.K_E_stages..., ts.K_I_stages...,
    (moving ? (ts.x_n, ts.x_cand, ts.X_stages..., ts.K_X_stages...) : ())...,
    ts.work.triggered, ts.work.changed, ts.work.dirty, ts.work.scratch, ts.work.snapshot, ts.work.triggered_step)
slot_buffers(ts::GeneralRKTimeStepper) = (ts.int_buffer.df_flux, ts.int_buffer.df_scratch, ts.int_buffer.mask)

particle_buffers(w::MLSWeights, ::Bool) = (w.inv_moment, w.ok)
# The two weight caches of a MUSCL scheme are the same object when the degrees coincide: count it once.
particle_buffers(m::MUSCL, moving::Bool) = (m.gradients, particle_buffers(m.grad_weights, moving)...,
    (m.div_weights === m.grad_weights ? () : particle_buffers(m.div_weights, moving))...)
slot_buffers(m::MUSCL) = _aligned_dissipation(m.flux) ? (m.dsigns,) : ()

_array_bytes(a::AbstractArray) = length(a) * sizeof(eltype(a))
_element_bytes(b::AbstractArray) = sizeof(eltype(b))

"""
    slots_per_particle(pg::ParticleGrid) -> Float64

Neighbour slots per particle that the current neighbour lists of `pg` occupy in its slot layout: the mean
neighbour count on a host grid (row-major slots), the maximum neighbour count on a device grid (slot-major
slots, `max_nb` slots per particle). The default of `reserve!`; raise it for a run whose neighbourhoods
grow (moving or adaptive grids).
"""
slots_per_particle(pg::ParticleGrid) = _slots_per_particle(pg)
# (internal name: the keyword of `reserve!` shadows the exported one inside its body)
_slots_per_particle(pg::ParticleGrid) = pg.meta.N == 0 ? 0.0 : _slots_needed(pg) / pg.meta.N

# The slot capacity the current neighbour lists need (the capacity `pg.neighbor(pg)` sized them for).
function _slots_needed(pg::ParticleGrid)
    N = pg.meta.N
    N == 0 && return 0
    total = sum(view(pg.neighbor.counts, 1:N))
    return _slot_capacity(eltype(pg.neighbor.ranges), total, pg.meta.max_nb, N)
end

"""
    memory_estimate(pg::ParticleGrid, objs...; particles = pg.meta.N, slots = _slots_needed(pg)) -> NamedTuple

Bytes the growable buffers of the grid `pg` and of the time stepper and interpolators `objs` take for
`particles` particles and `slots` neighbour slots, from the element types of the buffers (nothing is
allocated):

    bytes = particles * per_particle + slots * per_slot + fixed

`fixed` is the cell list of the bins, which depends on the domain and the interaction radius only. The
moving-grid position buffers of a time stepper count when the grid has a mover. Returns
`(; bytes, per_particle, per_slot, fixed, particles, slots)`.
"""
function memory_estimate(pg::ParticleGrid, objs...; particles::Int = pg.meta.N, slots::Int = _slots_needed(pg))
    moving = _is_moving(pg)
    all = (pg, objs...)
    ranges       = _element_bytes(pg.neighbor.ranges)                    # `ranges` holds N + 1 entries
    per_particle = sum(_element_bytes, Iterators.flatten(particle_buffers(o, moving) for o in all); init = 0) + ranges
    per_slot     = sum(_element_bytes, Iterators.flatten(slot_buffers(o) for o in all); init = 0)
    fixed        = sum(_array_bytes, Iterators.flatten(fixed_arrays(o) for o in all); init = 0) + ranges
    bytes = particles * per_particle + slots * per_slot + fixed
    return (; bytes, per_particle, per_slot, fixed, particles, slots)
end

"""
    reserve!(pg::ParticleGrid, objs...; limit, tolerance = 0.05, slots_per_particle = slots_per_particle(pg)) -> NamedTuple

Sizes every growable buffer of the grid `pg` and of the time stepper and interpolators `objs` once, to the
largest particle capacity whose `memory_estimate` fits `limit` bytes less the fraction `tolerance`, with
`slots_per_particle` neighbour slots per particle, and freezes the capacities: from then on
`ensure_capacity!` throws a `CapacityError` instead of reallocating a buffer (a reallocation holds the old
and the new array at the same time). Call it on the objects of the run that live on the device, right
after construction and before the first step. `slots_per_particle` defaults to what the current
neighbour lists occupy (the host grid's search is the "CPU pretest" of the run); raise it for a moving or
adaptive run whose neighbourhoods grow. Throws an `ArgumentError` if the current particle count or
neighbour lists do not fit the limit. Returns `(; particles, slots, bytes, limit)`: the reserved
capacities and the bytes the buffers occupy.
"""
function reserve!(pg::ParticleGrid, objs...; limit::Integer, tolerance::Real = 0.05,
                  slots_per_particle::Real = _slots_per_particle(pg))
    0 <= tolerance < 1 || throw(ArgumentError("tolerance must be in [0, 1), got $tolerance"))
    slots_per_particle >= 0 || throw(ArgumentError("slots_per_particle must be non-negative, got $slots_per_particle"))
    est    = memory_estimate(pg, objs...; particles = 0, slots = 0)
    budget = floor(Int, limit * (1 - tolerance)) - est.fixed
    budget > 0 || throw(ArgumentError("a limit of $limit bytes does not even hold the cell list ($(est.fixed) bytes)"))
    particles = floor(Int, budget / (est.per_particle + slots_per_particle * est.per_slot))
    slots     = ceil(Int, particles * slots_per_particle)
    N, needed = pg.meta.N, _slots_needed(pg)
    particles >= N || throw(ArgumentError(
        "a limit of $limit bytes holds $particles particles at $slots_per_particle slots each, the grid has $N " *
        "(the current layout needs $(memory_estimate(pg, objs...).bytes) bytes)"))
    slots >= needed || throw(ArgumentError(
        "the reservation holds $slots neighbour slots, the current neighbour lists need $needed: " *
        "raise `slots_per_particle` (now $slots_per_particle, the lists occupy $(_slots_per_particle(pg)))"))

    moving = _is_moving(pg)
    _reserve_ranges!(pg.neighbor, particles)            # N + 1 entries, the new tail marked empty
    for o in (pg, objs...)
        foreach(b -> _reserve!(b, particles), particle_buffers(o, moving))
        foreach(b -> _reserve!(b, slots), slot_buffers(o))
    end
    # Every buffer now has exactly the reserved length, so the estimate is what they occupy.
    return (; particles, slots, bytes = memory_estimate(pg, objs...; particles, slots).bytes, limit = Int(limit))
end

# Exactly `n` elements (no headroom; a buffer that is larger, e.g. from the 25 % headroom of a growth or
# of the host grid a device grid was copied from, is shrunk), then fixed.
function _reserve!(g::Growable, n::Int)
    length(g.data) == n || (g.data = _resized(g.data, n))
    g.fixed = true
    return nothing
end

_resized(v::Vector, n::Int) = resize!(v, n)
function _resized(v::AbstractVector, n::Int)
    w = similar(v, n)
    m = min(n, length(v))
    m == 0 || copyto!(w, 1, v, 1, m)
    return w
end

function _reserve_ranges!(nd::NeighborData, particles::Int)
    old_len = length(nd.ranges)
    _reserve!(nd.ranges, particles + 1)
    old_len < particles + 1 && fill!(view(nd.ranges, (old_len + 1):(particles + 1)), _empty_slots(eltype(nd.ranges)))
    return nothing
end
