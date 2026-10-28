export MOODu1, MOODu2, OnlyMOOD, EPD0, EPD1, EPD2, StrictEPD0, Halo

# =========================================================================
# STRATEGIES
# =========================================================================
"""
    Halo{Force_N} <: MOODStrategy

`Force_N` is the number of hops to which neighbours of a dropped particle are forced down to its
order. There is no recalculation depth: the stepper derives the recompute set from
`_order_reach(interp, strategy)` (see below).

Strategies differ in how the interface degrees of an edge `(i, j)` are resolved
(`_get_interface_orders`):
- `EPD0`      : `(o_i, o_j)` — mismatched reconstruction allowed, no forcing
- `EPD1`      : `(min, min)` over the edge
- `EPD2`      : `(min, min)` over `N(i) ∪ N(j)`
- `StrictEPD0`: `(o_i, o_j)` + 1-hop forcing
"""
abstract type Halo{Force_N} <: MOODStrategy end
"Edge degrees `(k_i, k_j)`: each side of an edge reconstructs at its own order. No forcing."
struct EPD0       <: Halo{0} end
"Edge degrees `min(k_i, k_j)` on both sides. No forcing."
struct EPD1       <: Halo{0} end
"Edge degrees: minimum order over `N(i) ∪ N(j) ∪ {i, j}` on both sides. No forcing."
struct EPD2       <: Halo{0} end
"Edge degrees `(k_i, k_j)`; when a particle drops, its neighbours are lowered to its order (one hop, no cascade)."
struct StrictEPD0 <: Halo{1} end

# --- Forcing (pull formulation) ---
# Each particle computes the order its triggered neighbours force on it from the post-decrement
# `snapshot`, so a forced particle never cascades further, no two particles write the same entry
# (no atomics) and the result does not depend on the order of the loop. This relies on the neighbour
# relation being symmetric, which the search guarantees (one radius for all particles, wrapped distances).

"""
    forces_neighbours(strategy) -> Bool

`true` if the strategy lowers the orders of neighbours of a dropped particle (the forcing pass is
skipped otherwise). `NoStrategy` and `Halo{0}` do not force.
"""
@inline forces_neighbours(::MOODStrategy) = false
@inline forces_neighbours(::Halo{1}) = true

"""
    forced_order(strategy, j, pg, triggered, snapshot) -> Int

The order particle `j` ends up with after forcing, given the particles that dropped in this pass
(`triggered`) and the `snapshot` of all orders taken after the decrement. Called for interior
particles only. Defaults to `snapshot[j]` (no forcing).
"""
@inline forced_order(::MOODStrategy, j, pg, triggered, snapshot) = @inbounds snapshot[j]

@inline function forced_order(::Halo{1}, j, pg, triggered, snapshot)
    nb_slices  = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices
    o = @inbounds snapshot[j]
    @inbounds for k in nb_slices[j]
        p = nb_indices[k]
        triggered[p] && (o = min(o, snapshot[p]))
    end
    return o
end

# --- Interface degree resolution (used by MUSCL) ---
@inline function _nb_min_order(pg, idx)
    orders = pg.core.particle_orders
    nb_idx = pg.neighbor.indices
    o = orders[idx]
    @inbounds for k in pg.neighbor.ranges[idx]
        o = min(o, orders[nb_idx[k]])
    end
    return o
end

@inline _get_interface_orders(::MOODStrategy, pg, i, j) =
    (pg.core.particle_orders[i], pg.core.particle_orders[j])          # EPD0, StrictEPD0, NoStrategy

@inline function _get_interface_orders(::EPD1, pg, i, j)
    o = min(pg.core.particle_orders[i], pg.core.particle_orders[j])
    return (o, o)
end

@inline function _get_interface_orders(::EPD2, pg, i, j)
    o = min(_nb_min_order(pg, i), _nb_min_order(pg, j))
    return (o, o)
end

# Locality (`_order_reach`), CFL degree (`_divergence_degree`) and cached-gradient access
# (`has_cached_gradients`, `cached_gradients`) have generic defaults in API.jl; the scheme-specific
# overrides live next to the scheme (MUSCL block at the end of MUSCL.jl), so this file loads first.

# =========================================================================
# CRITERIA
# =========================================================================
"""
    MOODu1{T} <: MOODCriterion

Discrete Maximum Principle: flags a particle whose candidate leaves the extrema of its
neighbourhood (padded by `d`).
"""
struct MOODu1{T} <: MOODCriterion
    d::T
end

"""
    MOODu2(d; plateau = 1e-3, ratio = 1/2) <: MOODCriterion

`MOODu1` plus a smooth-extremum override (Diot, Loubère, Clain) for the components that violate the DMP:
- **plateau**: a component that varies across the stencil by less than `plateau` times its global range
  (`max − min` over the cloud at the previous stage) is accepted. Such a component carries no structure the DMP
  could protect at the scale of the stencil; what violates it there is the truncation error of the update
  (e.g. the density in the tail of a vortex, driven by the momentum flux). Needs no curvature, so it also holds
  below order 3. `0` disables it. The default `1e-3` keeps a smooth vortex at full order from about 20 particles
  per core radius on, and bounds discontinuities to about `1e-3` of the jump.
- **smooth extremum** (curvature coefficients cached by MUSCL, consistency order ≥ 3): in every direction the
  curvature has one sign over the stencil (`(c_min h²)(c_max h²) > −d²`) and comparable magnitudes among the
  nearest neighbours, within `1.5h` (`|c|_min ≥ ratio |c|_max`, Diot's test; `ratio = 0` disables it), or it is
  negligible (`|c|_max h² < d`). The magnitudes are compared over the nearest neighbours only because over a
  whole meshfree stencil (radius `2.5h` and more) the curvature of a smooth solution changes by more than a factor
  2 wherever it decays on a few spacings; the test is still needed: next to a jump the curvature collapses from
  large to zero between neighbours, which the sign test does not see. Curvatures are compared as the
  variation `c h²` they cause over one particle spacing `h`, so `d` keeps the units of the state.
Falls back to `MOODu1` for interpolators without gradients.
"""
struct MOODu2{T} <: MOODCriterion
    d::T
    plateau::T
    ratio::T
end
MOODu2(d::T; plateau::Real = T(1e-3), ratio::Real = T(0.5)) where {T <: Real} = MOODu2{T}(d, T(plateau), T(ratio))

"Criterion that flags every particle (testing: one step drives all orders to 1)."
struct OnlyMOOD <: MOODCriterion end

MOOD(criterion::MOODCriterion) = MOOD(EPD1(), criterion)

@inline function findLocalExtrema(rho_i::State{M, T}, nb_slice::OrdinalRange{Int, Int}, neighbor_fs::AbstractVector{State{M, T}}) where {M, T}
    minU = rho_i; maxU = rho_i
    @inbounds for k in nb_slice
        rho_j = neighbor_fs[k]
        minU = math_min.(minU, rho_j)
        maxU = math_max.(maxU, rho_j)
    end
    return minU, maxU
end

@inline function findLocalExtremaAbs(c_i::State{M, T}, curve_idx::Int, nb_slice::OrdinalRange{Int, Int},
                                     neighbor_indices::AbstractVector{Int}, grad_vec::AbstractVector) where {M, T}
    mini = c_i; maxi = c_i
    minAbs = abs.(c_i); maxAbs = minAbs
    @inbounds for k in nb_slice
        c_j = grad_vec[neighbor_indices[k]][curve_idx]
        abs_cj = abs.(c_j)
        mini = math_min.(mini, c_j);     maxi = math_max.(maxi, c_j)
        minAbs = math_min.(minAbs, abs_cj); maxAbs = math_max.(maxAbs, abs_cj)
    end
    return mini, maxi, minAbs, maxAbs
end

# `neighbour_fs` is slot-indexed: the neighbour states of the previous stage (`neighbour_states(U_prev, pg)`).
@inline function _dmp_fails(δ::T, rho_i::State{M, T}, nb_slice, newRho::State{M, T}, neighbour_fs) where {M, T}
    minU, maxU = findLocalExtrema(rho_i, nb_slice, neighbour_fs)
    for m in 1:M
        if abs(maxU[m] - minU[m]) >= δ^3 && (newRho[m] < minU[m] - δ || newRho[m] > maxU[m] + δ)
            return true
        end
    end
    return false
end

(m::MOOD{<:MOODStrategy, NoMOOD})(args...) = false
(m::MOOD{<:MOODStrategy, OnlyMOOD})(args...) = true

function (m::MOOD{<:MOODStrategy, MOODu1{T}})(
    g::DivergenceInterpolator, p_idx::Int, rho_i::State{M, T}, nb_slice::OrdinalRange{Int, Int},
    newRho::State{M, T}, pg::ParticleGrid{D, M, T}, U_prev::AbstractVector{State{M, T}}
) where {D, M, T}
    return _dmp_fails(m.criterion.d, rho_i, nb_slice, newRho, neighbour_states(U_prev, pg))
end

(m::MOOD{<:MOODStrategy, MOODu2{T}})(
    g::DivergenceInterpolator, p_idx::Int, rho_i::State{M, T}, nb_slice::OrdinalRange{Int, Int},
    newRho::State{M, T}, pg::ParticleGrid{D, M, T}, U_prev::AbstractVector{State{M, T}}
) where {D, M, T} = _u2_fails(m.criterion, g, p_idx, rho_i, nb_slice, newRho, pg, U_prev, nothing)

"""
    _criterion_flags(mood, interp, p, rho_i, nb_slice, cand, pg, U_prev, range) -> Bool

What the MOOD check calls: the criterion functor, except that `MOODu2` also receives `range`, the global range
`max − min` per component of `U_prev` (`nothing` if its plateau test is off), see `_global_range`.
"""
@inline _criterion_flags(mood::MOOD, interp, p, rho_i, nb_slice, cand, pg, U_prev, range) =
    mood(interp, p, rho_i, nb_slice, cand, pg, U_prev)
@inline _criterion_flags(mood::MOOD{<:MOODStrategy, <:MOODu2}, interp, p, rho_i, nb_slice, cand, pg, U_prev, range) =
    _u2_fails(mood.criterion, interp, p, rho_i, nb_slice, cand, pg, U_prev, range)

"Global range `max − min` per component of `U[1:N]` for the plateau test of `MOODu2` (a reduction; `nothing` if off)."
_global_range(::MOODCriterion, U, N) = nothing
function _global_range(c::MOODu2, U::AbstractVector{State{M, T}}, N::Int) where {M, T}
    c.plateau > zero(c.plateau) || return nothing
    V = view(U, 1:N)
    lo = mapreduce(identity, (a, b) -> min.(a, b), V)
    hi = mapreduce(identity, (a, b) -> max.(a, b), V)
    return hi - lo
end

@inline function _u2_fails(c::MOODu2{T}, g::DivergenceInterpolator, p_idx::Int, rho_i::State{M, T},
                           nb_slice::OrdinalRange{Int, Int}, newRho::State{M, T}, pg::ParticleGrid{D, M, T},
                           U_prev::AbstractVector{State{M, T}}, range) where {D, M, T}
    δ = c.d
    nbs = neighbour_states(U_prev, pg)
    minU, maxU = findLocalExtrema(rho_i, nb_slice, nbs)
    # Components that violate the DMP (as in `_dmp_fails`) and are not a plateau
    violating = ntuple(Val(M)) do mm
        abs(maxU[mm] - minU[mm]) >= δ^3 && (newRho[mm] < minU[mm] - δ || newRho[mm] > maxU[mm] + δ) &&
            !(range !== nothing && maxU[mm] - minU[mm] < c.plateau * range[mm])
    end
    any(violating) || return false

    # The curvature override needs cached gradients with a degree-2 reconstruction, i.e. a scheme
    # that caches them (trait) at consistency order ≥ 3. Otherwise behave like MOODu1.
    has_cached_gradients(g) || return true
    if _extract_order(g) < 3 || pg.core.particle_orders[p_idx] < 3
        return true
    end

    grad_vec  = cached_gradients(g)
    neighbors = pg.neighbor.indices
    dists     = get_distances(pg)
    h  = minimum(pg.meta.dx)
    h2 = h^2
    near_sq = (_U2_RATIO_RADIUS * h)^2
    for d in 1:D
        curve_idx = D + d
        c_i = grad_vec[p_idx][curve_idx]
        # Sign over the whole stencil; magnitudes over the nearest neighbours only (see `_U2_RATIO_RADIUS`)
        mini, maxi = c_i, c_i
        minAbs = abs.(c_i); maxAbs = minAbs
        @inbounds for k in nb_slice
            c_j = grad_vec[neighbors[k]][curve_idx]
            mini = math_min.(mini, c_j); maxi = math_max.(maxi, c_j)
            if sum(abs2, dists[k]) <= near_sq
                minAbs = math_min.(minAbs, abs.(c_j)); maxAbs = math_max.(maxAbs, abs.(c_j))
            end
        end
        for mm in 1:M
            violating[mm] || continue                       # only the components that violate the DMP
            max(abs(mini[mm]), abs(maxi[mm])) * h2 < δ && continue   # negligible curvature: variation below δ per spacing
            same_sign  = (mini[mm] * h2) * (maxi[mm] * h2) > -δ^2
            comparable = c.ratio <= zero(T) || minAbs[mm] >= c.ratio * maxAbs[mm]
            (same_sign && comparable) || return true
        end
    end
    return false
end

# Radius (in particle spacings) of the neighbours whose curvature magnitudes `MOODu2` compares: the face neighbours of a
# lattice, for which Diot's ratio test was designed. Over a whole meshfree stencil (radius 2.5h and more) the
# curvature of a smooth solution changes by more than a factor 2 wherever it decays on a few spacings.
const _U2_RATIO_RADIUS = 1.5

# =========================================================================
# WORK BUFFERS + DETECTION
# =========================================================================


# Physical admissibility detection (PAD) of a candidate state; `nothing`: no equation given, no check.
@inline _admissible(::Nothing, U) = true
@inline _admissible(eq, U) = is_admissible(eq, U)

# Kernel body of the candidate check (see Backend.jl): a candidate is flagged if it is not admissible
# (PAD) or fails the criterion.
@inline function _mood_flag!(p, mood, interp, pg, rho_prev, rho_cand, nb_slices, is_boundary, dirty, triggered, full, eq, range)
    (is_boundary[p] || !(full || dirty[p])) && return nothing
    cand = rho_cand[p]
    (!_admissible(eq, cand) || _criterion_flags(mood, interp, p, rho_prev[p], nb_slices[p], cand, pg, rho_prev, range)) &&
        (triggered[p] = true)
    return nothing
end

# Kernel body of the decrement: a flagged particle drops one order; one already at order 1 is un-flagged
# so it does not force its neighbours.
@inline function _mood_drop!(p, orders, triggered, changed, triggered_step)
    triggered[p] || return nothing
    if orders[p] > 1
        orders[p] -= 1
        changed[p] = true
        triggered_step[p] = true              # diagnostic, accumulated per step
    else
        triggered[p] = false
    end
    return nothing
end

# Kernel body of the forcing pass: interior particles pull the order their triggered neighbours force.
@inline function _mood_force!(j, strategy, pg, is_boundary, triggered, snapshot, orders, changed)
    is_boundary[j] && return nothing
    o = forced_order(strategy, j, pg, triggered, snapshot)
    if o < snapshot[j]
        orders[j]  = o
        changed[j] = true
    end
    return nothing
end

# Kernel body of one hop of `expand_mask!`: a particle is marked if it or any neighbour was.
@inline function _expand_body!(j, scratch, mask, nb_slices, nb_indices)
    m = mask[j]
    if !m
        @inbounds for k in nb_slices[j]
            if mask[nb_indices[k]]
                m = true
                break
            end
        end
    end
    scratch[j] = m
    return nothing
end

"""
    expand_mask!(mask, scratch, pg, hops)

Marks every particle within `hops` topological hops of a marked one (boundary particles included;
the compute passes skip them). One kernel pass per hop.
"""
function expand_mask!(mask::AbstractVector{Bool}, scratch::AbstractVector{Bool}, pg::ParticleGrid, hops::Int)
    N  = pg.meta.N
    nb = pg.neighbor
    for _ in 1:hops
        foreach_particle(_expand_body!, get_backend(), N, (scratch, mask, nb.ranges, nb.indices))
        copyto!(mask, 1, scratch, 1, N)
    end
    return nothing
end

"""
    force_neighbours!(strategy, pg, triggered, snapshot, orders, changed)

Forcing pass of `detect_and_drop!`: lowers interior particles to `forced_order` from the `snapshot`
and marks them in `changed`.
"""
function force_neighbours!(strategy::MOODStrategy, pg::ParticleGrid, triggered, snapshot, orders, changed)
    foreach_particle(_mood_force!, get_backend(), pg.meta.N,
        (strategy, pg, pg.core.is_boundary, triggered, snapshot, orders, changed))
    return nothing
end

"""
    detect_and_drop!(mood, work, interp, pg, rho_prev, rho_cand, full; eq = nothing) -> Bool

With an equation `eq`, a candidate state that is not admissible (`is_admissible(eq, U)`, e.g. a NaN or a
negative pressure) is flagged as well. Checks `rho_cand` against the neighbourhood of `rho_prev` (neighbour states read from `rho_prev`,
gradients cached in `interp`) for all interior particles (`full`) or only those in `work.dirty`.
Drops one order on every flagged particle, applies forcing halos from a snapshot, and records the
set of changed orders in `work.changed`. Returns `true` if any order changed. Three kernel passes
(check, decrement, forcing) and one reduction; nothing runs on the host.
"""
function detect_and_drop!(
    mood::MOOD, work::MOODWork, interp::DivergenceInterpolator, pg::ParticleGrid{D, M, T},
    rho_prev::AbstractVector, rho_cand::AbstractVector, full::Bool; eq = nothing
) where {D, M, T}
    mood.criterion isa NoMOOD && return false

    N           = pg.meta.N
    nb_slices   = pg.neighbor.ranges
    is_boundary = pg.core.is_boundary
    orders      = pg.core.particle_orders
    triggered, changed, dirty, snapshot = parent(work.triggered), parent(work.changed), parent(work.dirty), parent(work.snapshot)
    triggered_step = parent(work.triggered_step)

    fill!(view(triggered, 1:N), false)
    fill!(view(changed, 1:N), false)

    # 1. Candidate check (per-particle writes only); the global range feeds the plateau test of MOODu2
    range = _global_range(mood.criterion, rho_prev, N)
    foreach_particle(_mood_flag!, get_backend(), N,
        (mood, interp, pg, rho_prev, rho_cand, nb_slices, is_boundary, dirty, triggered, full, eq, range))

    # 2. Decrement. Particles already at order 1 are un-flagged so they don't force.
    foreach_particle(_mood_drop!, get_backend(), N, (orders, triggered, changed, triggered_step))
    any(view(changed, 1:N)) || return false

    # 3. Forcing from a snapshot (no cascade)
    forces_neighbours(mood.strategy) || return true
    copyto!(snapshot, 1, orders, 1, N)
    force_neighbours!(mood.strategy, pg, triggered, snapshot, orders, changed)
    return true
end
