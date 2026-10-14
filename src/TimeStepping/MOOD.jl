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

# --- Forcing: NoStrategy and Halo{0} land on the no-op ---
@inline trigger_halo!(::MOODStrategy, p_idx, pg, orders, snapshot, changed) = nothing

# Reads the post-decrement snapshot so a forced particle never cascades further.
@inline function trigger_halo!(::Halo{1}, p_idx, pg, orders, snapshot, changed)
    nb_slices   = pg.neighbor.ranges
    nb_indices  = pg.neighbor.indices
    is_boundary = pg.core.is_boundary
    o_p = snapshot[p_idx]
    @inbounds for k in nb_slices[p_idx]
        j = nb_indices[k]
        if !is_boundary[j] && orders[j] > o_p
            orders[j]  = o_p
            changed[j] = true
        end
    end
    return nothing
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
    MOODu2{T} <: MOODCriterion

`MOODu1` plus a smooth-extremum override based on the curvature coefficients cached by MUSCL
(consistency order ≥ 3). Falls back to `MOODu1` for interpolators without gradients.
"""
struct MOODu2{T} <: MOODCriterion
    d::T
end

"Criterion that flags every particle (testing: one step drives all orders to 1)."
struct OnlyMOOD <: MOODCriterion end

MOOD(criterion::MOODCriterion) = MOOD(EPD1(), criterion)

@inline function findLocalExtrema(rho_i::State{M, T}, nb_slice::UnitRange{Int}, neighbor_fs::AbstractVector{State{M, T}}) where {M, T}
    minU = rho_i; maxU = rho_i
    @inbounds for k in nb_slice
        rho_j = neighbor_fs[k]
        minU = math_min.(minU, rho_j)
        maxU = math_max.(maxU, rho_j)
    end
    return minU, maxU
end

@inline function findLocalExtremaAbs(c_i::State{M, T}, curve_idx::Int, nb_slice::UnitRange{Int},
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

@inline function _dmp_fails(δ::T, rho_i::State{M, T}, nb_slice, newRho::State{M, T}, int_buffer_f) where {M, T}
    minU, maxU = findLocalExtrema(rho_i, nb_slice, int_buffer_f)
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
    g::DivergenceInterpolator, p_idx::Int, rho_i::State{M, T}, nb_slice::UnitRange{Int},
    newRho::State{M, T}, pg::ParticleGrid{D, M, T}, int_buffer_f::AbstractVector{State{M, T}}
) where {D, M, T}
    return _dmp_fails(m.criterion.d, rho_i, nb_slice, newRho, int_buffer_f)
end

function (m::MOOD{<:MOODStrategy, MOODu2{T}})(
    g::DivergenceInterpolator, p_idx::Int, rho_i::State{M, T}, nb_slice::UnitRange{Int},
    newRho::State{M, T}, pg::ParticleGrid{D, M, T}, int_buffer_f::AbstractVector{State{M, T}}
) where {D, M, T}
    δ = m.criterion.d
    _dmp_fails(δ, rho_i, nb_slice, newRho, int_buffer_f) || return false

    # The curvature override needs cached gradients with a degree-2 reconstruction, i.e. a scheme
    # that caches them (trait) at consistency order ≥ 3. Otherwise behave like MOODu1.
    has_cached_gradients(g) || return true
    if _extract_order(g) < 3 || pg.core.particle_orders[p_idx] < 3
        return true
    end

    grad_vec  = cached_gradients(g)
    neighbors = pg.neighbor.indices
    for d in 1:D
        curve_idx = D + d
        c_i = grad_vec[p_idx][curve_idx]
        mini, maxi, minAbs, maxAbs = findLocalExtremaAbs(c_i, curve_idx, nb_slice, neighbors, grad_vec)
        for mm in 1:M
            ratio = maxAbs[mm] < T(1e-12) ? one(T) : minAbs[mm] / maxAbs[mm]
            valid = (mini[mm] * maxi[mm] > -δ) && (ratio >= T(0.5) || maxAbs[mm] < δ)
            valid || return true
        end
    end
    return false
end

# =========================================================================
# WORK BUFFERS + DETECTION
# =========================================================================


"""
    expand_mask!(mask, scratch, pg, hops)

Marks every particle within `hops` topological hops of a marked one (boundary particles included;
the compute passes skip them).
"""
function expand_mask!(mask::Vector{Bool}, scratch::Vector{Bool}, pg::ParticleGrid, hops::Int)
    N = pg.meta.N
    nb_slices, nb_indices = pg.neighbor.ranges, pg.neighbor.indices
    for _ in 1:hops
        copyto!(scratch, 1, mask, 1, N)
        @inbounds for p in 1:N
            if mask[p]
                for k in nb_slices[p]; scratch[nb_indices[k]] = true; end
            end
        end
        copyto!(mask, 1, scratch, 1, N)
    end
    return nothing
end

"""
    detect_and_drop!(mood, work, interp, pg, int_buffer, rho_prev, rho_cand, full) -> Bool

Checks `rho_cand` against the neighbourhood of `rho_prev` (neighbour states in `int_buffer.f`,
gradients cached in `interp`) for all interior particles (`full`) or only those in `work.dirty`.
Drops one order on every flagged particle, applies forcing halos from a snapshot, and records the
set of changed orders in `work.changed`. Returns `true` if any order changed.
"""
function detect_and_drop!(
    mood::MOOD, work::MOODWork, interp::DivergenceInterpolator, pg::ParticleGrid{D, M, T},
    int_buffer::InteractionBuffer{D, M, T}, rho_prev::AbstractVector, rho_cand::AbstractVector, full::Bool
) where {D, M, T}
    mood.criterion isa NoMOOD && return false

    N           = pg.meta.N
    nb_slices   = pg.neighbor.ranges
    is_boundary = pg.core.is_boundary
    orders      = pg.core.particle_orders
    triggered, changed, dirty, snapshot = work.triggered, work.changed, work.dirty, work.snapshot
    use_threads = get_use_threads()

    fill!(view(triggered, 1:N), false)
    fill!(view(changed, 1:N), false)

    # 1. Candidate check (parallel; per-particle writes only)
    @smart_parallel use_threads for p in 1:N
        if is_boundary[p] || !(full || dirty[p]); continue; end
        if mood(interp, p, rho_prev[p], nb_slices[p], rho_cand[p], pg, int_buffer.f)
            triggered[p] = true
        end
    end

    # 2. Decrement (serial). Particles already at order 1 are un-flagged so they don't force.
    any_dropped = false
    @inbounds for p in 1:N
        if triggered[p]
            if orders[p] > 1
                orders[p] -= 1
                changed[p] = true
                work.triggered_step[p] = true         # diagnostic, accumulated per step
                any_dropped = true
            else
                triggered[p] = false
            end
        end
    end
    any_dropped || return false

    # 3. Forcing from a snapshot (serial, no cascade)
    copyto!(snapshot, 1, orders, 1, N)
    @inbounds for p in 1:N
        triggered[p] && trigger_halo!(mood.strategy, p, pg, orders, snapshot, changed)
    end
    return true
end
