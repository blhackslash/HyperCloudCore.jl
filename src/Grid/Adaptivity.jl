export ParticleAdaptivity, adapt_particles!

# =========================================================================
# PARTICLE INSERTION / MERGING WITH A CONSERVATIVE SHAPE-FUNCTION REMAP
# =========================================================================
#
# A topology change (particles added, removed or moved) replaces the partition of unity
# {φ_i^old} by {φ_i^new}. Every particle's conserved content changes by the "swept" amount
#
#     δm_i = ∫ (φ_i^new − φ_i^old)(x) Ũ(x) dx ,       δV_i = ∫ (φ_i^new − φ_i^old) dx ,
#
# where Ũ is the limited reconstruction of the OLD solution. Because both families sum to one
# pointwise, Σ_i δm_i = 0 and Σ_i δV_i = 0 for any quadrature of the integral: the remap is
# exactly conservative by construction, without cell intersections. New particles have
# φ^old = 0, removed particles have φ^new = 0. The integral is evaluated on a small throwaway
# lattice around the event; a limited (bound-clipped) linear reconstruction keeps Ũ in the local
# bounds, and a final repair pass enforces the discrete maximum principle while keeping the sum
# exact. All of this is the particle analogue of swept-region remapping in ALE finite volumes.

"""
    ParticleAdaptivity(; max_gap = 1.6, min_gap = 0.5, clearance = 0.7,
                         samples_per_dx = 2, exclude_reduced_order = true, max_events = typemax(Int))

Rules for keeping a moving point cloud well-spaced. Lengths are multiples of the particle spacing
`dx = minimum(pg.meta.dx)`.
- **Insertion**: a neighbour pair `(i, j)` farther apart than `max_gap·dx` whose midpoint has no
  particle within `clearance·dx` gets a new particle at the midpoint.
- **Merge**: a neighbour pair closer than `min_gap·dx` is replaced by one particle at the midpoint.
- `exclude_reduced_order`: skip events near particles whose MOOD order was lowered in the last step
  (the shock band), where interpolation of any order is unreliable.
- `samples_per_dx`: resolution of the remap quadrature lattice.
- At most one event per neighbourhood and per call; events are kept `4R` apart so their remaps do
  not interact. Call once between time steps.
"""
struct ParticleAdaptivity{T}
    max_gap::T
    min_gap::T
    clearance::T
    samples_per_dx::Int
    exclude_reduced_order::Bool
    max_events::Int
end

function ParticleAdaptivity(::Type{T} = Float64; max_gap = 1.6, min_gap = 0.5, clearance = 0.7,
                            samples_per_dx::Int = 2, exclude_reduced_order::Bool = true,
                            max_events::Int = typemax(Int)) where {T}
    @assert min_gap < 1 < max_gap "need min_gap < 1 < max_gap (in units of dx)"
    return ParticleAdaptivity{T}(T(max_gap), T(min_gap), T(clearance), samples_per_dx, exclude_reduced_order, max_events)
end

struct _AdaptEvent{D, T}
    center::Space{D, T}
    removed::Vector{Int}
    added::Vector{Space{D, T}}
end

# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------
"""
    adapt_particles!(pg, ad::ParticleAdaptivity, max_order::Int) -> (; inserted, removed, mass_residual)

Detects holes and clusters, applies the conservative remap for every event on the old topology,
changes the particle set, and rebuilds the neighbour lists. `max_order` is the scheme's maximum
consistency order (new particles start there, and `exclude_reduced_order` compares against it).
Requires valid neighbour lists and volumes (`update_volumes!`). `mass_residual` is the change of
`total_mass` over the call, which should be at round-off level.

The event work (detection, remap, compaction) is serial and runs on the host. On a grid built with
`device_grid` a kernel first checks whether any neighbour pair violates the spacing rules; only then
is the grid copied to the host, adapted there, and copied back into the device grid in place (its
arrays grow as needed), where the neighbour search runs. A well-spaced cloud costs one kernel and
one reduction per call.
"""
function adapt_particles!(pg::ParticleGrid{D, M, T}, ad::ParticleAdaptivity{T}, max_order::Int) where {D, M, T}
    nothing_done = (inserted = 0, removed = 0, mass_residual = zero(State{M, T}))
    pg.meta.N == 0 && return nothing_done
    _has_adapt_candidates(pg, ad) || return nothing_done
    mass0 = total_mass(pg)

    on_host = parent(pg.rhos) isa Array
    hg = on_host ? pg : device_grid(Array, pg)            # the event work is serial: on a host copy of a device grid
    N  = hg.meta.N

    events = _detect_events(hg, ad, max_order)
    isempty(events) && return nothing_done

    remove_mask = hg.shared.bit_buffer
    fill!(view(remove_mask, 1:N), false)
    new_pos, new_rho, new_vol = Space{D, T}[], State{M, T}[], T[]
    df = Vector{State{M, T}}(undef, length(hg.neighbor.indices))   # neighbour-slot scratch shared by all events

    for ev in events
        _remap_event!(hg, ad, ev, new_pos, new_rho, new_vol, df)
        for r in ev.removed; remove_mask[r] = true; end
    end

    n_removed = count(view(remove_mask, 1:N))
    _apply_topology!(hg, remove_mask, new_pos, new_rho, new_vol, max_order)
    on_host || _upload_particles!(pg, hg)
    pg.neighbor(pg)

    return (inserted = length(new_pos), removed = n_removed, mass_residual = total_mass(pg) - mass0)
end

"""
    _has_adapt_candidates(pg, ad) -> Bool

`true` if some interior particle has a neighbour pair that could trigger an event: closer than
`min_gap` (merge) or farther than `max_gap` with an empty midpoint (insertion). A superset of what
`_detect_events` selects (which also spaces events apart and skips the shock band). One kernel and one
reduction, on the grid's backend.
"""
function _has_adapt_candidates(pg::ParticleGrid{D, M, T}, ad::ParticleAdaptivity{T}) where {D, M, T}
    N    = pg.meta.N
    nb   = pg.neighbor
    dom  = pg.domain
    dx   = minimum(pg.meta.dx)
    flag = pg.shared.bit_buffer
    foreach_particle(_adapt_candidate!, get_backend(), N,
        (flag, pg.core.positions, pg.core.is_boundary, nb.ranges, nb.indices, nb.distances, dom.L_wrap, dom.invL_wrap,
         dom.is_periodic, dom.L, dom.canvas_mins, (ad.min_gap * dx)^2, (ad.max_gap * dx)^2, (ad.clearance * dx)^2))
    return any(view(flag, 1:N))
end

# Kernel body (see Backend.jl): the pair tests of `_detect_events` without the event bookkeeping.
@inline function _adapt_candidate!(i, flag, pos, isb, ranges, indices, dists, L, Li, is_per, Lv, mins,
                                   min_gap_sq, max_gap_sq, clear_sq)
    found = false
    if !isb[i]
        @inbounds for k in ranges[i]
            j = indices[k]
            j > i || continue
            d2 = sum(abs2, dists[k])
            if d2 < min_gap_sq
                isb[j] || (found = true)
            elseif d2 > max_gap_sq
                mid   = _wrap_coords(pos[i] + dists[k] / 2, is_per, Lv, mins)
                empty = true
                for k2 in ranges[i]
                    if sum(abs2, _wrapped_diff(pos[indices[k2]], mid, L, Li)) < clear_sq
                        empty = false
                        break
                    end
                end
                empty && (found = true)
            end
            found && break
        end
    end
    flag[i] = found
    return nothing
end

# Copies the adapted particle set of the host copy `hg` back into the device grid `pg` (growing it as needed).
function _upload_particles!(pg::ParticleGrid, hg::ParticleGrid)
    N = hg.meta.N
    ensure_capacity!(pg, N)
    for (dst, src) in ((pg.core.positions, hg.core.positions), (pg.core.is_boundary, hg.core.is_boundary),
                       (pg.core.tags, hg.core.tags), (pg.core.particle_orders, hg.core.particle_orders),
                       (pg.core.volumes, hg.core.volumes), (pg.rhos, hg.rhos), (pg.curvatures, hg.curvatures))
        copyto!(parent(dst), 1, parent(src), 1, N)
    end
    pg.meta.N          = N
    pg.meta.N_interior = hg.meta.N_interior
    pg.meta.N_ghost    = hg.meta.N_ghost
    return nothing
end

# -------------------------------------------------------------------------
# Detection
# -------------------------------------------------------------------------
function _detect_events(pg::ParticleGrid{D, M, T}, ad::ParticleAdaptivity{T}, max_order::Int) where {D, M, T}
    N       = pg.meta.N
    nb      = pg.neighbor
    pos     = pg.core.positions
    isb     = pg.core.is_boundary
    orders  = pg.core.particle_orders
    dx      = minimum(pg.meta.dx)
    R       = pg.meta.R
    L, Li   = pg.domain.L_wrap, pg.domain.invL_wrap
    touched = pg.shared.bit_buffer
    fill!(view(touched, 1:N), false)

    idx, dvec = Int[], Space{D, T}[]
    # Shock band: particles whose order was lowered, plus everything their remap could touch
    if ad.exclude_reduced_order
        for i in 1:N
            if !isb[i] && 0 < orders[i] < max_order   # 0 = never set (fresh grid), not a drop
                n = _collect_near!(pg, pos[i], (4R)^2, idx, dvec)
                for k in 1:n; touched[idx[k]] = true; end
            end
        end
    end

    max_gap_sq, min_gap_sq, clear_sq = (ad.max_gap * dx)^2, (ad.min_gap * dx)^2, (ad.clearance * dx)^2
    events = _AdaptEvent{D, T}[]

    @inbounds for i in 1:N
        (isb[i] || touched[i]) && continue
        ev = nothing

        # Merge: closest too-close neighbour
        best_k, best_d2 = 0, min_gap_sq
        for k in nb.ranges[i]
            j = nb.indices[k]
            d2 = sum(abs2, nb.distances[k])
            if j > i && !isb[j] && !touched[j] && d2 < best_d2
                best_k, best_d2 = k, d2
            end
        end
        if best_k != 0
            mid = _wrap_point(pg, pos[i] + nb.distances[best_k] / 2)
            ev = _AdaptEvent{D, T}(mid, [i, nb.indices[best_k]], [mid])
        else
            # Insertion: a far neighbour with an empty midpoint
            for k in nb.ranges[i]
                j = nb.indices[k]
                d2 = sum(abs2, nb.distances[k])
                (j > i && !touched[j] && d2 > max_gap_sq) || continue
                mid = _wrap_point(pg, pos[i] + nb.distances[k] / 2)
                empty = true
                for k2 in nb.ranges[i]
                    if sum(abs2, _wrapped_diff(pos[nb.indices[k2]], mid, L, Li)) < clear_sq
                        empty = false; break
                    end
                end
                if empty
                    ev = _AdaptEvent{D, T}(mid, Int[], [mid])
                    break
                end
            end
        end

        if ev !== nothing
            push!(events, ev)
            n = _collect_near!(pg, ev.center, (4R)^2, idx, dvec)      # keep events 4R apart
            for k in 1:n; touched[idx[k]] = true; end
            length(events) >= ad.max_events && break
        end
    end
    return events
end

# -------------------------------------------------------------------------
# The remap of one event (on the OLD topology)
# -------------------------------------------------------------------------
function _remap_event!(pg::ParticleGrid{D, M, T}, ad::ParticleAdaptivity{T}, ev::_AdaptEvent{D, T},
                       new_pos::Vector{Space{D, T}}, new_rho::Vector{State{M, T}}, new_vol::Vector{T},
                       df::Vector{State{M, T}} = Vector{State{M, T}}(undef, length(pg.neighbor.indices))) where {D, M, T}
    nb     = pg.neighbor
    pos    = pg.core.positions
    V      = pg.core.volumes
    U      = pg.rhos
    wf     = nb.weight_func
    R      = pg.meta.R
    R_sq   = R^2
    dx     = minimum(pg.meta.dx)
    L, Li  = pg.domain.L_wrap, pg.domain.invL_wrap
    geom   = pg.geometry
    is_per = pg.domain.is_periodic
    lmap   = pg.shared.int_buffer                     # global index -> local id (0 = not involved)
    fill!(view(lmap, 1:pg.meta.N), 0)                 # the neighbour search leaves prefix sums in here
    n_add  = length(ev.added)

    # 1. Local set: every old particle whose shape function can change (within R of the
    #    quadrature box, whose half-width is 2R)
    loc_idx, loc_d = Int[], Space{D, T}[]
    n_loc = _collect_near!(pg, ev.center, (R * (1 + 2 * sqrt(T(D))) + dx)^2, loc_idx, loc_d)
    @inbounds for l in 1:n_loc; lmap[loc_idx[l]] = l; end
    is_removed = falses(n_loc)
    for r in ev.removed; is_removed[lmap[r]] = true; end

    # 2. Limited linear reconstruction data of the old solution on the local set
    grads = Vector{SVector{D, State{M, T}}}(undef, n_loc)
    lo    = Vector{State{M, T}}(undef, n_loc)
    hi    = Vector{State{M, T}}(undef, n_loc)
    scale = dx
    @inbounds for l in 1:n_loc
        j  = loc_idx[l]
        sl = nb.ranges[j]
        mn, mx = U[j], U[j]
        for k in sl
            uk = U[nb.indices[k]]
            df[k] = uk - U[j]
            mn = math_min.(mn, uk); mx = math_max.(mx, uk)
        end
        lo[l], hi[l] = mn, mx
        g = isempty(sl) ? zero(SVector{D, State{M, T}}) :
            Interpolator{D, 1, 1}()(sl, nb.distances, nb.weights, df; scale = scale)
        grads[l] = SVector{D, State{M, T}}(ntuple(d -> g[d], Val(D)))
    end

    # 3. Quadrature over the box center ± 2R
    δV  = zeros(T, n_loc + n_add)
    δm  = zeros(State{M, T}, n_loc + n_add)
    hq  = dx / ad.samples_per_dx
    nq  = ceil(Int, 4R / hq)
    ω   = hq^D
    qidx, qd = Int[], Space{D, T}[]
    w_add = zeros(T, n_add)

    for I in CartesianIndices(ntuple(_ -> nq, Val(D)))
        y_raw = Space{D, T}(ntuple(d -> ev.center[d] - 2R + (I[d] - T(0.5)) * hq, Val(D)))
        y = _wrap_point(pg, y_raw)
        geom.is_interior(y) || continue
        n = _collect_near!(pg, y, R_sq, qidx, qd)
        n == 0 && continue

        # S_old, nearest donor, S_new
        S_old = zero(T); S_new = zero(T)
        jstar, dstar = 0, T(Inf)
        @inbounds for k in 1:n
            l  = lmap[qidx[k]]
            d2 = sum(abs2, qd[k])
            wk = wf(d2)
            S_old += wk
            removed_k = l != 0 && is_removed[l]
            removed_k || (S_new += wk)
            if l != 0 && !removed_k && d2 < dstar; jstar, dstar = k, d2; end
        end
        # Removed particles may be the only ones nearby: use them as donors too
        if jstar == 0
            @inbounds for k in 1:n
                d2 = sum(abs2, qd[k])
                if lmap[qidx[k]] != 0 && d2 < dstar; jstar, dstar = k, d2; end
            end
        end
        jstar == 0 && continue
        @inbounds for a in 1:n_add
            da = _wrapped_diff(ev.added[a], y, L, Li)
            d2 = sum(abs2, da)
            w_add[a] = d2 <= R_sq ? wf(d2) : zero(T)
            S_new += w_add[a]
        end
        (S_old > 0 && S_new > 0) || continue

        # Ũ(y): limited linear reconstruction of the nearest old particle
        lstar = lmap[qidx[jstar]]
        u_rec = U[loc_idx[lstar]]
        @inbounds for d in 1:D
            u_rec -= qd[jstar][d] * grads[lstar][d]  # qd = x_j − y  ⇒  y − x_j = −qd
        end
        u_rec = math_max.(lo[lstar], math_min.(hi[lstar], u_rec))

        # Accumulate swept contributions
        @inbounds for k in 1:n
            l = lmap[qidx[k]]
            (l == 0 || is_removed[l]) && continue
            wk = wf(sum(abs2, qd[k]))
            dφ = wk / S_new - wk / S_old
            δV[l] += ω * dφ
            δm[l] += (ω * dφ) * u_rec
        end
        @inbounds for a in 1:n_add
            dφ = w_add[a] / S_new
            δV[n_loc + a] += ω * dφ
            δm[n_loc + a] += (ω * dφ) * u_rec
        end
    end

    # 4. New volumes and contents; exact residual correction
    V_before = zero(T); m_before = zero(State{M, T})
    @inbounds for l in 1:n_loc
        j = loc_idx[l]
        V_before += V[j]; m_before += V[j] * U[j]
    end
    Vn = zeros(T, n_loc + n_add)
    mn = zeros(State{M, T}, n_loc + n_add)
    V_after = zero(T); m_after = zero(State{M, T})
    @inbounds for l in 1:n_loc
        is_removed[l] && continue
        j = loc_idx[l]
        Vn[l] = max(V[j] + δV[l], T(1e-12) * V[j])
        mn[l] = V[j] * U[j] + δm[l]
        V_after += Vn[l]; m_after += mn[l]
    end
    @inbounds for a in 1:n_add
        Vn[n_loc + a] = max(δV[n_loc + a], zero(T))
        mn[n_loc + a] = δm[n_loc + a]
        V_after += Vn[n_loc + a]; m_after += mn[n_loc + a]
    end
    V_after > 0 || error("Remap produced no volume at $(ev.center); increase samples_per_dx.")
    εV, εm = V_before - V_after, m_before - m_after
    @inbounds for l in 1:(n_loc + n_add)
        (l <= n_loc && is_removed[l]) && continue
        frac  = Vn[l] / V_after
        mn[l] += frac * εm
        Vn[l] += frac * εV
    end

    # 5. DMP repair on the affected set (conservative redistribution)
    aff = Int[]                                         # local ids of survivors + added
    for l in 1:n_loc; is_removed[l] || push!(aff, l); end
    for a in 1:n_add; push!(aff, n_loc + a); end
    Un  = [mn[l] / Vn[l] for l in aff]
    lo_a = Vector{State{M, T}}(undef, length(aff)); hi_a = similar(lo_a)
    for (p, l) in enumerate(aff)
        if l <= n_loc
            lo_a[p], hi_a[p] = lo[l], hi[l]
        else
            # Bounds for a new particle: old states within R of it
            n = _collect_near!(pg, ev.added[l - n_loc], R_sq, qidx, qd)
            if n == 0
                lo_a[p], hi_a[p] = Un[p], Un[p]
            else
                a_lo = U[qidx[1]]; a_hi = a_lo
                for k in 2:n; a_lo = math_min.(a_lo, U[qidx[k]]); a_hi = math_max.(a_hi, U[qidx[k]]); end
                lo_a[p], hi_a[p] = a_lo, a_hi
            end
        end
    end
    _dmp_repair!(Un, [Vn[l] for l in aff], lo_a, hi_a)

    # 6. Write back
    for (p, l) in enumerate(aff)
        if l <= n_loc
            j = loc_idx[l]
            V[j] = Vn[l]; U[j] = Un[p]
        else
            push!(new_pos, ev.added[l - n_loc]); push!(new_rho, Un[p]); push!(new_vol, Vn[l])
        end
    end
    @inbounds for l in 1:n_loc; lmap[loc_idx[l]] = 0; end
    return nothing
end

"""
    _dmp_repair!(U, V, lo, hi)

Clips every `U[p]` to `[lo[p], hi[p]]` and redistributes the clipped content `Σ V (U − U_clip)`
to the particles that still have room, keeping `Σ V U` exact. If there is not enough room, the
remainder is spread proportionally to volume (bounds are then violated by that remainder).
"""
function _dmp_repair!(U::Vector{State{M, T}}, V::Vector{T}, lo::Vector{State{M, T}}, hi::Vector{State{M, T}}) where {M, T}
    n = length(U)
    Um = [MVector{M, T}(u) for u in U]
    for m in 1:M
        excess = zero(T)
        for p in 1:n
            u = Um[p][m]
            if u > hi[p][m]
                excess += V[p] * (u - hi[p][m]); Um[p][m] = hi[p][m]
            elseif u < lo[p][m]
                excess += V[p] * (u - lo[p][m]); Um[p][m] = lo[p][m]
            end
        end
        abs(excess) <= eps(T) && continue
        room = zero(T)
        for p in 1:n
            room += excess > 0 ? V[p] * (hi[p][m] - Um[p][m]) : V[p] * (Um[p][m] - lo[p][m])
        end
        if room >= abs(excess) && room > 0
            f = excess / room
            for p in 1:n
                r = excess > 0 ? (hi[p][m] - Um[p][m]) : (Um[p][m] - lo[p][m])
                Um[p][m] += f * r
            end
        else
            # Fill all rooms, spread the remainder by volume
            Vtot = sum(V)
            rem = excess - (excess > 0 ? room : -room)
            for p in 1:n
                Um[p][m] = excess > 0 ? hi[p][m] : lo[p][m]
                Um[p][m] += rem / Vtot
            end
        end
    end
    for p in 1:n; U[p] = State{M, T}(Um[p]); end
    return nothing
end

# -------------------------------------------------------------------------
# Topology change: compaction of removed particles, append of new ones
# -------------------------------------------------------------------------
function _apply_topology!(pg::ParticleGrid{D, M, T}, remove_mask::AbstractVector{Bool},
                          new_pos::Vector{Space{D, T}}, new_rho::Vector{State{M, T}}, new_vol::Vector{T},
                          max_order::Int) where {D, M, T}
    N     = pg.meta.N
    n_add = length(new_pos)
    core  = pg.core
    N_new = N - count(view(remove_mask, 1:N)) + n_add
    ensure_capacity!(pg, N_new)
    ensure_particle_capacity!(pg.neighbor, N_new)

    w = 1
    @inbounds for i in 1:N
        remove_mask[i] && continue
        if w != i
            core.positions[w]       = core.positions[i]
            core.is_boundary[w]     = core.is_boundary[i]
            core.tags[w]            = core.tags[i]
            core.particle_orders[w] = core.particle_orders[i]
            core.volumes[w]         = core.volumes[i]
            pg.rhos[w]              = pg.rhos[i]
            pg.curvatures[w]        = pg.curvatures[i]
        end
        w += 1
    end
    @inbounds for a in 1:n_add
        core.positions[w]       = new_pos[a]
        core.is_boundary[w]     = false
        core.tags[w]            = 0
        core.particle_orders[w] = max_order
        core.volumes[w]         = new_vol[a]
        pg.rhos[w]              = new_rho[a]
        pg.curvatures[w]        = zero(State{M, T})
        w += 1
    end
    pg.meta.N          = N_new
    pg.meta.N_interior = count(!, view(core.is_boundary, 1:N_new))
    pg.meta.N_ghost    = N_new - pg.meta.N_interior
    return nothing
end
