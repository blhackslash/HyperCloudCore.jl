# =========================================================================
# PAIR-CONSISTENT DIVERGENCE NEXT TO MOOD DROPS ("halo correction")
# =========================================================================
# A particle k at a reduced order uses lower-degree divergence weights than its full-order neighbour j, so
# the face vectors of the pair, S_kj = 2 V_k c_kj and S_jk = 2 V_j c_jk, do not cancel at all (their defect
# exceeds the face magnitude, see `conservation_defect`): a front of dropped particles loses mass. The fix
# keeps the dropped particle as it is (its low-degree fit is what MOOD relies on for stability) and lets the
# full-order neighbour use the mirrored face vector, S_jk := -S_kj, for every lower-order neighbour. The
# first-order consistency of j's weights (zero sum, exact gradient of a linear flux) is then restored with a
# degree-1 correction over its remaining slots, w_jl (Λ0 + Λ1 x_jl), which never touches the mirrored slots.
# Mixed pairs become exactly antisymmetric and j keeps a consistent divergence; it gives up its higher-order
# consistency next to the drop, where the solution is not smooth anyway.
#
# Two kernel paths: `_face_body!` (launched by `face_prepass!` before the flux pass) writes the face vectors
# of every lower-order particle that has a higher-order neighbour into the slot buffer `faces`; the flux pass
# of a particle with such a neighbour takes `_halo_divergence` instead of the cached or dynamic fit. Everything
# is accumulated over the slots in one sweep (moment matrix, flux block, the sums the residuals need), so the
# per-slot cost is that of a dynamic fit.

# Divergence degree of particle `p` (as `_divergence_index`, minus one).
@inline _div_degree(orders, ranges, p, ::Val{DO}, ::Val{D}) where {DO, D} =
    _divergence_index(Val(DO), Val(D), orders[p], length(ranges[p])) - 1

# `j` is a lower-order neighbour whose face vectors `i` mirrors: a lower particle order (a MOOD drop, not a
# stencil too small for the degree), a lower divergence degree, and not a boundary particle (no divergence).
@inline function _mirrors(orders, ranges, is_boundary, i, j, q_i, valDO, valD)
    is_boundary[j] && return false
    orders[j] < orders[i] || return false
    return _div_degree(orders, ranges, j, valDO, valD) < q_i
end

# Slot of particle `i` in the list `sl` (`0` if `i` is not listed: a one-sided pair).
@inline function _slot_of(sl, indices, i)
    @inbounds for s in sl
        indices[s] == i && return s
    end
    return 0
end

@inline function _has_lower_neighbour(pg, i, div_idx, nb_slice, valDO, valD)
    orders, ranges, indices, is_boundary = pg.core.particle_orders, pg.neighbor.ranges, pg.neighbor.indices, pg.core.is_boundary
    q_i = div_idx - 1
    @inbounds for k in nb_slice
        _mirrors(orders, ranges, is_boundary, i, indices[k], q_i, valDO, valD) && return true
    end
    return false
end

"""
    face_prepass!(interp, pg, mask, full)

Before the flux pass of a right-hand side: writes the face vectors `2 V_k c_kj` of every particle `k` whose
divergence weights a higher-order neighbour mirrors (pair-consistent divergence next to MOOD drops, see
`PairConsistent.jl`), for all interior particles (`full`) or those in `mask`. A no-op for schemes without it.
"""
@inline face_prepass!(::DivergenceInterpolator, pg, mask, full) = nothing

function face_prepass!(muscl::MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER}, pg::ParticleGrid{D, M, T}, mask, full::Bool) where {D, M, T, B_LEN, MAX_ORDER, DIV_ORDER}
    nb = pg.neighbor
    ensure_capacity!(muscl.faces, length(nb.indices))
    invL = one(T) / minimum(pg.meta.dx)
    qmax = Val(_full_div_idx(Val(MAX_ORDER), Val(DIV_ORDER), Val(D)) - 1)
    foreach_particle(_face_body!, get_backend(), pg.meta.N,
        (muscl.faces, pg.core.particle_orders, nb.ranges, nb.indices, nb.distances, nb.weights, pg.core.volumes,
         pg.core.is_boundary, mask, full, invL, Val(DIV_ORDER), Val(D), qmax))
    return nothing
end

# Kernel body (see Backend.jl): face vectors of particle `k` if a higher-order neighbour mirrors them.
@inline function _face_body!(k, faces, orders, ranges, indices, dists, wts, volumes, is_boundary, mask, full, invL, valDO, valD, qmax)
    (is_boundary[k] || !(full || mask[k])) && return nothing
    q_k = _div_degree(orders, ranges, k, valDO, valD)
    needed = false
    @inbounds for s in ranges[k]
        j = indices[s]
        if !is_boundary[j] && orders[j] > orders[k] && _div_degree(orders, ranges, j, valDO, valD) > q_k
            needed = true
            break
        end
    end
    needed || return nothing
    _dispatch_faces(q_k, qmax, faces, ranges[k], dists, wts, 2 * volumes[k], invL, valD)
    return nothing
end

# Runtime degree -> static degree of the face vectors (degrees below the full one).
@generated function _dispatch_faces(q::Int, ::Val{QMAX}, faces, sl, dists, wts, twoV, invL, ::Val{D}) where {QMAX, D}
    top  = max(QMAX - 1, 1)
    expr = :(_write_faces!(Val($top), faces, sl, dists, wts, twoV, invL, Val($D)))
    for d in (top - 1):-1:1
        expr = :(q == $d ? _write_faces!(Val($d), faces, sl, dists, wts, twoV, invL, Val($D)) : $expr)
    end
    return expr
end

@inline _write_faces!(::Val{q}, faces, sl, dists, wts, twoV::T, invL::T, ::Val{D}) where {q, T, D} =
    _write_faces_b!(basis_length(Val(D), Val(q)), Val(q), faces, sl, dists, wts, twoV, invL, Val(D))

# Face vectors 2 V_k c_kj of the degree-q divergence fit (intercept basis, as `_mls_divergence`): rows 2..D+1
# of N⁻¹ w_j p_j, scaled by invL. Zero if the fit is not solvable (the particle's divergence is zero then too).
@inline function _write_faces_b!(::Val{B0}, ::Val{q}, faces, sl, dists, wts, twoV::T, invL::T, ::Val{D}) where {B0, q, T, D}
    B   = B0 + 1
    N_s = zero(SMatrix{B, B, T, B * B})
    @inbounds for s in sl
        p = _intercept_basis(Val(q), dists[s] * invL)
        N_s += wts[s] * (p * p')
    end
    C, ok = _cholesky_moment(N_s)
    @inbounds for s in sl
        if ok
            c = C \ (wts[s] * _intercept_basis(Val(q), dists[s] * invL))
            faces[s] = (twoV * invL) * Space{D, T}(ntuple(d -> c[d + 1], Val(D)))
        else
            faces[s] = zero(Space{D, T})
        end
    end
    return nothing
end

# Runtime divergence entry -> static degree of the halo divergence.
@generated function _dispatch_halo(div_idx::Int, ::Val{IMAX}, faces, i, nb_slice, pg, dists, wts, df_flux, ::Val{DO}) where {IMAX, DO}
    expr = :(_halo_divergence(Val($(IMAX - 1)), faces, i, nb_slice, pg, dists, wts, df_flux, Val($DO)))
    for idx in (IMAX - 1):-1:2
        expr = :(div_idx == $idx ? _halo_divergence(Val($(idx - 1)), faces, i, nb_slice, pg, dists, wts, df_flux, Val($DO)) : $expr)
    end
    return expr
end

@inline _halo_divergence(::Val{q}, faces, i, nb_slice, pg::ParticleGrid{D, M, T}, dists, wts, df_flux, valDO) where {q, D, M, T} =
    _halo_divergence_b(basis_length(Val(D), Val(q)), Val(q), faces, i, nb_slice, pg, dists, wts, df_flux, valDO)

# Divergence of particle `i` (degree q, before the factor 2) with mirrored face vectors on its lower-order
# neighbours and the degree-1 consistency correction over the other ("free") slots. In the scaled coordinates
# x = d invL the fit's weights are ĉ (Σ ĉ = 0, Σ ĉ xᵀ = I); a mirrored weight is ĉ = -S_ji / (2 V_i invL).
@inline function _halo_divergence_b(::Val{B0}, ::Val{q}, faces, i, nb_slice, pg::ParticleGrid{D, M, T}, dists, wts,
                                    df_flux::AbstractVector{Flux{D, M, T}}, valDO) where {B0, q, D, M, T}
    B  = B0 + 1
    DM = D * M
    orders, ranges, indices, is_boundary = pg.core.particle_orders, pg.neighbor.ranges, pg.neighbor.indices, pg.core.is_boundary
    invL   = one(T) / minimum(pg.meta.dx)
    mscale = -one(T) / (2 * pg.core.volumes[i] * invL)
    N_s = zero(SMatrix{B, B, T, B * B})
    b_s = zero(SMatrix{B, DM + 1 + D, T, B * (DM + 1 + D)})     # per slot w p [row | 1 | xᵀ]
    Mc  = zero(SMatrix{D + 1, D + 1, T, (D + 1) * (D + 1)})     # degree-1 moments of the free slots
    Qc  = zero(SMatrix{D + 1, DM, T, (D + 1) * DM})              # Σ w p̂ rowᵀ over the free slots
    mir = zero(SMatrix{D, DM, T, D * DM})                        # mirrored weights applied to their flux rows
    m0  = zero(Space{D, T})                                      # Σ mirrored ĉ
    m1  = zero(SMatrix{D, D, T, D * D})                          # Σ mirrored ĉ xᵀ
    @inbounds for k in nb_slice
        j   = indices[k]
        x   = dists[k] * invL
        F   = df_flux[k]
        row = SVector{DM, T}(ntuple(n -> F[(n - 1) ÷ M + 1][(n - 1) % M + 1], Val(DM)))
        kk  = _mirrors(orders, ranges, is_boundary, i, j, q, valDO, Val(D)) ? _slot_of(ranges[j], indices, i) : 0
        w   = wts[k]
        p   = _intercept_basis(Val(q), x)
        N_s += w * (p * p')                                      # the fit itself uses every neighbour
        if kk == 0
            ph = vcat(SVector{1, T}(one(T)), x)
            b_s += w * (p * vcat(row, SVector{1, T}(one(T)), x)')
            Mc  += w * (ph * ph')
            Qc  += w * (ph * row')
        else
            a   = mscale * faces[kk]
            mir += a * row'
            m0  += a
            m1  += a * x'
        end
    end
    C, ok = _cholesky_moment(N_s)
    ok || return zero(State{M, T})
    X  = C \ b_s                                                 # B × (DM + 1 + D)
    R  = SMatrix{D, DM, T}(ntuple(n -> X[(n - 1) % D + 2, (n - 1) ÷ D + 1], Val(D * DM)))
    r0 = Space{D, T}(ntuple(d -> X[d + 1, DM + 1], Val(D))) + m0
    r1 = SMatrix{D, D, T}(ntuple(n -> X[(n - 1) % D + 2, DM + 2 + (n - 1) ÷ D], Val(D * D))) + m1 - I
    _, okc = _cholesky_moment(Mc)
    corr = okc ? -(hcat(r0, r1) * inv(Mc)) * Qc : zero(SMatrix{D, DM, T, D * DM})
    total = R + mir + corr
    return State{M, T}(ntuple(Val(M)) do m
        acc = zero(T)
        for d in 1:D
            acc += total[d, (d - 1) * M + m]
        end
        acc * invL
    end)
end
