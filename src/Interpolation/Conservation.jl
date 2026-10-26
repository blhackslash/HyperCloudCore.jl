# =========================================================================
# CONSERVATION DIAGNOSTIC (host only)
# =========================================================================
"""
    conservation_defect(muscl::MUSCL, pg::ParticleGrid) -> NamedTuple

Pair term of the conservation defect of the divergence weights (see the MLS interpolation docs). For every
neighbour slot `(i, j)` the face vector `S_ij = 2 V_i c_ij` is the coefficient with which the interface flux
`F*_ij` enters the mass budget of particle `i`; exact conservation needs `S_ij = -S_ji`. The weights are
those of the divergence fit each particle currently uses (its `particle_orders` entry, i.e. MOOD drops
included, and its neighbour count, as in `_divergence_index`), rebuilt on the host with the volumes of `pg`.

Returns `(; particle, degree, equal, mixed, onesided, total)`:
- `particle[i]`: relative defect `Σ_j |S_ij + S_ji| / Σ_j |S_ij|` of particle `i`;
- `degree[i]`: the divergence degree particle `i` uses;
- `equal`, `mixed`: `(; pairs, rel)` over the unordered pairs whose particles use the same / different
  degrees: the number of pairs and `Σ |S_ij + S_ji| / Σ ½(|S_ij| + |S_ji|)`;
- `onesided`: `(; pairs, rel)` over the slots whose partner does not list the particle back (`S_ji = 0`,
  the whole face vector is a defect);
- `total`: `(; pairs, rel)` over all of them.

A diagnostic for grids on the host (it allocates and is not a kernel); a device grid has to be copied back
first.
"""
function conservation_defect(muscl::MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER},
                             pg::ParticleGrid{D, M, T}) where {D, M, T, B_LEN, MAX_ORDER, DIV_ORDER}
    N  = pg.meta.N
    nb = pg.neighbor
    ranges, indices = nb.ranges, nb.indices
    invL = one(T) / minimum(pg.meta.dx)
    vol  = pg.core.volumes

    S      = zeros(Space{D, T}, length(indices))
    degree = Vector{Int}(undef, N)
    for i in 1:N
        sl = ranges[i]
        q  = _divergence_index(Val(DIV_ORDER), Val(D), pg.core.particle_orders[i], length(sl)) - 1
        degree[i] = q
        _face_vectors!(S, sl, nb.distances, nb.weights, invL, 2 * vol[i], q, Val(D))
    end

    # (neighbour index, slot) of every particle, sorted by index, for the reverse lookup
    sorted = [sort!([(indices[k], k) for k in ranges[j]]; by = first) for j in 1:N]
    slot_of(j, i) = (p = searchsortedfirst(sorted[j], (i, 0); by = first);
                     p <= length(sorted[j]) && sorted[j][p][1] == i ? sorted[j][p][2] : 0)

    particle = zeros(T, N)
    n_eq = n_mx = n_os = 0
    d_eq = d_mx = d_os = zero(T)          # defect sums
    s_eq = s_mx = s_os = zero(T)          # face-vector magnitude sums (the reference)
    for i in 1:N
        num = den = zero(T)
        for k in ranges[i]
            j   = indices[k]
            mag = norm(S[k])
            den += mag
            kk  = slot_of(j, i)
            if kk == 0
                num += mag
                n_os += 1; d_os += mag; s_os += mag
            else
                d = norm(S[k] + S[kk])
                num += d
                if i < j                                       # every unordered pair once
                    half = (mag + norm(S[kk])) / 2
                    if degree[i] == degree[j]
                        n_eq += 1; d_eq += d; s_eq += half
                    else
                        n_mx += 1; d_mx += d; s_mx += half
                    end
                end
            end
        end
        particle[i] = den > 0 ? num / den : zero(T)
    end
    rel(d, s) = s > 0 ? d / s : zero(T)
    return (; particle, degree,
            equal    = (; pairs = n_eq, rel = rel(d_eq, s_eq)),
            mixed    = (; pairs = n_mx, rel = rel(d_mx, s_mx)),
            onesided = (; pairs = n_os, rel = rel(d_os, s_os)),
            total    = (; pairs = n_eq + n_mx + n_os, rel = rel(d_eq + d_mx + d_os, s_eq + s_mx + s_os)))
end

# Face vectors `2 V_i c_ij` of one particle for the divergence fit of degree `q` (intercept basis, as in
# `_mls_divergence`): rows 2..D+1 of `inv(N) w_j p_j`, scaled by `invL`. Zero if the fit is not solvable.
function _face_vectors!(S, sl, dists, wts, invL::T, twoV::T, q::Int, ::Val{D}) where {T, D}
    B = basis_count(D, q) + 1
    inv_N, ok = _inverse_moment(sl, dists, wts, invL, Val(q), zero(SMatrix{B, B, T, B * B}), Val(true))
    @inbounds for k in sl
        if ok
            c = inv_N * (wts[k] * _intercept_basis(Val(q), dists[k] * invL))
            S[k] = (twoV * invL) * Space{D, T}(ntuple(d -> c[d + 1], Val(D)))
        else
            S[k] = zero(Space{D, T})
        end
    end
    return nothing
end
