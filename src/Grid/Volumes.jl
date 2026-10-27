export VolumeEstimator, ShepardVolumes, MLSVolumes, update_volumes!, domain_volume, total_mass

# =========================================================================
# PARTICLE VOLUMES — a partition-of-unity quadrature on the point cloud
# =========================================================================
#
# Collocation particles carry no mass. To make inserting/merging particles exactly conservative
# (and to have a well-defined discrete mass at all) every particle gets a positive weight V_i with
#
#     Σ_i V_i = |Ω|,        Σ_i V_i f(x_i) ≈ ∫_Ω f dx .
#
# V_i = ∫_Ω φ_i(x) dx for shape functions φ_i forming a partition of unity (Σ_i φ_i ≡ 1).
# No Voronoi tessellation is stored or built; only the scalar V_i lives on `pg.core.volumes`.
#
#   ShepardVolumes   φ_i(x) = w(|x−x_i|) / Σ_k w(|x−x_k|).   Particle-based evaluation, one
#                    neighbour sweep, second order on quasi-uniform clouds. Default, cheap enough
#                    to recompute after every step of a moving grid, but biased (independently of
#                    the resolution) where the spacing varies, i.e. on moving grids.
#   MLSVolumes{P}    φ_i = reproducing-kernel (MLS) shape functions of degree P, integrated on a
#                    throwaway background lattice. Σ V_i q(x_i) = ∫ q exactly for polynomials q of
#                    degree ≤ P (up to the lattice error), i.e. a quadrature of order P+1. Costs a
#                    serial lattice sweep (3–20 time steps when re-estimated every step).

abstract type VolumeEstimator end

"""
    ShepardVolumes(; sweeps = 1)

Kernel-density volumes `V_i ∝ 1 / S_i`, `S_i = w(0) + Σ_j w_ij`, followed by `sweeps`
partition-of-unity smoothing sweeps `V_i ← Σ_j (w_ij / S_j) V_j` (which preserve Σ V exactly),
and a final renormalisation to `Σ V = |Ω|`. Requires a kernel that is finite at zero distance
(`ExponentialWeightFunction`; `InverseWeightFunction` is not suitable).

Accurate on (quasi-)uniform clouds. On a cloud whose spacing varies smoothly (a moving grid) `S_i` is a
lattice sum over a fixed number of neighbours, which is not proportional to the local density, so
`Σ V_i f(x_i)` has a bias that depends on the local stretch and does **not** shrink with the resolution
(−2e-3 for a smooth profile on a grid stretched by ±50 %, at every N); use [`MLSVolumes`](@ref) where the
discrete mass of a moving grid matters, at a higher cost per update.
"""
struct ShepardVolumes <: VolumeEstimator
    sweeps::Int
end
ShepardVolumes(; sweeps::Int = 1) = ShepardVolumes(sweeps)

"""
    MLSVolumes(P; samples_per_dx = 2)

`V_i = ∫_Ω φ_i dx` with MLS shape functions reproducing polynomials of degree `P`, integrated by a
midpoint rule on a background lattice with `samples_per_dx` points per particle spacing (the
lattice is not stored). `P = 0` is the background-integrated Shepard partition.

A consistent quadrature on any cloud (the discrete mass of a moving grid converges), but the lattice sweep
is a serial host loop: as the re-estimation of a moving grid it costs about 3 (1-D) to 20 (2-D) explicit time
steps of a low-order scheme. On a device grid it runs on a host copy (only the volumes are copied back).
"""
struct MLSVolumes{P} <: VolumeEstimator
    samples_per_dx::Int
end
MLSVolumes(P::Int; samples_per_dx::Int = 2) = MLSVolumes{P}(samples_per_dx)

# -------------------------------------------------------------------------
# Domain volume |Ω|
# -------------------------------------------------------------------------
"""
    domain_volume(geom::GeometricDomain) -> T

Volume of the physical region. Analytic for the built-in shapes; define a method for custom
geometry labels (or integrate `geom.is_interior` on a lattice).
"""
domain_volume(geom::GeometricDomain{Val{:rectangular}}) = prod(geom.maxs .- geom.mins)

function domain_volume(geom::GeometricDomain{Val{:spherical}, D, T}) where {D, T}
    r = (geom.maxs[1] - geom.mins[1]) / 2
    return D == 1 ? 2r : D == 2 ? T(π) * r^2 : T(4π / 3) * r^3
end

domain_volume(geom::GeometricDomain{GEO}) where {GEO} =
    error("domain_volume is not defined for geometry label $GEO; add a method or integrate `geom.is_interior`.")

domain_volume(pg::ParticleGrid) = domain_volume(pg.geometry)

"""
    total_mass(pg) -> State{M, T}

Discrete conserved content `Σ_i V_i U_i` over all particles of the current cloud.
"""
function total_mass(pg::ParticleGrid{D, M, T}) where {D, M, T}
    N = pg.meta.N
    N == 0 && return zero(State{M, T})
    return mapreduce((v, u) -> v * u, +, view(pg.core.volumes, 1:N), view(pg.rhos, 1:N); init = zero(State{M, T}))
end

# -------------------------------------------------------------------------
# Point queries on the spatial hash (valid after `pg.neighbor(pg)`)
# -------------------------------------------------------------------------
@inline function _wrapped_diff(a::Space{D, T}, b::Space{D, T}, L::Space{D, T}, invL::Space{D, T}) where {D, T}
    return Space{D, T}(ntuple(Val(D)) do d
        dx = a[d] - b[d]
        dx - L[d] * round(dx * invL[d])
    end)
end

"""
    _collect_near!(pg, y, R_sq, idx, dvec) -> n

Fills `idx[1:n]` with the particles within `√R_sq` of the point `y` and `dvec[1:n]` with the
wrapped offsets `x_j − y`. Walks every coarse bin that intersects the search ball (not just the
3ᴰ neighbouring bins), so any radius is allowed; periodic axes wrap, others clamp. Valid after
`pg.neighbor(pg)` has (re)built the bins.
"""
function _collect_near!(pg::ParticleGrid{D, M, T}, y::Space{D, T}, R_sq::T,
                        idx::Vector{Int}, dvec::Vector{Space{D, T}}) where {D, M, T}
    bins   = pg.bins
    pos    = pg.core.positions
    L, Li  = pg.domain.L_wrap, pg.domain.invL_wrap
    is_per = pg.domain.is_periodic
    dims   = bins.coarse_dims
    r      = sqrt(R_sq)

    ranges = ntuple(Val(D)) do d
        lo = floor(Int, (y[d] - r - bins.mins[d]) / bins.coarse_size[d]) + 1
        hi = floor(Int, (y[d] + r - bins.mins[d]) / bins.coarse_size[d]) + 1
        if is_per[d]
            span = min(hi - lo + 1, dims[d])          # never visit a bin twice
            lo:(lo + span - 1)
        else
            max(lo, 1):min(hi, dims[d])
        end
    end
    li = LinearIndices(dims)

    n = 0
    for I in CartesianIndices(ranges)
        cart = ntuple(d -> is_per[d] ? mod1(I[d], dims[d]) : I[d], Val(D))
        b = li[CartesianIndex(cart)]
        @inbounds for s in bins.cell_start[b]:(bins.cell_start[b + 1] - 1)
            j = bins.cell_particles[s]
            d = _wrapped_diff(pos[j], y, L, Li)
            if sum(abs2, d) <= R_sq
                n += 1
                if n > length(idx); push!(idx, j); push!(dvec, d); else; idx[n] = j; dvec[n] = d; end
            end
        end
    end
    return n
end

# Wrap a point into the primary cell on periodic axes (no-op otherwise)
@inline _wrap_point(pg::ParticleGrid, y::Space) = _wrap_coords(y, pg.domain.is_periodic, pg.domain.L, pg.domain.canvas_mins)
@inline function _wrap_coords(y::Space{D, T}, is_per, Lv, mins) where {D, T}
    return Space{D, T}(ntuple(d -> is_per[d] ? mins[d] + mod(y[d] - mins[d], Lv[d]) : y[d], Val(D)))
end

# -------------------------------------------------------------------------
# Shepard volumes
# -------------------------------------------------------------------------
function update_volumes!(pg::ParticleGrid{D, M, T}, est::ShepardVolumes) where {D, M, T}
    N   = pg.meta.N
    N == 0 && return nothing
    nb  = pg.neighbor
    V   = pg.core.volumes
    S   = pg.shared.float_buffer
    Vt  = pg.shared.float_buffer2
    wf  = nb.weight_func
    w0  = wf(zero(T))
    if !isfinite(w0) || w0 > T(1e8) * wf(pg.meta.dx[1]^2)
        # Singular kernel (e.g. InverseWeightFunction): the Shepard partition degenerates to a delta.
        @warn "ShepardVolumes needs a kernel that is bounded at zero distance; using uniform volumes |Ω|/N." maxlog = 1
        fill!(view(V, 1:N), domain_volume(pg) / N)
        return nothing
    end

    backend = get_backend()
    foreach_particle(_shepard_sums!, backend, N, (S, V, nb.ranges, nb.weights, w0, one(T)))
    for _ in 1:est.sweeps
        foreach_particle(_shepard_sweep!, backend, N, (Vt, V, S, nb.ranges, nb.indices, nb.weights, w0))
        copyto!(V, 1, Vt, 1, N)
    end

    _renormalise_volumes!(pg)
    return nothing
end

# Kernel bodies of the Shepard partition (see Backend.jl).
@inline function _shepard_sums!(i, S, V, ranges, weights, w0, one_T)
    s = w0
    @inbounds for k in ranges[i]; s += weights[k]; end
    S[i] = s
    V[i] = one_T / s
    return nothing
end

@inline function _shepard_sweep!(i, Vt, V, S, ranges, indices, weights, w0)
    acc = w0 / S[i] * V[i]
    @inbounds for k in ranges[i]
        j = indices[k]
        acc += weights[k] / S[j] * V[j]
    end
    Vt[i] = acc
    return nothing
end

# Clamps negative volumes to zero and rescales to Σ V = |Ω| (broadcasts and a reduction: runs on a device grid).
function _renormalise_volumes!(pg::ParticleGrid{D, M, T}) where {D, M, T}
    V = view(pg.core.volumes, 1:pg.meta.N)
    V .= max.(V, zero(T))
    V .*= domain_volume(pg) / sum(V)
    return nothing
end

# -------------------------------------------------------------------------
# MLS (reproducing-kernel) volumes on a background lattice
# -------------------------------------------------------------------------
@inline _full_basis(::Val{0}, d::Space{D, T}) where {D, T} = SVector{1, T}(one(T))
@inline _full_basis(::Val{P}, d::Space{D, T}) where {P, D, T} = vcat(SVector{1, T}(one(T)), build_basis(Val(P), d))

function update_volumes!(pg::ParticleGrid{D, M, T}, est::MLSVolumes{P}) where {D, M, T, P}
    N    = pg.meta.N
    N == 0 && return nothing
    parent(pg.core.positions) isa Array || return _update_volumes_on_host!(pg, est)
    V    = pg.core.volumes
    fill!(view(V, 1:N), zero(T))
    geom = pg.geometry
    wf   = pg.neighbor.weight_func
    R_sq = pg.meta.R^2
    invL = one(T) / minimum(pg.meta.dx)
    hq   = minimum(pg.meta.dx) / est.samples_per_dx

    nq = ntuple(d -> max(1, round(Int, (geom.maxs[d] - geom.mins[d]) / hq)), Val(D))
    h  = ntuple(d -> (geom.maxs[d] - geom.mins[d]) / nq[d], Val(D))
    ω  = prod(h)

    B_full = (P == 0 ? 0 : typeof(basis_length(Val(D), Val(P))).parameters[1]) + 1
    idx  = Int[]
    dvec = Space{D, T}[]

    for I in CartesianIndices(nq)
        y = Space{D, T}(ntuple(d -> geom.mins[d] + (I[d] - T(0.5)) * h[d], Val(D)))
        geom.is_interior(y) || continue
        n = _collect_near!(pg, y, R_sq, idx, dvec)
        n == 0 && continue
        _accumulate_mls_shapes!(V, idx, dvec, n, wf, Val(P), Val(B_full), invL, ω)
    end

    _renormalise_volumes!(pg)
    return nothing
end

# The lattice sweep of `MLSVolumes` is a host loop (point queries on the cell list). On a device grid (a moving
# grid re-estimates every step) it runs on a host copy of the grid and only the volumes are copied back.
function _update_volumes_on_host!(pg::ParticleGrid, est::VolumeEstimator)
    N  = pg.meta.N
    hg = device_grid(Array, pg)
    update_volumes!(hg, est)
    copyto!(parent(pg.core.volumes), 1, parent(hg.core.volumes), 1, N)
    return nothing
end

# φ_i(y) = w_i p_i' a,  (Σ_k w_k p_k p_k') a = e_1,  p_k = (1, basis((x_k − y)/dx))
@inline function _accumulate_mls_shapes!(V, idx, dvec, n, wf, ::Val{P}, ::Val{B}, invL::T, ω::T) where {P, B, T}
    Mm = zero(SMatrix{B, B, T, B * B})
    @inbounds for k in 1:n
        p = _full_basis(Val(P), dvec[k] * invL)
        Mm += wf(sum(abs2, dvec[k])) * (p * p')
    end
    C, ok = _cholesky_moment(Mm)                  # regularised only if the neighbourhood is degenerate
    if !ok
        # Degenerate neighbourhood: fall back to the Shepard partition at this point
        s = zero(T)
        @inbounds for k in 1:n; s += wf(sum(abs2, dvec[k])); end
        @inbounds for k in 1:n; V[idx[k]] += ω * wf(sum(abs2, dvec[k])) / s; end
        return nothing
    end
    e1 = SVector{B, T}(ntuple(b -> b == 1 ? one(T) : zero(T), Val(B)))
    a  = C \ e1
    @inbounds for k in 1:n
        p = _full_basis(Val(P), dvec[k] * invL)
        V[idx[k]] += ω * wf(sum(abs2, dvec[k])) * dot(p, a)
    end
    return nothing
end
