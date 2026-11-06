export ParticleGrid, get_time_step

include("Domains.jl")

# ---------------------------------------------------------
# 1. Grid Metadata
# ---------------------------------------------------------
"""
    GridMetadata{D, T}
    SharedBuffers{D, M, T, ...}
    NeighborData{D, T, WF, ...}
    ReorderData{D}
    GlobalBins{D, T, N_OFF, ...}
    ParticleGridCore{D, T, ...}

Internal structures managing the state, topology, and execution context of the mesh-free solver.
- `GridMetadata`: Stores grid capacities, resolutions, and the active interaction radius.
- `SharedBuffers`: Thread-safe, pre-allocated workspaces for floating-point, integer, and boolean operations.
- `NeighborData`: Maintains the ranges, indices, weights, and distances for all active particle neighborhoods.
- `ReorderData`: Manages permutation buffers utilized for spatial sorting and memory optimization.
- `GlobalBins`: Defines the coarse spatial hashing bins and linked lists for the neighbor search algorithm.
- `ParticleGridCore`: Encapsulates the fundamental particle geometry including positions, boundary flags, and tags.
"""
mutable struct GridMetadata{D, T}
    N::Int
    N_interior::Int
    N_ghost::Int
    
    R::T
    max_dx::T
    dx::Space{D, T}
    
    interp_range_factor::T
    max_nb::Int
    # Incremented whenever neighbour lists, distances or weights change (neighbour search,
    # `refresh_geometry!`), so caches derived from the geometry can detect that they are stale.
    geometry_version::Int
end

"""
    GridMetadataView{D, T}

Immutable snapshot of a `GridMetadata`, the form in which the metadata travels into kernels
(see `DeviceAdapt.jl`). Kernels only read it.
"""
struct GridMetadataView{D, T}
    N::Int
    N_interior::Int
    N_ghost::Int
    R::T
    max_dx::T
    dx::Space{D, T}
    interp_range_factor::T
    max_nb::Int
    geometry_version::Int
end
GridMetadataView(m::GridMetadata{D, T}) where {D, T} = GridMetadataView{D, T}(
    m.N, m.N_interior, m.N_ghost, m.R, m.max_dx, m.dx, m.interp_range_factor, m.max_nb, m.geometry_version)

# The storage containers are parametric in their vector types, so the same structs hold host `Vector`s,
# device arrays (`CuArray`, ...) or the device views KernelAbstractions passes into kernels. Their arrays are
# only ever resized in place (`ensure_capacity!`), never rebound, so the structs are immutable.

# ---------------------------------------------------------
# 2. Shared Workspace Buffers
# ---------------------------------------------------------
struct SharedBuffers{D, M, T, VS <: AbstractVector{State{M, T}}, VP <: AbstractVector{Space{D, T}},
                     VT <: AbstractVector{T}, VB <: AbstractVector{Bool}, VI <: AbstractVector{Int}}
    rho_buffer::VS
    pos_buffer::VP
    float_buffer::VT
    float_buffer2::VT
    bit_buffer::VB
    int_buffer::VI
end
SharedBuffers{D, M, T}(rho, pos, f1::VT, f2::VT, bit, int) where {D, M, T, VT} =
    SharedBuffers{D, M, T, typeof(rho), typeof(pos), VT, typeof(bit), typeof(int)}(rho, pos, f1, f2, bit, int)

# ---------------------------------------------------------
# 3. Neighbor Search Context
# ---------------------------------------------------------
# The per-slot arrays (`indices`, `weights`, `distances`) are `Growable`s on the host grid: their length is
# the neighbour total, which every search may change, on the device too. The per-particle arrays grow
# only with adaptivity (host).
# `ranges[i]` is the slot range of particle `i`: a `UnitRange` in the row-major layout (host grids: a
# particle's slots are contiguous) or a `StepRange` with stride `N` in the slot-major layout (device
# grids: slot k of all particles is contiguous, so a wavefront reads consecutive addresses). Kernel
# bodies only ever iterate the range, so they serve both. See `NeighborLogic.jl`.
struct NeighborData{D, T, WF, VR <: AbstractVector{<:OrdinalRange{Int, Int}}, VI <: AbstractVector{Int},
                    GI <: AbstractVector{Int}, GT <: AbstractVector{T}, GP <: AbstractVector{Space{D, T}}}
    weight_func::WF
    ranges::VR
    indices::GI
    weights::GT
    distances::GP
    counts::VI
    offsets::VI
end
NeighborData{D, T, WF}(wf::WF, ranges, indices, weights, distances, counts::VI, offsets::VI) where {D, T, WF, VI} =
    NeighborData{D, T, WF, typeof(ranges), VI, typeof(indices), typeof(weights), typeof(distances)}(
        wf, ranges, indices, weights, distances, counts, offsets)

# ---------------------------------------------------------
# 4. Reordering / Sorting Context (host only)
# ---------------------------------------------------------
struct ReorderData{D}
    permutation::Vector{Int}
end

# Counting-sort cell list; see `GlobalBins(...)` in NeighborLogic.jl for the layout.
struct GlobalBins{D, T, N_OFF, VI <: AbstractVector{Int}, GI <: AbstractVector{Int}, VN <: AbstractVector{SVector{N_OFF, Int}}}
    mins::Space{D, T}
    maxs::Space{D, T}
    coarse_size::Space{D, T}
    coarse_dims::NTuple{D, Int}
    cell_start::VI          # per bin (+ dummy bin + end sentinel)
    cell_count::VI          # per bin (+ dummy bin)
    cell_particles::GI      # per particle: indices sorted by bin (`Growable` on the host grid)
    bin_of::GI              # per particle: its bin
    bin_neighbors::VN       # per bin: the 3^D window
end
GlobalBins{D, T, N_OFF}(mins, maxs, coarse_size, coarse_dims, cell_start::VI, cell_count::VI, cell_particles::GI, bin_of::GI,
                        bin_neighbors) where {D, T, N_OFF, VI, GI} =
    GlobalBins{D, T, N_OFF, VI, GI, typeof(bin_neighbors)}(mins, maxs, coarse_size, coarse_dims, cell_start, cell_count,
                                                            cell_particles, bin_of, bin_neighbors)

# ---------------------------------------------------------
# 6. Particle Grid Core (Geometry & Topology)
# ---------------------------------------------------------
struct ParticleGridCore{D, T, VP <: AbstractVector{Space{D, T}}, VB <: AbstractVector{Bool},
                        VT <: AbstractVector{T}, VI <: AbstractVector{Int}}
    positions::VP
    is_boundary::VB
    volumes::VT
    tags::VI

    # `volumes`: partition-of-unity quadrature weights, Σ V = |Ω| (see Volumes.jl). They define the
    # discrete mass Σ V U and make particle insertion/merging exactly conservative.
    # `particle_orders`: per-particle consistency order (read by every scheme, written by MOOD).
    particle_orders::VI
end
ParticleGridCore{D, T}(positions, is_boundary, volumes, tags::VI, particle_orders::VI) where {D, T, VI} =
    ParticleGridCore{D, T, typeof(positions), typeof(is_boundary), typeof(volumes), VI}(positions, is_boundary, volumes, tags, particle_orders)

"""
    ParticleGrid{D, M, T, WF, GM, N_OFF, Geom, META, CORE, SH, NB, RD, BINS, VS}

The primary orchestrator representing the active computational domain, tying together the physical geometry, state vectors, and mesh-free topology.

# Details
- Initializes the geometric and computational domains.
- Invokes the universal narrow-band point generator to allocate `ParticleGridCore`.
- Pre-allocates and assigns all internal components including `SharedBuffers`, `NeighborData`, `ReorderData`, and `GlobalBins`.
- Runs the initial spatial reordering and constructs the initial neighbor list upon instantiation.

The trailing type parameters are the storage types of the components. On the host they are the
`Vector`-based structs; inside a kernel the grid arrives as a bits type with device arrays, an immutable
`GridMetadataView` and `nothing` for the host-only parts (geometry, scratch buffers, reordering), see
`DeviceAdapt.jl`.
"""
struct ParticleGrid{D, M, T, WF, GM, N_OFF, Geom, META, CORE, SH, NB, RD, BINS, VS}
    meta::META
    geometry::Geom
    domain::ComputationalDomain{D, T}
    core::CORE
    shared::SH
    neighbor::NB
    reorder::RD
    bins::BINS
    mover::GM

    rhos::VS
    curvatures::VS
end

function ParticleGrid(meta, geometry, domain::ComputationalDomain{D, T}, core::ParticleGridCore{D, T}, shared,
                      neighbor::NeighborData{D, T, WF}, reorder, bins::GlobalBins{D, T, N_OFF}, mover::GM,
                      rhos::VS, curvatures::VS) where {D, T, WF, N_OFF, GM, M, VS <: AbstractVector{State{M, T}}}
    return ParticleGrid{D, M, T, WF, GM, N_OFF, typeof(geometry), typeof(meta), typeof(core), typeof(shared),
                        typeof(neighbor), typeof(reorder), typeof(bins), VS}(
        meta, geometry, domain, core, shared, neighbor, reorder, bins, mover, rhos, curvatures)
end

include("MLSWeightFunctions.jl")
include("NeighborLogic.jl")
include("BoundaryConditions.jl")
include("Reordering.jl")
include("Volumes.jl")
include("Adaptivity.jl")

export get_positions, get_weights, get_distances, get_neighbors

# --- 1. Unified Position Accesso rs (No more reinterpret hacks!) ---
@inline get_positions(pg::ParticleGrid) = pg.core.positions
@inline get_positions(sb::SharedBuffers) = sb.pos_buffer

# --- 2. Unified Neighbor Accessors ---
@inline get_weights(pg::ParticleGrid)   = pg.neighbor.weights
@inline get_distances(pg::ParticleGrid) = pg.neighbor.distances # Returns Space{D}
@inline get_neighbors(pg::ParticleGrid) = pg.neighbor.indices
# =========================================================================
# UNIFIED DISTANCE CALCULATIONS 
# =========================================================================
"""
    get_distance(pos, i, j, L_wrap, invL_wrap)

Calculates the strictly shortest distance vector between two particles, actively enforcing periodic wrapping limits defined by `L_wrap` and `invL_wrap`.
"""
@inline function get_distance(pos::AbstractVector{Space{D, T}}, i::Int, j::Int, L_wrap::Space{D, T}, invL_wrap::Space{D, T}) where {D, T}
    @inbounds begin
        p_i = pos[i]
        p_j = pos[j]
        
        return Space{D, T}(ntuple(Val(D)) do d
            dx = p_j[d] - p_i[d]
            dx - L_wrap[d] * round(dx * invL_wrap[d]) 
        end)
    end
end

function ParticleGrid(
    geom::GeometricDomain{GEO, D, T, FI, FT},
    nominal_dx::NTuple{D, T},
    weight_func::MLSWeightFunction,
    M::Int;
    randomness::Tuple = ntuple(i -> zero(T), D),
    rng = Random.default_rng(),
    mover = NoGridMover(),
    volume_estimator::VolumeEstimator = ShepardVolumes()
) where {D, T, FI, FT, GEO}
    
    R = get_cutoff(weight_func)
    dxs_f = Space{D, T}(nominal_dx...)
    max_dx = maximum(dxs_f)
    interp_range_factor = R / max_dx
    
    comp_domain = ComputationalDomain(geom, nominal_dx, interp_range_factor)
    
    positions, is_boundary, tags, volumes, _ = get_points(
        comp_domain, geom;
        nominal_dx = nominal_dx,
        interp_range_factor = interp_range_factor,
        randomness = randomness,
        rng = rng
    )
    _check_bc_tags(geom, tags, is_boundary)
    
    N = length(positions)
    N_interior = count(!, is_boundary)
    N_ghost = N - N_interior
    
    meta = GridMetadata{D, T}(N, N_interior, N_ghost, R, max_dx, dxs_f, interp_range_factor, 0, 0)
    
    # Every per-particle array is a `Growable`, so a grid can grow on a device as well (adaptivity copies
    # its result back into the device grid in place). The reordering state stays a host object.
    core = ParticleGridCore{D, T}(Growable(positions), Growable(is_boundary), Growable(volumes), Growable(tags), Growable(zeros(Int, N)))

    shared = SharedBuffers{D, M, T}(Growable(zeros(State{M, T}, N)), Growable(similar(positions)), Growable(zeros(T, N)),
                                    Growable(zeros(T, N)), Growable(zeros(Bool, N)), Growable(zeros(Int, N)))
    reorder = ReorderData{D}(collect(1:N))

    bins = GlobalBins(T, comp_domain.canvas_mins, comp_domain.canvas_maxs, R, N, comp_domain.is_periodic)
    neighbors = NeighborData{D, T, typeof(weight_func)}(
        weight_func, Growable(fill(1:0, N + 1)), Growable{Int}(Array), Growable{T}(Array), Growable{Space{D, T}}(Array),
        Growable(zeros(Int, N)), Growable(zeros(Int, N))
    )

    pg = ParticleGrid(meta, geom, comp_domain, core, shared, neighbors, reorder, bins, mover,
                      Growable(zeros(State{M, T}, N)), Growable(zeros(State{M, T}, N)))

    # The spatial reordering only needs positions and the canvas box, so the neighbour search is run
    # once, after the particles have been laid out in memory.
    pg.reorder(pg)
    pg.neighbor(pg)
    update_volumes!(pg, volume_estimator)
    _check_starved(pg)

    return pg
end

# An interior particle with fewer than D + 1 neighbours cannot carry even a linear fit: the schemes give it a zero
# divergence (it is frozen) or a lower order. Usually the support radius is too small for the jitter; warn once.
function _check_starved(pg::ParticleGrid{D}) where {D}
    n = count(i -> !pg.core.is_boundary[i] && length(pg.neighbor.ranges[i]) < D + 1, 1:pg.meta.N)
    n > 0 && @warn "$n interior particle(s) have fewer than $(D + 1) neighbours and cannot carry a linear fit " *
                   "(frozen or first order); increase the support radius (R = $(pg.meta.R)) or reduce the jitter."
    return nothing
end

# =========================================================================
# EXACT GEOMETRIC CFL CALCULATION FOR HIGH-ORDER MLS
# =========================================================================

@inline _get_Lambda(eq, U_i, ::Val{D}) where {D} = Space{D}(ntuple(d -> max_eigenvalue(eq, U_i, d), Val(D)))

@generated function _compute_cfl_coeffs(::Val{D}, ::Val{IO}, ::Val{B_LEN}, nb_slice, dist_vec, w_vec, invL, ::Type{T},
                                        rhos, indices, eq, Lambda_i) where {D, IO, B_LEN, T}
    quote
        N_s = zero(SMatrix{B_LEN, B_LEN, T, B_LEN * B_LEN})

        @inbounds for idx in 1:length(nb_slice)
            k = nb_slice[idx]
            p_s = build_basis(Val(IO), dist_vec[k] * invL)
            N_s += w_vec[k] * (p_s * p_s')
        end

        # 1. Cholesky factorisation, regularised only if the stencil is degenerate (`_cholesky_moment`)
        C, ok = _cholesky_moment(N_s)
        if !ok
            return zero(T)
        end

        # 2. Safely invert the moment matrix
        inv_N_s = inv(C)
        sum_c = zero(T)

        @inbounds for idx in 1:length(nb_slice)
            k = nb_slice[idx]
            w = w_vec[k]
            p_s = build_basis(Val(IO), dist_vec[k] * invL)

            # |c_ij| per axis times the pair speed max(λ_i, λ_j): the coefficient of neighbour j in the
            # first-order update, so a fast neighbour (a boundary particle of an inflow included) limits dt
            c = abs.(_linear_coeffs(inv_N_s, p_s, Val(D)) * w * invL)
            sum_c += dot(c, _pair_speed(Lambda_i, _get_Lambda(eq, rhos[indices[k]], Val(D))))
        end
        return sum_c
    end
end

# The wave speed a pair shares: the larger of the two particles' maximum eigenvalues, per axis.
@inline _pair_speed(a::SVector{D, T}, b::SVector{D, T}) where {D, T} = max.(a, b)

# MLS coefficients of the D linear basis terms for one neighbour (rows 1:D of inv_N_s times p_s).
# Static on purpose: it runs inside the CFL kernel (`_cfl_body!`), where an `MVector` accumulator does
# not compile on device backends. A separate function because a generated body cannot hold the closure.
@inline function _linear_coeffs(inv_N_s::SMatrix{B_LEN, B_LEN, T}, p_s::SVector{B_LEN, T}, ::Val{D}) where {B_LEN, T, D}
    return SVector{D, T}(ntuple(Val(D)) do d
        c_val = zero(T)
        @inbounds for j in 1:B_LEN
            c_val += inv_N_s[d, j] * p_s[j]
        end
        c_val
    end)
end

# Dispatch on the polynomial DEGREE of the divergence MLS (1..4, 1..3 in 3-D; higher degrees clamp,
# which slightly over-estimates dt for such schemes).
@generated function dispatch_cfl(degree::Int, ::Val{D}, nb_slice, dist_vec, w_vec, invL, ::Type{T}, rhos, indices, eq, Lambda) where {D, T}
    MAX_DEG = D == 3 ? 3 : 4          # keep the basis small enough for the inline Cholesky (35 terms at degree 4 in 3-D would not be)
    B = typeof(basis_length(Val(D), Val(MAX_DEG))).parameters[1]
    expr = :(_compute_cfl_coeffs(Val($D), Val($MAX_DEG), Val($B), nb_slice, dist_vec, w_vec, invL, T, rhos, indices, eq, Lambda))
    for deg in (MAX_DEG - 1):-1:1
        B = typeof(basis_length(Val(D), Val(deg))).parameters[1]
        expr = :(degree <= $deg ? _compute_cfl_coeffs(Val($D), Val($deg), Val($B), nb_slice, dist_vec, w_vec, invL, T, rhos, indices, eq, Lambda) : $expr)
    end
    return expr
end

"""
    get_time_step(pg::ParticleGrid, eq::HyperbolicPDE, main_grad)

The CFL-restricted time step (CFL number 1) over the active particle grid:
`Δt = min_i 1 / (D Σ_j Σ_d |c_ij,d| max(λ_d(U_i), λ_d(U_j)))`, with `λ_d` from `max_eigenvalue` and `c_ij` the
first-derivative weights of the divergence fit at particle `i` (degree `_divergence_degree(main_grad)`).

# Details
- Every neighbour enters with the pair speed `max(λ_i, λ_j)`, boundary particles included: a fast inflow state
  limits the step of its interior neighbours before it is inside (the first-order coefficients of a particle
  are `|c_ij| λ_pair`, so this is the local-extremum-diminishing bound).
- If the interpolator keeps the per-slot weights of its divergence fit for the current geometry
  (`_cfl_slot_weights`, a `MUSCL` after its first right-hand side), they are used and no moment matrix is
  factorised (50–100× cheaper); `cached_weights = false` always assembles and factorises the Taylor fit of
  degree `_divergence_degree(main_grad)` per particle (`_cfl_body!`), as every call did before. The two differ
  where the fits differ: for degree-5 (odd-order) divergence fits the weights in use sum to about 1.8× the
  Taylor fit's, i.e. the cached estimate is that much smaller; for degrees ≤ 3 they agree.
- Returns `Inf` for an empty grid or for particles without neighbours / without a solvable fit.
"""
@inline function get_time_step(pg::ParticleGrid{D, M, T}, eq::HyperbolicPDE, main_grad; cached_weights::Bool = true) where {D, M, T}
    # 1. Graceful exit for empty grids
    if pg.meta.N == 0
        return T(Inf)
    end

    dt_buffer = pg.shared.float_buffer

    # Only fill the active view to avoid touching dead memory
    fill!(view(dt_buffer, 1:pg.meta.N), T(Inf))

    nb = pg.neighbor
    sw = cached_weights ? _cfl_slot_weights(main_grad, pg) : nothing
    if sw === nothing
        degree = _divergence_degree(main_grad)
        invL   = one(T) / minimum(pg.meta.dx)
        foreach_particle(_cfl_body!, get_backend(), pg.meta.N,
            (dt_buffer, pg.core.is_boundary, nb.ranges, nb.indices, pg.rhos, eq, degree, Val(D), get_distances(pg), get_weights(pg),
             invL, T(1e-14), one(T) / D))
    else
        foreach_particle(_cfl_weights_body!, get_backend(), pg.meta.N,
            (dt_buffer, pg.core.is_boundary, nb.ranges, nb.indices, pg.rhos, eq, sw, Val(D), T(1e-14), one(T) / D))
    end

    # 2. Provide init=T(Inf) to prevent any future reduction errors
    return minimum(view(dt_buffer, 1:pg.meta.N); init=T(Inf))
end

"""
    _cfl_slot_weights(interp, pg) -> AbstractVector{Space} or nothing

The per-slot first-derivative weights of the divergence operator of `interp`, if it keeps them for the
current geometry of `pg` (a `MUSCL` after `prepare_geometry!`); `nothing` lets `get_time_step` assemble and
factorise the moment matrices itself.
"""
@inline _cfl_slot_weights(interp, pg) = nothing

# Kernel body of the CFL estimate (see Backend.jl): fit and sum of particle `i`.
@inline function _cfl_body!(i, dt_buffer, is_boundary, ranges, indices, rhos, eq, degree, valD, dist_vec, w_vec, invL, tol, inv_dim)
    is_boundary[i] && return nothing
    nb_slice = ranges[i]
    isempty(nb_slice) && return nothing
    Lambda = _get_Lambda(eq, rhos[i], valD)
    sum_c  = dispatch_cfl(degree, valD, nb_slice, dist_vec, w_vec, invL, typeof(invL), rhos, indices, eq, Lambda)
    sum_c > tol && (dt_buffer[i] = inv_dim / sum_c)
    return nothing
end

# Kernel body of the CFL estimate with cached per-slot weights: no factorisation, one eigenvalue per slot.
@inline function _cfl_weights_body!(i, dt_buffer, is_boundary, ranges, indices, rhos, eq, dweights, valD, tol, inv_dim)
    is_boundary[i] && return nothing
    nb_slice = ranges[i]
    isempty(nb_slice) && return nothing
    Lambda = _get_Lambda(eq, rhos[i], valD)
    sum_c  = zero(tol)
    @inbounds for k in nb_slice
        sum_c += dot(abs.(dweights[k]), _pair_speed(Lambda, _get_Lambda(eq, rhos[indices[k]], valD)))
    end
    sum_c > tol && (dt_buffer[i] = inv_dim / sum_c)
    return nothing
end

# =========================================================================
# OVERLOADED CAPACITY MANAGERS
# =========================================================================

"""
    ensure_capacity!(rd::ReorderData, req_capacity::Int)
"""
@inline function ensure_capacity!(rd::ReorderData{D}, req_capacity::Int) where {D}
    if length(rd.permutation) < req_capacity
        resize!(rd.permutation, ceil(Int, req_capacity * 1.25))
    end
    return nothing
end

"""
    ensure_capacity!(core::ParticleGridCore, req_capacity::Int)
"""
@inline function ensure_capacity!(core::ParticleGridCore{D, T}, req_capacity::Int) where {D, T}
    ensure_capacity!(core.positions, req_capacity)
    ensure_capacity!(core.is_boundary, req_capacity)
    ensure_capacity!(core.volumes, req_capacity)
    ensure_capacity!(core.tags, req_capacity)
    ensure_capacity!(core.particle_orders, req_capacity)
    return nothing
end

"""
    ensure_capacity!(bins::GlobalBins, req_particles::Int)
"""
@inline function ensure_capacity!(bins::GlobalBins, req_particles::Int)
    ensure_capacity!(bins.cell_particles, req_particles)
    ensure_capacity!(bins.bin_of, req_particles)
    return nothing
end

"""
    ensure_capacity!(nd::NeighborData, req_neighbors::Int)

Grows the per-slot arrays to hold `req_neighbors` entries (on the device too: they are `Growable`s).
"""
@inline function ensure_capacity!(nd::NeighborData, req_neighbors::Int)
    ensure_capacity!(nd.indices, req_neighbors)
    ensure_capacity!(nd.weights, req_neighbors)
    ensure_capacity!(nd.distances, req_neighbors)
    return nothing
end

"""
    ensure_particle_capacity!(nd::NeighborData, N::Int)

Grows the per-particle arrays of the neighbour structure (`ranges` needs `N + 1` entries).
"""
function ensure_particle_capacity!(nd::NeighborData, N::Int)
    old_len = length(nd.ranges)
    if old_len < N + 1
        ensure_capacity!(nd.ranges, N + 1)
        fill!(view(nd.ranges, (old_len + 1):length(nd.ranges)), _empty_slots(eltype(nd.ranges)))
    end
    ensure_capacity!(nd.counts, N)
    ensure_capacity!(nd.offsets, N)
    return nothing
end

@inline function ensure_capacity!(sb::SharedBuffers{D, M, T}, req_capacity::Int) where {D, M, T}
    ensure_capacity!(sb.pos_buffer, req_capacity)
    ensure_capacity!(sb.bit_buffer, req_capacity)
    ensure_capacity!(sb.float_buffer, req_capacity)
    ensure_capacity!(sb.float_buffer2, req_capacity)
    ensure_capacity!(sb.int_buffer, req_capacity)
    ensure_capacity!(sb.rho_buffer, req_capacity)
    return nothing
end

"""
    ensure_capacity!(pg::ParticleGrid, req_capacity::Int)

Makes room for `req_capacity` particles in every per-particle array of the grid (on the device too).
"""
@inline function ensure_capacity!(pg::ParticleGrid, req_capacity::Int)
    ensure_capacity!(pg.rhos, req_capacity)
    ensure_capacity!(pg.curvatures, req_capacity)
    ensure_capacity!(pg.core, req_capacity)
    ensure_capacity!(pg.shared, req_capacity)
    ensure_capacity!(pg.reorder, req_capacity)
    ensure_particle_capacity!(pg.neighbor, req_capacity)
    ensure_capacity!(pg.bins, req_capacity)
    
    return nothing
end