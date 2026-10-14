export ParticleGrid, get_time_step

include("Domains.jl")

# ---------------------------------------------------------
# 1. Grid Metadata
# ---------------------------------------------------------
"""
    GridMetadata{D, T}
    SharedBuffers{D, M, T}
    NeighborData{D, T, WF}
    ReorderData{D}
    GlobalBins{D, T, N_OFF}
    ParticleGridCore{D, T}

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
end

# ---------------------------------------------------------
# 2. Shared Workspace Buffers
# ---------------------------------------------------------
mutable struct SharedBuffers{D, M, T}
    rho_buffer::Vector{State{M, T}}      
    pos_buffer::Vector{Space{D, T}}
    float_buffer::Vector{T}
    float_buffer2::Vector{T}
    bit_buffer::Vector{Bool}
    int_buffer::Vector{Int}
end

# ---------------------------------------------------------
# 3. Neighbor Search Context
# ---------------------------------------------------------
mutable struct NeighborData{D, T, WF}    
    weight_func::WF
    ranges::Vector{UnitRange{Int}}
    indices::Vector{Int}
    weights::Vector{T}
    distances::Vector{Space{D, T}} 
    counts::Vector{Int}
    offsets::Vector{Int}
end

# ---------------------------------------------------------
# 4. Reordering / Sorting Context
# ---------------------------------------------------------
struct ReorderData{D}
    permutation::Vector{Int}          
    inv_permutation::Vector{Int}      
    new_permutation_buffer::Vector{Int} 
    seen_buffer::Vector{Bool}            
end

struct GlobalBins{D, T, N_OFF}
    mins::Space{D, T}
    maxs::Space{D, T}
    coarse_size::Space{D, T}
    coarse_dims::NTuple{D, Int}
    head::Vector{Int}
    next::Vector{Int}
    bin_neighbors::Vector{SVector{N_OFF, Int}} 
end

# ---------------------------------------------------------
# 6. Particle Grid Core (Geometry & Topology)
# ---------------------------------------------------------
mutable struct ParticleGridCore{D, T}
    positions::Vector{Space{D, T}}
    is_boundary::Vector{Bool}
    volumes::Vector{T}
    tags::Vector{Int}

    # `volumes`: partition-of-unity quadrature weights, Σ V = |Ω| (see Volumes.jl). They define the
    # discrete mass Σ V U and make particle insertion/merging exactly conservative.
    # `particle_orders`: per-particle consistency order (read by every scheme, written by MOOD).
    particle_orders::Vector{Int}
end
"""
    ParticleGrid{D, M, T, WF, GM, N_OFF, Geom}

The primary orchestrator representing the active computational domain, tying together the physical geometry, state vectors, and mesh-free topology.

# Details
- Initializes the geometric and computational domains.
- Invokes the universal narrow-band point generator to allocate `ParticleGridCore`.
- Pre-allocates and assigns all internal components including `SharedBuffers`, `NeighborData`, `ReorderData`, and `GlobalBins`.
- Automatically forces an initial spatial reordering and constructs the initial neighbor list upon instantiation if the dimension is greater than 1.
"""
struct ParticleGrid{D, M, T, WF, GM, N_OFF, Geom}
    meta::GridMetadata{D, T}
    geometry::Geom                             
    domain::ComputationalDomain{D, T}          
    core::ParticleGridCore{D, T}
    shared::SharedBuffers{D, M, T}
    neighbor::NeighborData{D, T, WF}
    reorder::ReorderData{D}
    bins::GlobalBins{D, T, N_OFF} 
    mover::GM
    
    rhos::Vector{State{M, T}}
    curvatures::Vector{State{M, T}}
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
    
    N = length(positions)
    N_interior = count(!, is_boundary)
    N_ghost = N - N_interior
    
    meta = GridMetadata{D, T}(N, N_interior, N_ghost, R, max_dx, dxs_f, interp_range_factor, 0)
    
    core = ParticleGridCore{D, T}(positions, is_boundary, volumes, tags, zeros(Int, N))
    
    shared = SharedBuffers{D, M, T}(zeros(State{M, T}, N), similar(positions), zeros(T, N), zeros(T, N), zeros(Bool, N), zeros(Int, N))
    reorder = ReorderData{D}(collect(1:N), collect(1:N), zeros(Int, N), zeros(Bool, N))
    
    bins = GlobalBins(T, comp_domain.canvas_mins, comp_domain.canvas_maxs, R, N)
    neighbors = NeighborData{D, T, typeof(weight_func)}(
        weight_func, fill(1:0, N + 1), Int[], Vector{T}(undef, 0), Vector{Space{D, T}}(undef, 0), zeros(Int, N), zeros(Int, N)
    )

    N_OFF = get_n_offsets(Val(D))
    
    pg = ParticleGrid{D, M, T, typeof(weight_func), typeof(mover), N_OFF, typeof(geom)}(
        meta, geom, comp_domain, core, shared, neighbors, reorder, bins, mover,
        zeros(State{M, T}, N), zeros(State{M, T}, N)
    )

    # The spatial reordering only needs positions and the canvas box, so the neighbour search is run
    # once, after the particles have been laid out in memory.
    pg.reorder(pg)
    pg.neighbor(pg)
    update_volumes!(pg, volume_estimator)

    return pg
end

# =========================================================================
# EXACT GEOMETRIC CFL CALCULATION FOR HIGH-ORDER MLS
# =========================================================================

@inline _get_Lambda(eq, U_i, ::Val{D}) where {D} = Space{D}(ntuple(d -> max_eigenvalue(eq, U_i, d), Val(D)))

@generated function _compute_cfl_coeffs(::Val{D}, ::Val{IO}, ::Val{B_LEN}, nb_slice, dist_vec, w_vec, invL, ::Type{T}) where {D, IO, B_LEN, T}
    quote
        N_s = zero(SMatrix{B_LEN, B_LEN, T, B_LEN * B_LEN})
        
        @inbounds for idx in 1:length(nb_slice)
            k = nb_slice[idx]
            p_s = build_basis(Val(IO), dist_vec[k] * invL)
            N_s += w_vec[k] * (p_s * p_s')
        end
        
        # 1. Scale-invariant Tikhonov Regularization
        eps_reg = T(1e-10) * tr(N_s) / B_LEN
        N_s_reg = N_s + eps_reg * I
        
        # 2. Fast Cholesky factorization to guarantee invertibility
        C = cholesky(Symmetric(N_s_reg), check=false)
        if !LinearAlgebra.issuccess(C)
            return zero(SVector{D, T})
        end
        
        # 3. Safely invert the regularized matrix
        inv_N_s = inv(C)
        sum_c_local = zero(MVector{D, T})
        
        @inbounds for idx in 1:length(nb_slice)
            k = nb_slice[idx]
            w = w_vec[k]
            p_s = build_basis(Val(IO), dist_vec[k] * invL)
            
            # Extract the effective geometric coefficient for the linear spatial derivatives
            for d in 1:D
                c_val = zero(T)
                for j in 1:B_LEN
                    c_val += inv_N_s[d, j] * p_s[j]
                end
                sum_c_local[d] += abs(c_val * w * invL)
            end
        end
        return SVector{D, T}(sum_c_local)
    end
end

# Dispatch on the polynomial DEGREE of the divergence MLS (1..4; higher degrees clamp to 4,
# which slightly over-estimates dt for such schemes).
@generated function dispatch_cfl(degree::Int, ::Val{D}, nb_slice, dist_vec, w_vec, invL, ::Type{T}) where {D, T}
    MAX_DEG = 4
    B = typeof(basis_length(Val(D), Val(MAX_DEG))).parameters[1]
    expr = :(_compute_cfl_coeffs(Val($D), Val($MAX_DEG), Val($B), nb_slice, dist_vec, w_vec, invL, T))
    for deg in (MAX_DEG - 1):-1:1
        B = typeof(basis_length(Val(D), Val(deg))).parameters[1]
        expr = :(degree <= $deg ? _compute_cfl_coeffs(Val($D), Val($deg), Val($B), nb_slice, dist_vec, w_vec, invL, T) : $expr)
    end
    return expr
end

"""
    get_time_step(pg::ParticleGrid, eq::HyperbolicPDE, main_grad)

Computes the exact geometric CFL-restricted time step dynamically across the active particle grid.

# Details
- Safely returns infinite time steps for completely empty grids or completely orphaned particles.
- Evaluates the maximum eigenvalues from the equation system (`_get_Lambda`) against the localized geometric coefficients of the moving least squares (MLS) formulation.
- Employs a Cholesky factorization of the localized MLS moment matrix to safely invert and extract the effective spatial derivative scales.
"""
@inline function get_time_step(pg::ParticleGrid{D, M, T}, eq::HyperbolicPDE, main_grad) where {D, M, T}
    # 1. Graceful exit for empty grids
    if pg.meta.N == 0
        return T(Inf)
    end
    
    dt_buffer = pg.shared.float_buffer
    
    # Only fill the active view to avoid touching dead memory
    fill!(view(dt_buffer, 1:pg.meta.N), T(Inf))

    w_vec = get_weights(pg)
    dist_vec = get_distances(pg)

    valD = Val(D)
    dim = D 
    invL = one(T) / minimum(pg.meta.dx)
    
    degree = _divergence_degree(main_grad)

    @batch for i in 1:pg.meta.N
        if pg.core.is_boundary[i]
            continue
        end
        
        nb_slice = pg.neighbor.ranges[i]
        if isempty(nb_slice)
            continue
        end

        U_i = pg.rhos[i]
        Lambda = _get_Lambda(eq, U_i, valD)

        sum_c_vec = dispatch_cfl(degree, valD, nb_slice, dist_vec, w_vec, invL, T)
        sum_c = dot(Lambda, sum_c_vec)

        if sum_c > T(1e-14)
            dt_buffer[i] = one(T) / (dim * sum_c)
        end
    end
    
    # 2. Provide init=T(Inf) to prevent any future reduction errors
    return minimum(view(dt_buffer, 1:pg.meta.N); init=T(Inf))
end
# =========================================================================
# OVERLOADED CAPACITY MANAGERS
# =========================================================================

"""
    ensure_capacity!(rd::ReorderData, req_capacity::Int)
"""
@inline function ensure_capacity!(rd::ReorderData{D}, req_capacity::Int) where {D}
    if length(rd.permutation) < req_capacity
        new_cap = ceil(Int, req_capacity * 1.25)
        
        resize!(rd.permutation, new_cap)
        resize!(rd.inv_permutation, new_cap)
        resize!(rd.new_permutation_buffer, new_cap)
        resize!(rd.seen_buffer, new_cap)
    end
    return nothing
end

"""
    ensure_capacity!(core::ParticleGridCore, req_capacity::Int)
"""
@inline function ensure_capacity!(core::ParticleGridCore{D, T}, req_capacity::Int) where {D, T}
    if length(core.positions) < req_capacity
        new_cap = ceil(Int, req_capacity * 1.25)
        
        resize!(core.positions, new_cap)
        resize!(core.is_boundary, new_cap)
        resize!(core.volumes, new_cap)
        resize!(core.tags, new_cap)
        resize!(core.particle_orders, new_cap)
    end
    return nothing
end

"""
    ensure_capacity!(bins::GlobalBins, req_particles::Int)
"""
@inline function ensure_capacity!(bins::GlobalBins, req_particles::Int)
    if length(bins.next) < req_particles
        new_cap = ceil(Int, req_particles * 1.25)
        resize!(bins.next, new_cap)
    end
    return nothing
end

"""
    ensure_capacity!(nd::NeighborData, req_particles::Int)
"""
@inline function ensure_capacity!(nd::NeighborData{D, T, WF}, req_neighbors::Int) where {D, T, WF}
    if length(nd.indices) < req_neighbors
        new_cap = ceil(Int, req_neighbors * 1.25)
        resize!(nd.indices, new_cap)
        resize!(nd.weights, new_cap)
        resize!(nd.distances, new_cap)
    end
    return nothing
end

"""
    ensure_particle_capacity!(nd::NeighborData, N::Int)

Grows the per-particle arrays of the neighbour structure (`ranges` needs `N + 1` entries).
"""
function ensure_particle_capacity!(nd::NeighborData, N::Int)
    if length(nd.ranges) < N + 1
        new_cap = ceil(Int, (N + 1) * 1.25)
        old_len = length(nd.ranges)
        resize!(nd.ranges, new_cap)
        for k in (old_len + 1):new_cap; nd.ranges[k] = 1:0; end
    end
    length(nd.counts)  < N && resize!(nd.counts,  ceil(Int, N * 1.25))
    length(nd.offsets) < N && resize!(nd.offsets, ceil(Int, N * 1.25))
    return nothing
end

@inline function ensure_capacity!(sb::SharedBuffers{D, M, T}, req_capacity::Int) where {D, M, T}
    if length(sb.pos_buffer) < req_capacity
        new_cap = ceil(Int, req_capacity * 1.25)
        
        resize!(sb.pos_buffer, new_cap)
        resize!(sb.bit_buffer, new_cap)
        resize!(sb.float_buffer, new_cap)
        resize!(sb.float_buffer2, new_cap)
        resize!(sb.int_buffer, new_cap)
        resize!(sb.rho_buffer, new_cap)
    end
    return nothing
end

@inline function ensure_capacity!(pg::ParticleGrid, req_capacity::Int)
    if length(pg.rhos) < req_capacity
        new_cap = ceil(Int, req_capacity * 1.25)
        
        resize!(pg.rhos, new_cap)
        resize!(pg.curvatures, new_cap)
    end
    
    ensure_capacity!(pg.core, req_capacity)
    ensure_capacity!(pg.shared, req_capacity)
    ensure_capacity!(pg.reorder, req_capacity)
    ensure_particle_capacity!(pg.neighbor, req_capacity)
    ensure_capacity!(pg.bins, req_capacity)
    
    return nothing
end