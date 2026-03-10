module ParticleGrids

export ParticleGrid, ParticleGrid1D, ParticleGrid2D, getPeriodicDistance, saveGrid, plotDensity, 
       animateDensity, getTimeStep, findLocalExtrema, updateVoxelInformation!, gridToLinearIndex, linearIndexToGrid, 
       findneighboringVoxels, updateNeighbors!, getEuclideanDistance, logMOODEvents!, findLocalExtremaAbs, sort_1d_particles!,
       determineVolumes!, getDistance, apply_boundary_conditions!, set_df!, getNBSlice, reorder_particles_for_locality!,
       manage_particles!, sort_particles!

using Random
#using Statistics
using LinearAlgebra
using CellListMap
using StaticArrays
using Base.Threads # For Atomic operations
using ProgressMeter
using ..SimSettings
using ..HyperbolicPDEs
using ..MLSWeightFunctions

export get_positions, get_weights, get_xdistance, get_ydistance
export ParticleGrid, GridMetadata, SharedBuffers, NeighborData, ReorderData, ManagementData, ParticleGridCore, createParticleGrid
# ---------------------------------------------------------
# 1. Grid Metadata
# ---------------------------------------------------------
mutable struct GridMetadata{D}
    N::Int                  # Current number of active particles
    N_interior::Int         # Number of interior (non-boundary) particles
    N_ghost::Int            # Number of ghost/boundary particles
    
    # Domain Boundaries
    mins::SVector{D, Float64}
    maxs::SVector{D, Float64}
    
    # Resolution / Topology
    h::Float64              # Smoothing length
    dx::SVector{D, Float64} # Base spacing (for regular grids)
    regular::Bool
    bc::Symbol              # :periodic, :fixed_dirichlet, etc.
    range_factor::Float64
    max_nb::Int             # Estimated max neighbors per particle
end

# ---------------------------------------------------------
# 2. Shared Workspace Buffers
# ---------------------------------------------------------
"""
Shared buffers to prevent allocating new arrays during sorting, 
particle management, or MOOD limiting.
"""
mutable struct SharedBuffers{D, M}
    # N x M matrix for generic physics variable shuffling
    rho_buffer::Matrix{Float64}      
    
    # N-length buffers for positions and flags
    pos_buffer::Vector{SVector{D, Float64}} 
    bit_buffer::Vector{Bool}
    int_buffer::Vector{Int}
end

# ---------------------------------------------------------
# 3. Neighbor Search Context
# ---------------------------------------------------------
struct NeighborData{D, S, WF}
    system::S      # CellListMap system or similar
    weight_func::WF

    # CSR (Compressed Sparse Row) format for Neighbors
    pointers::Vector{Int}
    amount::Vector{Int}
    indices::Vector{Int}
    
    # Matrix holding (Weight, dx, [dy, dz]) for cache-friendly access
    # Rows: D + 1, Columns: Total number of neighbor pairs
    data::Matrix{Float64} 

    # Thread-safety atomics for building neighbor lists in parallel
    atomic_counts::Vector{Atomic{Int}}
    atomic_offsets::Vector{Atomic{Int}}
end

# ---------------------------------------------------------
# 4. Reordering / Sorting Context
# ---------------------------------------------------------
struct ReorderData{D}
    permutation::Vector{Int}          # Logical to physical index
    inv_permutation::Vector{Int}      # Physical to logical index
    new_permutation_buffer::Vector{Int} 
    seen_buffer::Vector{Bool}            # O(1) lookup for graph traversal
    # Note: We use the SharedBuffers for the actual data shuffling!
end

# ---------------------------------------------------------
# 5. Particle Management Context
# ---------------------------------------------------------
struct ManagementData{D}
    # Placeholders for your reworked ParticleManagement.jl
    # e.g., tracking volumes, split/merge targets, etc.
    merge_flags::Vector{Bool}
    split_targets::Vector{Int}
end

# ---------------------------------------------------------
# 6. Particle Grid Core (Geometry & Topology)
# ---------------------------------------------------------
mutable struct ParticleGridCore{D}
    positions::Vector{SVector{D, Float64}}
    is_boundary::Vector{Bool}
    volumes::Vector{Float64}
end


# ---------------------------------------------------------
# 7. The Top-Level Particle Grid
# ---------------------------------------------------------
mutable struct ParticleGrid{D, M, S, WF}
    # Sub-structs
    meta::GridMetadata{D}
    core::ParticleGridCore{D}
    shared::SharedBuffers{D, M}
    neighbor::NeighborData{D, S, WF}
    reorder::ReorderData{D}
    manage::ManagementData{D}
    
    # Physics Data (N x M matrices)
    rhos::Matrix{Float64}
    mood_events::Matrix{Bool}
    curvatures::Matrix{Float64}
end

# 1D Intercept
@inline get_positions(pg::ParticleGrid{1}) = reinterpret(Float64, pg.core.positions)
# 2D Normal Access
@inline get_positions(pg::ParticleGrid{2}) = pg.core.positions

# Column-Major Views for Neighbors
@inline get_weights(pg::ParticleGrid)   = view(pg.neighbor.data, :, 1)
@inline get_xdistance(pg::ParticleGrid) = view(pg.neighbor.data, :, 2)
@inline get_ydistance(pg::ParticleGrid) = view(pg.neighbor.data, :, 3)

# --- Aliases for convenience ---
const ParticleGrid1D{M, S, WF} = ParticleGrid{1, M, S, WF}
const ParticleGrid2D{M, S, WF} = ParticleGrid{2, M, S, WF}

function createParticleGrid(
    ::Val{1}, xmin::Real, xmax::Real, N_interior::Integer, bc::Symbol,
    interp_range_factor::Real;
    M::Int = 1, # Number of system variables
    randomness::Real = 0.0, rng = Random.default_rng(), merge_factor = 0.2,
    weight_func = exponentialWeightFunction(1.,1.)
)
    # --- 1. Base Calculations ---
    N_ghost::Int = bc == :periodic ? 0 : ceil(Int, interp_range_factor)
    if bc == :periodic
        @assert N_ghost == 0 "Periodic grids do not use ghost cells."
        N = N_interior
        dx = (xmax - xmin) / max(N_interior, 1.0)
    else
        @assert N_ghost >= 0 "N_ghost must be non-negative."
        N = N_interior + 2 * N_ghost
        dx = (xmax - xmin) / max(N_interior - 1, 1.0)
    end

    xmin_tot = xmin - N_ghost * dx
    xmax_tot = xmax + N_ghost * dx
    h = dx * interp_range_factor
    regular = (randomness == 0.0)

    # --- 2. Initialize Positions and Boundaries ---
    positions = Vector{SVector{1, Float64}}(undef, N)
    is_boundary = zeros(Bool,N)

    if bc == :periodic
        for i in 1:N_interior
            positions[i] = SVector(xmin + dx*(i-0.5) + randomness*(rand(rng, Float64)*2 - 1))
        end
    else
        for i in 1:N_ghost
            positions[i] = SVector(xmin - (N_ghost - i + 1) * dx)
            is_boundary[i] = true
        end
        for i in 1:N_interior
            base_pos = (N_interior == 1) ? (xmin+xmax)/2.0 : xmin + (i-1) * dx
            positions[N_ghost + i] = SVector(base_pos + randomness*(rand(rng, Float64)*2 - 1))
        end
        for i in 1:N_ghost
            positions[N_ghost + N_interior + i] = SVector(xmax + i * dx)
            is_boundary[N_ghost + N_interior + i] = true
        end
    end

    # --- 3. Construct Sub-Structs ---
    meta = GridMetadata{1}(
        N, N_interior, N_ghost, 
        SVector(xmin_tot), SVector(xmax_tot), 
        h, SVector(dx), regular, bc, Float64(interp_range_factor), 0
    )

    # Core uses a 2x0 matrix initially: Row 1 = Weight, Row 2 = dx
    core = ParticleGridCore{1}(
        positions, is_boundary, zeros(Int, N),
    )

    shared = SharedBuffers{1, M}(
        zeros(N, M), similar(positions), zeros(Bool,N), zeros(Int, N)
    )

    neighbor = NeighborData{1, Nothing, typeof(weight_func)}(
        nothing, weight_func, zeros(Int, N + 1), zeros(Int, N), Int[], 
        Matrix{Float64}(undef, 0, 2), 
        [Atomic{Int}(0) for _ in 1:N], [Atomic{Int}(0) for _ in 1:N]
    )

    permutation = collect(1:N)
    reorder = ReorderData{1}(
        permutation, copy(permutation), zeros(Int, N), zeros(Bool,N)
    )

    manage = ManagementData{1}(zeros(Bool,N), Int[])

    # --- 4. Assemble Final Grid ---
    pg = ParticleGrid{1, M, Nothing, typeof(weight_func)}(
        meta, core, shared, neighbor, reorder, manage,
        zeros(N, M), zeros(Bool,N,M), zeros(N, M) # rhos, mood_events, curvatures
    )

    pg.reorder(pg)
    pg.neighbor(pg)
    
    return pg
end

function createParticleGrid(
    ::Val{2}, xmin::Real, xmax::Real, ymin::Real, ymax::Real, 
    Nx_interior::Int, Ny_interior::Int, bc::Symbol, interp_range_factor::Real;
    M::Int = 1,
    randomness::NTuple{2, Float64} = (0.0, 0.0), rng = Random.default_rng(), 
    weight_func = exponentialWeightFunction(1.,1.)
)
# 1. Force strict types to avoid Real propagation (The Fix)
    xmin_f, xmax_f = Float64(xmin), Float64(xmax)
    ymin_f, ymax_f = Float64(ymin), Float64(ymax)
    range_factor_f = Float64(interp_range_factor)
    
    # Strictly typecast the randomness variables!
    rand_x = Float64(randomness[1])
    rand_y = Float64(randomness[2])

    # --- 1. Base Calculations ---
    N_ghost::Int = bc == :periodic ? 0 : ceil(Int, range_factor_f)
    
    if bc == :periodic
        @assert N_ghost == 0 "Periodic grids do not use ghost cells."
        Nx_total, Ny_total = Nx_interior, Ny_interior
        dx_nominal = (xmax_f - xmin_f) / Nx_interior
        dy_nominal = (ymax_f - ymin_f) / Ny_interior
    else
        @assert N_ghost >= 0 "N_ghost must be non-negative."
        Nx_total = Nx_interior + 2*N_ghost
        Ny_total = Ny_interior + 2*N_ghost
        dx_nominal = (xmax_f - xmin_f) / max(Nx_interior - 1, 1.0)
        dy_nominal = (ymax_f - ymin_f) / max(Ny_interior - 1, 1.0)
    end
    
    N = Nx_total * Ny_total
    interp_range = range_factor_f < 1e-10 ? max(dx_nominal, dy_nominal) : range_factor_f * max(dx_nominal, dy_nominal)

    # --- 2. Initialize Positions and Boundaries ---
    positions = Vector{SVector{2, Float64}}(undef, N)
    is_boundary = zeros(Bool,N)
    
    # =========================================================================
    # THE FUNCTION BARRIER
    # This completely isolates the `system` type, preventing Union explosions.
    # =========================================================================
    function _build_grid(sys)
        meta = GridMetadata{2}(
            N, Nx_interior * Ny_interior, N - (Nx_interior * Ny_interior), 
            SVector{2, Float64}(xmin_f, ymin_f), SVector{2, Float64}(xmax_f, ymax_f), 
            interp_range, SVector{2, Float64}(dx_nominal, dy_nominal), 
            (randomness == (0.0, 0.0)), bc, range_factor_f, 0
        )

        core = ParticleGridCore{2}(
            positions, is_boundary, zeros(Float64, N) # Volumes array
        )

        shared = SharedBuffers{2, M}(
            zeros(N, M), similar(positions), zeros(Bool,N), zeros(Int, N)
        )

        neighbors = NeighborData{2, typeof(sys), typeof(weight_func)}(
            sys, weight_func, zeros(Int, N + 1), zeros(Int, N), Int[], 
            Matrix{Float64}(undef, 0, 3),
            [Atomic{Int}(0) for _ in 1:N], [Atomic{Int}(0) for _ in 1:N]
        )

        reorder = ReorderData{2}(
            collect(1:N), collect(1:N), zeros(Int, N), zeros(Bool,N)
        )

        manage = ManagementData{2}(zeros(Bool,N), Int[])

        return ParticleGrid{2, M, typeof(sys), typeof(weight_func)}(
            meta, core, shared, neighbors, reorder, manage,
            zeros(N, M), zeros(Bool,N,M), zeros(N, M)
        )
    end

    # --- 3. Build Positions and System ---
    if bc == :periodic
        for i in 1:Nx_total, j in 1:Ny_total
            index = (i - 1) * Ny_total + j
            posX = xmin_f + dx_nominal*(i-0.5) + rand_x*(rand(rng, Float64)*2 - 1)
            posY = ymin_f + dy_nominal*(j-0.5) + rand_y*(rand(rng, Float64)*2 - 1)
            positions[index] = SVector{2, Float64}(posX, posY)
        end
        # CRITICAL FIX: SVector instead of Vector
        unit_cell = SVector{2, Float64}(xmax_f - xmin_f, ymax_f - ymin_f)
        system = InPlaceNeighborList(x=positions, cutoff=interp_range, unitcell=unit_cell, parallel=true)
        
        return _build_grid(system)
    else
        for i in 1:Nx_total, j in 1:Ny_total
            index = (i - 1) * Ny_total + j
            is_interior = (N_ghost < i <= Nx_interior + N_ghost) && (N_ghost < j <= Ny_interior + N_ghost)
            
            posX = i <= N_ghost ? xmin_f - (N_ghost-i+1)*dx_nominal : (i > Nx_interior+N_ghost ? xmax_f+(i-(Nx_interior+N_ghost))*dx_nominal : xmin_f+(i-N_ghost-1)*dx_nominal + rand_x*(rand(rng,Float64)*2-1))
            posY = j <= N_ghost ? ymin_f - (N_ghost-j+1)*dy_nominal : (j > Ny_interior+N_ghost ? ymax_f+(j-(Ny_interior+N_ghost))*dy_nominal : ymin_f+(j-N_ghost-1)*dy_nominal + rand_y*(rand(rng,Float64)*2-1))
            
            positions[index] = SVector{2, Float64}(posX, posY)
            is_boundary[index] = !is_interior
        end
        
        system = InPlaceNeighborList(x=positions, cutoff=interp_range, parallel=true)
        return _build_grid(system)
    end
end

"""
    LocalVoxels

Helper struct to manage the relative voxel map for gap detection.
- `num_bins`: Total number of voxels (odd number to center one on the particle).
- `half_bins`: Number of bins on one side (e.g., if num_bins=5, half_bins=2).
- `voxel_size`: Spatial length of one voxel.
- `occupation`: Re-usable boolean buffer to mark occupied voxels.
"""
mutable struct LocalVoxels
    num_bins::Int
    half_bins::Int
    voxel_size::Float64
    occupation::Vector{Bool}

    function LocalVoxels(min_nb::Int, R::Float64)
        # Formula: 2 * k + 1 ensures symmetry around 0.
        # k = ceil(Int, min_nb) usually ensures we have 'min_nb' slots per side.
        #k = max(ceil(Int, min_nb), 2) # Ensure at least 2 neighbors per side support
        
        num_bins = 2 * min_nb + 1
        
        # Total coverage is [-R, R], length 2*R
        # voxel_size = (2 * R) / num_bins
        voxel_size = (2.0 * R) / num_bins
        
        occupation = zeros(Bool, num_bins)
        
        new(num_bins, min_nb, voxel_size, occupation)
    end
end



@inline function getNBSlice(pg::ParticleGrid, p_idx::Int)
    num_nb = pg.neighbor.amount[p_idx]
    pointer = pg.neighbor.pointers[p_idx]
    neighbor_slice = pointer:(pointer + num_nb - 1)
    return neighbor_slice
end

sort_particles!(pg::ParticleGrid) = pg.reorder(pg) 

function (rd::ReorderData{D})(pg::ParticleGrid{D, M, S, WF}) where {D, M, S, WF}
    N = pg.meta.N
    
    # --- 1. Calculate RCM Permutation ---
    # We use the grid's seen_buffer as the visited set to avoid allocations
    visited = rd.seen_buffer
    fill!(visited, false)
    perm_idx = 0
    
    queue = Int[]
    neighbor_buffer = Int[]

    for i in 1:N 
        if !visited[i]
            # Start BFS from an unvisited component
            start_node = i
            visited[start_node] = true
            
            resize!(queue, 0)
            push!(queue, start_node)

            while !isempty(queue)
                current_node = popfirst!(queue)
                
                # Add current node to the permutation
                perm_idx += 1
                rd.permutation[perm_idx] = current_node

                # Get unvisited neighbors
                resize!(neighbor_buffer, 0)
                num_nb = pg.neighbor.amount[current_node]
                
                if num_nb > 0
                    start_ptr = pg.neighbor.pointers[current_node]
                    neighbor_slice = start_ptr:(start_ptr + num_nb - 1)
                    
                    @inbounds for k in neighbor_slice
                        nb_idx = pg.neighbor.indices[k]
                        if !visited[nb_idx]
                            visited[nb_idx] = true # Mark visited when adding
                            push!(neighbor_buffer, nb_idx)
                        end
                    end
                end
                
                # Sort neighbors by their degree (low to high) for Cuthill-McKee
                sort!(neighbor_buffer, by = idx -> pg.neighbor.amount[idx])
                append!(queue, neighbor_buffer)
            end
        end
    end
    
    # Reverse the permutation in-place for RCM
    reverse!(rd.permutation)

    # --- 2. Physically Reorder Data ---
    # Step 2a: Copy old data to our pre-allocated shared buffers
    copyto!(pg.shared.pos_buffer, pg.core.positions)
    copyto!(pg.shared.bit_buffer, pg.core.is_boundary)
    copyto!(pg.shared.rho_buffer, pg.rhos)

    # Step 2b: Write reordered data back
    Threads.@threads for i in 1:N
        # Get the OLD source index from the permutation map
        src_idx = rd.permutation[i]
        
        pg.core.positions[i]   = pg.shared.pos_buffer[src_idx]
        pg.core.is_boundary[i] = pg.shared.bit_buffer[src_idx]
        
        # Handle the N x M matrices
        for m in 1:M
            pg.rhos[i, m] = pg.shared.rho_buffer[src_idx, m]
        end
    end

    # Step 2c: Update the inverse permutation map
    Threads.@threads for i in 1:N
        rd.inv_permutation[rd.permutation[i]] = i
    end
    
    return nothing
end


"""
    _build_connectivity_graph!(pg::ParticleGrid2D, system)

Populates the grid's neighbor graph (`num_neighbors`, `neighbor_pointers`, 
`neighbor_indices`) using a two-pass parallel neighbor search.
This is the minimum information needed for the RCM algorithm.
"""
function _build_connectivity_graph!(pg::ParticleGrid2D, system)
    N = pg.meta.N
    
    # # 1. Ensure atomic buffers are ready
    # if !isdefined(pg, :atomic_counts_buffer) || length(pg.atomic_counts_buffer) != N
    #     pg.atomic_counts_buffer = [Atomic{Int}(0) for _ in 1:N]
    # end
    # if !isdefined(pg, :atomic_offsets_buffer) || length(pg.atomic_offsets_buffer) != N
    #     pg.atomic_offsets_buffer = [Atomic{Int}(0) for _ in 1:N]
    # end

    # --- PASS 1: Count Neighbors (Parallel) ---
    # We must reset counts to zero.
    @threads for i in 1:N
        pg.atomic_counts_buffer[i][] = 0
    end

    map_pairwise!(
        (xi, xj, i, j, d2, null) -> begin
            atomic_add!(pg.atomic_counts_buffer[i], 1)
            atomic_add!(pg.atomic_counts_buffer[j], 1)
            null
        end,
        0, system.box, system.cl; parallel = true
    )

    # --- Serial Prefix-Sum ---
    # Copy counts to `num_neighbors` and build `neighbor_pointers`.
    total_neighbors = 0
    for i in 1:N
        num_nb = pg.atomic_counts_buffer[i][]
        pg.neighbor.amount[i] = num_nb
        pg.neighbor.pointers[i] = total_neighbors + 1
        total_neighbors += num_nb
    end

    # --- PASS 2: Fill Neighbor Indices (Parallel) ---
    # Resize neighbor_indices array if needed
    if length(pg.neighbor.indices) < total_neighbors
        resize!(pg.neighbor.indices, total_neighbors)
    end
    
    # Reset atomic offsets for the fill pass
    @threads for i in 1:N
        pg.atomic_offsets_buffer[i][] = 0
    end
    
    atomic_offsets = pg.atomic_offsets_buffer

    map_pairwise!(
        (xi, xj, i, j, d2, null) -> begin
            # Fill neighbor list for i
            offset_i = atomic_add!(atomic_offsets[i], 1)
            write_idx_i = pg.neighbor.pointers[i] + offset_i
            pg.neighbor.indices[write_idx_i] = j

            # Fill neighbor list for j
            offset_j = atomic_add!(atomic_offsets[j], 1)
            write_idx_j = pg.neighbor.pointers[j] + offset_j
            pg.neighbor.indices[write_idx_j] = i
            null
        end,
        0, system.box, system.cl; parallel = true
    )
    
    return nothing
end

updateNeighbors!(pg::ParticleGrid) = pg.neighbors(pg)

function (nd::NeighborData{1, S, WF})(pg::ParticleGrid{1, M, S, WF}) where {M, S, WF}
    N = pg.meta.N
    maxDist = pg.meta.range_factor * pg.meta.dx[1]
    weightFunc = nd.weight_func
    
    # --- PASS 1: Count Neighbors (Serial) ---
    max_nb = 0
    total_neighbors = 0
    for i in 1:N
        num_nb = length(_find_neighbors_1d(pg, i, maxDist))
        pg.neighbor.amount[i] = num_nb
        pg.neighbor.pointers[i] = total_neighbors + 1
        total_neighbors += num_nb
        max_nb = max(max_nb, num_nb)
    end
    pg.neighbor.pointers[N+1] = total_neighbors + 1
    pg.meta.max_nb = max_nb
    
    # --- Buffer Resizing Optimization ---
    # Only resize if we exceed current capacity. Grow by 25% to minimize allocations.
    current_capacity = length(pg.neighbor.indices)
    if total_neighbors > current_capacity
        new_capacity = ceil(Int, total_neighbors * 1.25)
        resize!(pg.neighbor.indices, new_capacity)
        # Allocate new matrix: Row 1 = weight, Row 2 = dx
        pg.core.neighbor_data = Matrix{Float64}(undef, new_capacity, 2)
    end

    # --- PASS 2: Fill Neighbor Data (Serial) ---
    offset_counts = zeros(Int, N) 
    
    for i in 1:N
        neighbor_list = _find_neighbors_1d(pg, i, maxDist)
        
        for j in neighbor_list
            offset = offset_counts[i]
            write_idx = pg.neighbor.pointers[i] + offset
            
            dist_x = getDistance(pg, i, j) 
            d2 = dist_x^2

            pg.neighbor.indices[write_idx] = j
            
            # Write directly to the pre-allocated matrix
            pg.core.neighbor_data[write_idx, 1] = weightFunc(d2)
            pg.core.neighbor_data[write_idx, 2] = dist_x
            
            offset_counts[i] += 1
        end
    end
    determineVolumes!(pg) 
    return nothing
end
# --- In ParticleGrids.jl ---

# --- In ParticleGrids.jl, add this function ---

"""
    getDistance(pg::ParticleGrid1D, i::Integer, j::Integer)

Calculates the shortest distance between two 1D particles,
correctly handling periodic boundary conditions.
"""
function getDistance(pg::ParticleGrid1D, i::Integer, j::Integer)
    # 1. Calculate the simple, non-periodic distance
    dist = get_positions(pg)[j] - get_positions(pg)[i]

    # 2. Apply periodic correction if necessary
    if pg.meta.bc == :periodic
        domain_size = pg.meta.xmax[1] - pg.meta.xmin[1]
        # Correct the distance by the shortest wrap-around
        dist -= round(dist / domain_size) * domain_size
    end
    
    return dist
end

"""
Finds all neighbors for particle `i` in a 1D grid within `maxDist`.
This is a helper function for `updateNeighbors!`.
"""
function _find_neighbors_1d(pg::ParticleGrid1D, i::Int, maxDist::Float64)
    N = pg.meta.N
    positions = get_positions(pg)
    pos_i = positions[i]
    
    # Pre-allocate a reasonable number of neighbors
    neighbor_list = Vector{Int}()
    sizehint!(neighbor_list, 2 * ceil(Int, maxDist / pg.meta.dx[1]) + 2)

    if pg.meta.bc == :periodic
        # Search left, wrapping around the boundary
        for j_offset in 1:div(N, 2)
            j = mod1(i - j_offset, N)
            dist = abs(getDistance(pg, i, j))
            if dist <= maxDist
                push!(neighbor_list, j)
            else
                break # Particles are sorted
            end
        end
        # Search right, wrapping around the boundary
        for j_offset in 1:div(N, 2)
            j = mod1(i + j_offset, N)
            dist = abs(getDistance(pg, i, j))
            if dist <= maxDist
                push!(neighbor_list, j)
            else
                break
            end
        end
    else # Non-periodic
        # Search left
        for j in (i-1):-1:1
            if abs(positions[j] - pos_i) <= maxDist
                push!(neighbor_list, j)
            else
                break
            end
        end
        # Search right
        for j in (i+1):N
            if abs(positions[j] - pos_i) <= maxDist
                push!(neighbor_list, j)
            else
                break
            end
        end
    end
    return neighbor_list
end

function (nd::NeighborData{D, S, WF})(pg::ParticleGrid{D, M, S, WF}) where {D, M, S, WF}
    system = nd.system
    weightFunc = nd.weight_func    

    CellListMap.update!(system, pg.core.positions)

    # --- PASS 1: COUNT NEIGHBORS (Thread-Safe) ---
    @inbounds for i in 1:pg.meta.N; nd.atomic_counts[i][] = 0; end
    
    atomic_counts = nd.atomic_counts

    map_pairwise!(
        (xi, xj, i, j, d2, null) -> begin
            atomic_add!(atomic_counts[i], 1)
            atomic_add!(atomic_counts[j], 1)
            null
        end,
        0, system.box, system.cl; parallel = true
    )

    max_so_far = 0 
    @inbounds for i in 1:pg.meta.N
        count = atomic_counts[i][]
        pg.neighbor.amount[i] = count
        if count > max_so_far
            max_so_far = count
        end
    end
    pg.meta.max_nb = max_so_far

    # --- PREPARE FOR PASS 2 ---
    total_neighbors = sum(pg.neighbor.amount)
    
    # --- Buffer Resizing Optimization ---
    current_capacity = length(pg.neighbor.indices)
    if total_neighbors > current_capacity
        new_capacity = ceil(Int, total_neighbors * 1.25)
        resize!(pg.neighbor.indices, new_capacity)
        # Allocate new matrix: Row 1 = weight, Rows 2 to D+1 = spatial distances
        pg.core.neighbor_data = Matrix{Float64}(undef, new_capacity, D+1)
    end
    
    pg.neighbor.pointers[1] = 1
    @inbounds for i in 1:pg.meta.N
        pg.neighbor.pointers[i+1] = pg.neighbor.pointers[i] + pg.neighbor.amount[i]
    end
    
    # --- PASS 2: FILL DATA (Thread-Safe) ---
    @inbounds for i in 1:pg.meta.N; nd.atomic_offsets[i][] = 0; end
    atomic_offsets = nd.atomic_offsets
    
    map_pairwise!(
        (xi, xj, i, j, d2, null) -> begin
            # SVector math automatically handles D dimensions
            dist = xj - xi 
            
            if pg.meta.bc == :periodic
                domainSize = pg.meta.maxs - pg.meta.mins
                # Element-wise periodic wrapping for D dimensions
                dist -= round.(dist ./ domainSize) .* domainSize
            end
            
            weight = weightFunc(d2)

            # i -> j
            offset_i = atomic_add!(atomic_offsets[i], 1)
            write_idx_i = pg.neighbor.pointers[i] + offset_i
            pg.neighbor.indices[write_idx_i] = j
            pg.core.neighbor_data[write_idx_i, 1] = weight
            for d in 1:D
                pg.core.neighbor_data[write_idx_i, 1 + d] = dist[d]
            end

            # j -> i (Symmetric)
            offset_j = atomic_add!(atomic_offsets[j], 1) 
            write_idx_j = pg.neighbor.pointers[j] + offset_j
            pg.neighbor.indices[write_idx_j] = i
            pg.core.neighbor_data[write_idx_j, 1] = weight
            for d in 1:D
                pg.core.neighbor_data[write_idx_j, 1 + d] = -dist[d]
            end
            
            null
        end,
        0, system.box, system.cl; parallel = true
    )
    return nothing
end
# A clean top-level call for your physics loops
reorder_particles!(pg::ParticleGrid) = pg.reorder(pg)

function (rd::ReorderData{1})(pg::ParticleGrid{1, M, S, WF}) where {M, S, WF}
    N = pg.meta.N
    
    # 1. Determine the permutation that sorts the active positions [cite: 376]
    range = (N + 1):length(pg.core.positions)
    p = [sortperm(pg.core.positions[1:N]); collect(range)]

    # Optimization: Exit early if already sorted (common in small time steps) [cite: 376]
    if issorted(p)
        return nothing
    end

    # 2. Permute core geometry fields in-place [cite: 377]
    Base.permute!(pg.core.positions, p)
    Base.permute!(pg.core.is_boundary, p)
    
    # 3. Permute physics matrices (N x M). 
    # Row-wise assignment handles the multiple species/variables elegantly.
    pg.rhos .= pg.rhos[p, :]
    pg.curvatures .= pg.curvatures[p, :]
    pg.mood_events .= pg.mood_events[p, :]

    # Note: We do not permute the CSR neighbor buffers here because 
    # the spatial swap invalidates them. updateNeighbors! must be called next.
    return nothing
end

# --- 1D Boundary Conditions ---
function apply_boundary_conditions!(pg::ParticleGrid{1}, rhos_buffer::AbstractArray)
    bc = pg.meta.bc
    is_matrix = ndims(rhos_buffer) == 2

    if bc == :periodic
        return 
        
    elseif bc == :fixed_dirichlet
        @inbounds for i in 1:pg.meta.N
            if pg.core.is_boundary[i]
                if is_matrix
                    rhos_buffer[i, :] .= pg.rhos[i, :]
                else
                    rhos_buffer[i] = pg.rhos[i] # Fallback for 1D vectors
                end
            end
        end

    elseif bc == :outflow
        # Find bounds of the interior dynamically since particles are sorted
        first_interior = findfirst(==(false), pg.core.is_boundary)
        last_interior  = findlast(==(false), pg.core.is_boundary)
        
        if isnothing(first_interior) || isnothing(last_interior)
            return
        end
        
        if is_matrix
            val_left = rhos_buffer[first_interior, :]
            val_right = rhos_buffer[last_interior, :]
            for i in 1:(first_interior-1)
                rhos_buffer[i, :] .= val_left
            end
            for i in (last_interior+1):pg.meta.N
                rhos_buffer[i, :] .= val_right
            end
        else
            val_left = rhos_buffer[first_interior]
            val_right = rhos_buffer[last_interior]
            rhos_buffer[1:(first_interior-1)] .= val_left
            rhos_buffer[(last_interior+1):end] .= val_right
        end
    end
    return nothing
end

# --- 2D Boundary Conditions ---
function apply_boundary_conditions!(pg::ParticleGrid{2}, rhos_buffer::AbstractArray)
    bc = pg.meta.bc
    is_matrix = ndims(rhos_buffer) == 2

    if bc == :periodic
        return 
        
    elseif bc == :fixed_dirichlet
        @inbounds for i in 1:pg.meta.N
            if pg.core.is_boundary[i]
                if is_matrix
                    rhos_buffer[i, :] .= pg.rhos[i, :]
                else
                    rhos_buffer[i] = pg.rhos[i] 
                end
            end
        end
        
    elseif bc == :outflow
        @inbounds for ghost_idx in 1:pg.meta.N
            if pg.core.is_boundary[ghost_idx]
                num_nb = pg.neighbor.amount[ghost_idx]
                if num_nb == 0
                    continue
                end

                start_idx = pg.neighbor.pointers[ghost_idx]
                min_dist_sq = Inf
                closest_interior_idx = -1

                # Search through neighbors using the cache-friendly matrix
                @inbounds for k in start_idx:(start_idx + num_nb - 1)
                    neighbor_idx = pg.neighbor.indices[k]
                    
                    if !pg.core.is_boundary[neighbor_idx]
                        dx = pg.core.neighbor_data[k, 2]
                        dy = pg.core.neighbor_data[k, 3]
                        dist_sq = dx^2 + dy^2

                        if dist_sq < min_dist_sq
                            min_dist_sq = dist_sq
                            closest_interior_idx = neighbor_idx
                        end
                    end
                end
                
                # Copy values from the closest interior neighbor found
                if closest_interior_idx != -1
                    if is_matrix
                        rhos_buffer[ghost_idx, :] .= rhos_buffer[closest_interior_idx, :]
                    else
                        rhos_buffer[ghost_idx] = rhos_buffer[closest_interior_idx]
                    end
                end
            end
        end
    end
    return nothing
end

"""
Calculates the 1D 'volume' (length of the Voronoi cell) for each particle.
"""
function determineVolumes!(pg::ParticleGrid1D)
    N = pg.meta.N
    if N == 0; return; end

    positions = get_positions(pg)
    volumes = pg.volumes
    
    if pg.meta.bc == :periodic
        for i in 1:N
            prev_idx = mod1(i - 1, N)
            next_idx = mod1(i + 1, N)
            # Use getDistance to correctly handle wrapping for edge particles
            deltaPosL = abs(getDistance(pg, i, prev_idx))
            deltaPosR = abs(getDistance(pg, i, next_idx))
            volumes[i] = (deltaPosL + deltaPosR) / 2.0
        end
    else
        # For non-periodic, only calculate for interior points
        for i in 1:N
            if pg.core.is_boundary[i]; continue end
            volumes[i] = (positions[i+1] - positions[i-1]) / 2.0
        end
    end
    return
end

"""
Finds the local min/max of `rho` in the neighborhood using direct indexing.
"""
function findLocalExtrema(
    rho_i::Float64,
    nb_slice::UnitRange{Int},          # Slice for the current particle
    neighbor_fs::AbstractVector{Float64}, # The grid's full neighbor index list
)::Tuple{Float64, Float64}
    
    minU = rho_i
    maxU = rho_i
    
    # Iterate through the slice of the full neighbor index list
    @inbounds for k in nb_slice 
        rho_j = neighbor_fs[k]
        minU = min(minU, rho_j)
        maxU = max(maxU, rho_j)
    end
    
    return (minU, maxU)
end

"""
Finds the local min/max and absolute min/max of curvature (1D) using direct indexing.
"""
function findLocalExtremaAbs(
    curve_i::Float64,
    nb_slice::UnitRange{Int},          # Slice for the current particle
    neighbor_indices_full::Vector{Int}, # The grid's full neighbor index list
    curveVec::AbstractVector{Float64}   # Full curvature vector from workspace
)::Tuple{Float64, Float64, Float64, Float64}
    
    mini = maxi = curve_i
    minAbs = maxAbs = abs(curve_i)
    
    @inbounds for k in nb_slice
        j = neighbor_indices_full[k] # Get neighbor index
        curve_j = curveVec[j]
        abs_curve_j = abs(curve_j)

        mini = min(mini, curve_j)
        maxi = max(maxi, curve_j)
        minAbs = min(minAbs, abs_curve_j)
        maxAbs = max(maxAbs, abs_curve_j)
    end
    
    return (mini, maxi, minAbs, maxAbs)
end


"""
Finds the local min/max and absolute min/max of curvatures (xx, yy) (2D) using direct indexing.
"""
function findLocalExtremaAbs(
    curve_xx_i::Float64,
    curve_yy_i::Float64,
    nb_slice::UnitRange{Int},          # Slice for the current particle
    neighbor_indices_full::Vector{Int}, # The grid's full neighbor index list
    curve_xx_Vec::AbstractVector{Float64}, # Full xx curvature vector from workspace
    curve_yy_Vec::AbstractVector{Float64}  # Full yy curvature vector from workspace
)::NTuple{8, Float64}

    # Initialize with values at the central particle i
    mini1 = maxi1 = curve_xx_i
    mini2 = maxi2 = curve_yy_i
    minAbs1 = maxAbs1 = abs(curve_xx_i)
    minAbs2 = maxAbs2 = abs(curve_yy_i)
    
    @inbounds for k in nb_slice
        j = neighbor_indices_full[k] # Get neighbor index
        
        # Fetch neighbor curvatures
        curve_xx_j = curve_xx_Vec[j]
        curve_yy_j = curve_yy_Vec[j]
        abs_curve_xx_j = abs(curve_xx_j)
        abs_curve_yy_j = abs(curve_yy_j)

        # Update extrema for the xx component
        mini1 = min(mini1, curve_xx_j)
        maxi1 = max(maxi1, curve_xx_j)
        minAbs1 = min(minAbs1, abs_curve_xx_j)
        maxAbs1 = max(maxAbs1, abs_curve_xx_j)
        
        # Update extrema for the yy component
        mini2 = min(mini2, curve_yy_j)
        maxi2 = max(maxi2, curve_yy_j)
        minAbs2 = min(minAbs2, abs_curve_yy_j)
        maxAbs2 = max(maxAbs2, abs_curve_yy_j)
    end
    
    return (mini1, maxi1, minAbs1, maxAbs1, mini2, maxi2, minAbs2, maxAbs2)
end

# --- 1D Time Step ---
function getTimeStep(pg::ParticleGrid{1}, eq)#::LinearAdvection{1})
    dtMax = Inf
    vel = velocity(eq, 0.0)

    for i in 1:pg.meta.N
        # Skip boundary particles
        if pg.core.is_boundary[i]
            continue
        end
        
        num = 0.0
        denum = 0.0
        
        start_idx = pg.neighbor.pointers[i]
        num_nb = pg.neighbor.amount[i]
        w_vec = get_weights(pg)
        dx_vec = get_xdistance(pg)
        
        @inbounds for k in start_idx:(start_idx + num_nb - 1)
            # Row 1 is Weight, Row 2 is dx
            w  = w_vec[k]
            dx = dx_vec[k]
            
            # Upwind condition
            if ((vel >= 0.0) && (dx <= 0.0)) || ((vel <= 0.0) && (dx >= 0.0))
                num += w * dx
                denum += w * dx * dx
            end
        end

        if abs(vel * num) > 1e-14
            dtMax = min(-denum / (vel * num), dtMax)
        end
    end
    return dtMax
end

# --- 2D Time Step ---
function getTimeStep(pg::ParticleGrid{2}, eq)#::LinearAdvection{2})
    dtMax = Inf
    vel = eq.vel

    for i in 1:pg.meta.N
        if pg.core.is_boundary[i]
            continue
        end
        
        num_nb = pg.neighbor.amount[i]
        if num_nb == 0
            continue
        end
        
        start_idx = pg.neighbor.pointers[i]
        w_vec = get_weights(pg)
        dx_vec = get_xdistance(pg)
        dy_vec = get_ydistance(pg)
        # --- First Pass: Least Squares Matrix ---
        A11 = 0.0; A12 = 0.0; A22 = 0.0
        @inbounds for k in start_idx:(start_idx + num_nb - 1)
            # Row 1: w, Row 2: dx, Row 3: dy
            w  = w_vec[k]
            dx = dx_vec[k]
            dy = dy_vec[k]

            A11 += w * dx * dx
            A12 += w * dx * dy
            A22 += w * dy * dy
        end

        D = A11 * A22 - (A12^2)
        if abs(D) < 1e-14
            continue
        end

        # --- Second Pass: Calculate sumCij ---
        sumCij = 0.0
        @inbounds for k in start_idx:(start_idx + num_nb - 1)
            w  = w_vec[k]
            dx = dx_vec[k]
            dy = dy_vec[k]
            
            coeff_x = (A22 * w * dx - A12 * w * dy) / D
            coeff_y = (A11 * w * dy - A12 * w * dx) / D
            
            angle = atan(dy, dx)
            n_x, n_y = cos(angle), sin(angle)
            s_x, s_y = -n_y, n_x
            
            alfaBar = n_x * coeff_x + n_y * coeff_y
            betaBar = s_x * coeff_x + s_y * coeff_y
            
            dot_vel_n = vel[1] * n_x + vel[2] * n_y
            dot_vel_s = vel[1] * s_x + vel[2] * s_y
            
            bracketMinus = dot_vel_n > 0.0 ? 0.0 : dot_vel_n
            bracketMinus2 = betaBar * dot_vel_s > 0.0 ? 0.0 : betaBar * dot_vel_s
            
            sumCij -= alfaBar * bracketMinus + bracketMinus2
        end
        
        if abs(sumCij) > 1e-14
            dtMax = min(1 / (2 * sumCij), dtMax)
        end
    end
    return dtMax
end

include("ParticleManagement.jl")

end  # module ParticleGrids