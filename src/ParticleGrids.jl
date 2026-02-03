module ParticleGrids

export ParticleGrid, ParticleGrid1D, ParticleGrid2D, getPeriodicDistance, saveGrid, plotDensity, 
       animateDensity, getTimeStep, findLocalExtrema, updateVoxelInformation!, gridToLinearIndex, linearIndexToGrid, 
       findneighboringVoxels, updateNeighbors!, getEuclideanDistance, logMOODEvents!, findLocalExtremaAbs, sort_1d_particles!,
       determineVolumes!, getDistance, apply_boundary_conditions!, ParticleGridSystem, set_df!, getNBSlice, reorder_particles_for_locality!,
       manage_particles!

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
import Meshfree4ScalarEq

# # --- Export new types and functions ---
# export ParticleGrid, ParticleGrid1D, ParticleGrid2D, setInitialConditions!, 
#        getDistance, saveGrid, plotDensity, animateDensity, getTimeStep, 
#        findLocalExtrema!, updateNeighbors!, determineVolumes!, 
#        apply_boundary_conditions!, ParticleGridSystem

# --- Core Abstract Type and System Alias ---
abstract type ParticleGrid{D} end # Now parameterized by dimension
#const ParticleGridSystem{N, D} = NTuple{N, <:ParticleGrid{D}}

# In ParticleGrids.jl

"""
    ParticleGridSystem{D, N_grids}

A container for a coupled system of particle grids. 
- `grids`: The tuple of individual particle grids.
- `velocity_indices`: Indices of the species (grids) that contribute to the grid velocity.
- `grid_velocities`: A buffer storing the calculated grid velocity for every particle.
"""
struct ParticleGridSystem{N_grids, D}
    grids::NTuple{N_grids, ParticleGrid{D}}
    velocity_indices::Vector{Int} 
    grid_velocities::Vector{Float64} # For 1D (use Vector{SVector{2,Float64}} for 2D)

    function ParticleGridSystem(grids::NTuple{N_grids, ParticleGrid{D}}, velocity_indices::Vector{Int}) where {N_grids,D}
        N_particles = grids[1].N # Assuming all grids have same N (coupled)
        
        # Pre-allocate buffer for grid velocities
        grid_vels = zeros(Float64, N_particles)
        
        new{N_grids, D}(grids, velocity_indices, grid_vels)
    end
end



# Helper to support iteration/indexing like a Tuple (backward compatibility)
Base.getindex(pgs::ParticleGridSystem, i::Int) = pgs.grids[i]
Base.length(pgs::ParticleGridSystem) = length(pgs.grids)
Base.iterate(pgs::ParticleGridSystem, state=1) = iterate(pgs.grids, state)
Base.eachindex(pgs::ParticleGridSystem) = eachindex(pgs.grids)

"""
    safe_resize!(v::Vector, n::Integer)

Resizes the vector `v` to at least `n` elements. If `n` is greater than the
current length, it resizes to `ceil(n * 1.25)` to provide buffer capacity
and avoid frequent allocations.
"""
function safe_resize!(v::AbstractVector, n::Integer)
    if n > length(v)
        resize!(v, ceil(Int, n * 1.25))
    end
    return nothing
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



#==============================================================================
  1D PARTICLE GRID (Struct of Arrays Implementation)
==============================================================================#

# --- In ParticleGrids.jl ---

mutable struct ParticleGrid1D{WF} <: ParticleGrid{1}
    # --- Persistent State (SoA) ---
    positions::Vector{Float64}
    rhos::Vector{Float64}
    curvatures::Vector{Float64}
    is_boundary::BitVector
    volumes::Vector{Float64}
    mood_events::BitVector

    # Particle Sized Buffers
    neighbor_pointers::Vector{Int}
    num_neighbors::Vector{Int}

    # --- NEW: Flattened Neighbor Data Buffers (like 2D) ---
    neighbor_indices::Vector{Int}
    neighbor_xdistance::Vector{Float64}
    neighbor_weights::Vector{Float64}

# --- AMR Buffers (Pre-allocated) ---
    merged_buffer::BitVector          # To mark particles absorbed during merge
    split_buffer_pos::Vector{Float64} # Temporary storage for new particles
    split_buffer_rho::Vector{Float64}

    # --- Grid Properties ---
    weight_func::WF     
    xmin::Float64
    xmax::Float64
    inner_xmin::Float64
    inner_xmax::Float64
    N::Int # Total number of particles
    N_ghost::Int
    dx::Float64
    regular::Bool
    bc::Symbol
    interior_indices::UnitRange{Int}
    range_factor::Float64 
    max_dist::Float64
    min_dist::Float64
    max_nb::Int  
    local_voxels::LocalVoxels       

    function ParticleGrid1D(
        xmin::Real, xmax::Real, N_interior::Integer, bc::Symbol,
        interp_range_factor::Real; 
        randomness::Real = 0.0, rng = nothing, merge_factor = .2,
        # --- MODIFIED: Added default value ---
        weight_func::MLSWeightFunction = exponentialWeightFunction(1.,1.)
    )
        #rng = MersenneTwister(10)
        # --- (Existing constructor logic for N, interior_indices, dx) ---
        N_ghost::Int = bc == :periodic ? 0 : ceil(Int, interp_range_factor)
        local dx
        if bc == :periodic
            @assert N_ghost == 0 "Periodic grids do not use ghost cells."
            N = N_interior
            interior_indices = 1:N
            dx = (xmax - xmin) / (N_interior > 1 ? (N_interior) : 1.0)
        else
            @assert N_ghost >= 0 "N_ghost must be non-negative."
            if N_ghost == 0; @warn "No ghost cells given for non-periodic BCs! This is only supported for Analytical Functions!" end
            N = N_interior + 2 * N_ghost
            interior_indices = N_ghost == 0 ? (2:(N-1)) : ((N_ghost + 1):(N_ghost + N_interior))
            dx = (xmax - xmin) / (N_interior > 1 ? (N_interior - 1) : 1.0)
        end

        # --- (Existing logic for positions, is_boundary) ---
        positions = Vector{Float64}(undef, N)
        is_boundary = falses(N)
        if bc == :periodic
            for i in 1:N_interior
                # --- MODIFIED: Citation removed ---
                positions[i] = xmin + dx*(i-0.5) + randomness*(rand(rng, Float64)*2 - 1) 
            end
        else
            for i in 1:N_ghost
                positions[i] = xmin - (N_ghost - i + 1) * dx
                is_boundary[i] = true
            end
            for i in 1:N_interior
                # --- MODIFIED: Citation removed ---
                base_pos = (N_interior == 1) ? (xmin+xmax)/2.0 : xmin + (i-1) * dx 
                positions[N_ghost + i] = base_pos + randomness*(rand(rng, Float64)*2 - 1)
            end
            for i in 1:N_ghost
                positions[N_ghost + N_interior + i] = xmax + i * dx
                is_boundary[N_ghost + N_interior + i] = true
            end
        end
        
        min_dist = dx * merge_factor
        max_dist = dx * interp_range_factor
        voxels = LocalVoxels(floor(Int,interp_range_factor), max_dist)
        #error("Test")
        # --- (Initialize other state fields) ---
        rhos = zeros(Float64, N)
        curvatures = zeros(Float64, N)
        volumes = zeros(Float64, N)
        mood_events = falses(N)
        regular = (randomness == 0.0)
        xmin_tot = xmin - N_ghost * dx
        xmax_tot = xmax + N_ghost * dx

        # --- NEW: Initialize flat neighbor buffers ---
        neighbor_indices = Int[]
        neighbor_pointers = zeros(Int, N + 1)
        num_neighbors = zeros(Int, N)
        neighbor_xdistance = Float64[]
        neighbor_weights = Float64[]
        # --- Create the new grid object ---
        pg = new{typeof(weight_func)}(
            positions, rhos, curvatures, is_boundary, volumes, mood_events,
            neighbor_pointers, num_neighbors, neighbor_indices, 
            neighbor_xdistance, neighbor_weights,
            falses(N), Float64[], Float64[],
            weight_func,
            xmin_tot, xmax_tot, xmin, xmax, N, N-N_interior, dx, regular, bc, 
            interior_indices,
            convert(Float64, interp_range_factor), max_dist, min_dist, 0, voxels
        )
        sort_1d_particles!(pg)
        # --- Populate neighbor buffers ---
        updateNeighbors!(pg) # Call the new function
        return pg
    end
end

#==============================================================================
  2D PARTICLE GRID (Struct of Arrays Implementation)
==============================================================================#

mutable struct ParticleGrid2D{S, WF} <: ParticleGrid{2}
    # --- Persistent State (SoA) ---
    positions::Vector{SVector{2, Float64}}
    rhos::Vector{Float64}
    is_boundary::BitVector
    #mood_events::Bitvector

    neighbor_system::S
    weight_func::WF

    # Particle Sized Buffers
    neighbor_pointers::Vector{Int}
    num_neighbors::Vector{Int}    
    # --- Flattened Neighbor Data Buffers ---
    neighbor_indices::Vector{Int}
    neighbor_xdistance::Vector{Float64}
    neighbor_ydistance::Vector{Float64}
    neighbor_weights::Vector{Float64}

    # --- NEW: Pre-allocated atomic buffers for thread-safety ---
    atomic_counts_buffer::Vector{Atomic{Int}}
    atomic_offsets_buffer::Vector{Atomic{Int}}
    atomic_seen_buffer::Vector{Atomic{Int}}

    # --- NEW: For Adaptive Reordering ---
    permutation::Vector{Int}          # Maps logical index `i` to its current physical storage index.
    inv_permutation::Vector{Int}      # Maps physical storage index `p` back to its logical index.
    new_permutation_buffer::Vector{Int} # Buffer to discover the next optimal permutation.
    reorder_buffer_rhos::Vector{Float64}      # NEW: Pre-allocated buffer for rhos
    reorder_buffer_pos::Vector{SVector{2, Float64}} # NEW: Pre-allocated buffer for positions
    reorder_buffer_boundary::BitVector
    seen_buffer::BitVector # Simple boolean buffer for serial execution
    

    # --- Grid Metadata ---
    xmin::Float64; xmax::Float64; ymin::Float64; ymax::Float64
    N::Int; N_ghost::Int; dx::Float64; dy::Float64
    regular::Bool; bc::Symbol; range_factor::Float64; max_nb::Int

    function ParticleGrid2D(
        xmin::Real, xmax::Real, ymin::Real, ymax::Real, 
        Nx_interior::Integer, Ny_interior::Integer, bc::Symbol, interp_range_factor::Real; 
        randomness::NTuple{2, Real} = (0.0, 0.0), rng = Meshfree4ScalarEq.rng, weight_func::MLSWeightFunction = exponentialWeightFunction(1.,1.)
    )
        N_ghost::Int = bc == :periodic ? 0 : ceil(Int, interp_range_factor)
        local Nx_total, Ny_total, interior_indices, dx_nominal, dy_nominal
        if bc == :periodic
            @assert N_ghost == 0 "Periodic grids do not use ghost cells."
            Nx_total, Ny_total = Nx_interior, Ny_interior
            interior_indices = collect(1:(Nx_interior * Ny_interior))
            dx_nominal = (xmax - xmin) / Nx_interior
            dy_nominal = (ymax - ymin) / Ny_interior
        else
            @assert N_ghost >= 0 "N_ghost must be non-negative."
            if N_ghost == 0; @warn "No ghost cells given for non-periodic BCs! This is only supported for Analytical Functions!" end
            Nx_total = Nx_interior + 2*N_ghost
            Ny_total = Ny_interior + 2*N_ghost
            interior_indices = Int[]
            dx_nominal = (xmax - xmin) / (Nx_interior > 1 ? Nx_interior - 1 : 1.0)
            dy_nominal = (ymax - ymin) / (Ny_interior > 1 ? Ny_interior - 1 : 1.0)
        end
        N = Nx_total * Ny_total

        positions = Vector{SVector{2, Float64}}(undef, N)
        is_boundary = falses(N)
        interp_range = interp_range_factor < 1e-10 ? max(dx_nominal, dy_nominal) : interp_range_factor * max(dx_nominal, dy_nominal)  
        # --- Populate Particle Positions ---
        if bc == :periodic
            # CORRECTED: Use cell-centered positions for periodic case
            for i in 1:Nx_total, j in 1:Ny_total
                index = (i - 1) * Ny_total + j
                posX = xmin + dx_nominal*(i-0.5) + randomness[1]*(rand(rng, Float64)*2 - 1)
                posY = ymin + dy_nominal*(j-0.5) + randomness[2]*(rand(rng, Float64)*2 - 1)
                positions[index] = SVector(posX, posY)
            end
            system = InPlaceNeighborList(
                x=positions, 
                cutoff=interp_range, 
                unitcell=[xmax-xmin; ymax-ymin],
                parallel=true # Enable parallelization
            )
        else # Non-periodic logic
            for i in 1:Nx_total, j in 1:Ny_total
                index = (i - 1) * Ny_total + j
                is_interior = (N_ghost < i <= Nx_interior + N_ghost) && (N_ghost < j <= Ny_interior + N_ghost)
                
                posX = if i <= N_ghost; xmin - (N_ghost-i+1)*dx_nominal; elseif i > Nx_interior+N_ghost; xmax+(i-(Nx_interior+N_ghost))*dx_nominal; else xmin+(i-N_ghost-1)*dx_nominal + randomness[1]*(rand(rng,Float64)*2-1); end
                posY = if j <= N_ghost; ymin - (N_ghost-j+1)*dy_nominal; elseif j > Ny_interior+N_ghost; ymax+(j-(Ny_interior+N_ghost))*dy_nominal; else ymin+(j-N_ghost-1)*dy_nominal + randomness[2]*(rand(rng,Float64)*2-1); end
                
                positions[index] = SVector(posX, posY)
                is_boundary[index] = !is_interior
                if is_interior; push!(interior_indices, index); end
            end
            system = InPlaceNeighborList(
                x=positions, 
                cutoff=interp_range, 
                parallel=true # Enable parallelization
            )
        end

        permutation = collect(1:N)
        # 3. Create the stateful object
        N_ghost = N - Nx_interior * Ny_interior
        pg = new{typeof(system),typeof(weight_func)}(positions, zeros(N), is_boundary, 
        system, weight_func, zeros(Int,N+1), zeros(Int,N), Int[], Float64[], Float64[], Float64[],
        [Atomic{Int}(0) for _ in 1:N], [Atomic{Int}(0) for _ in 1:N], [Atomic{Int}(0) for _ in 1:N], 
        permutation, copy(permutation), zeros(Int,N), zeros(N), similar(positions), copy(is_boundary), falses(N),
        xmin, xmax, ymin, ymax, N, N_ghost,
        dx_nominal, dy_nominal, (randomness == (0.0, 0.0)), bc, convert(Float64,interp_range_factor), 0)
        reorder_particles_for_locality!(pg)
        updateNeighbors!(pg)
        return pg
    end
end


@inline function getNBSlice(pg::ParticleGrid, p_idx::Int)
    num_nb = pg.num_neighbors[p_idx]
    pointer = pg.neighbor_pointers[p_idx]
    neighbor_slice = pointer:(pointer + num_nb - 1)
    return neighbor_slice
end

"""
    reorder_particles_for_locality!(pg::ParticleGrid2D, system)

Calculates an optimal particle permutation using the Reverse Cuthill-McKee (RCM)
algorithm and then physically reorders all particle data (`rhos`, `positions`, etc.)
to match this new order for improved cache locality.

This function should be called ONCE after the grid is first initialized.
"""
function reorder_particles_for_locality!(pg::ParticleGrid2D)
    @info "Building connectivity graph for reordering..."
    # 1. Build the minimal connectivity graph (neighbor lists)
    _build_connectivity_graph!(pg, pg.neighbor_system)

    @info "Calculating RCM permutation..."
    # 2. Calculate the optimal permutation (writes to pg.permutation)
    _calculate_rcm_permutation!(pg)

    @info "Physically reordering particle data..."
    # 3. Physically reorder all particle data using this permutation
    reorder_particles!(pg)
    
    @info "Particle reordering complete."
    return nothing
end

"""
    reorder_particles!(pg::ParticleGrid2D)

Physically reorders the persistent *particle data* (`rhos`, `positions`, 
`is_boundary`) based on the permutation stored in `pg.permutation`.

This function does NOT reorder the graph itself, as it assumes
`updateNeighbors!` will be called immediately after to rebuild
the graph from scratch.
"""
function reorder_particles!(pg::ParticleGrid2D)
    N = pg.N
    
    # 1. Copy all current (old) particle data to buffers (fast, serial)
    copyto!(pg.reorder_buffer_rhos, pg.rhos)
    copyto!(pg.reorder_buffer_pos, pg.positions)
    copyto!(pg.reorder_buffer_boundary, pg.is_boundary)
    # 2. Use the buffers to write all reordered data back in ONE parallel loop
    @threads for i in 1:N
        # i = The NEW, destination index (1..N)
        
        # --- THE FIX ---
        # Get the OLD source index from the permutation map
        src_idx = pg.permutation[i]
        # --- END FIX ---
        
        # Reorder all particle data arrays at once
        pg.rhos[i]         = pg.reorder_buffer_rhos[src_idx]
        pg.positions[i]    = pg.reorder_buffer_pos[src_idx]
        pg.is_boundary[i]  = pg.reorder_buffer_boundary[src_idx]
    end

    # 3. Update the grid's permutation maps for consistency
    # (Even though the graph is stale, these maps reflect the data)
    @threads for i in 1:N
        pg.inv_permutation[pg.permutation[i]] = i
    end
    
    return nothing
end

"""
    _calculate_rcm_permutation!(pg::ParticleGrid2D)

Performs a serial Reverse Cuthill-McKee (RCM) graph traversal,
writing the resulting permutation *directly into* `pg.permutation`.
"""
function _calculate_rcm_permutation!(pg::ParticleGrid2D)
    N = pg.N
    adj = pg.neighbor_indices
    ptr = pg.neighbor_pointers
    degrees = pg.num_neighbors
    
    # Use the grid's permutation buffer as the output
    permutation = pg.permutation
    # Use the grid's seen_buffer as the visited set
    visited = pg.seen_buffer
    fill!(visited, false)

    perm_idx = 0
    
    # Local allocations are fine here (small, and only run once)
    queue = Int[]
    neighbor_buffer = Int[] 

    for i in 1:N # Loop over all particles to find starting points
        if !visited[i]
            # Found an unvisited component. Find the best starting node
            # (lowest degree) in this component.
            
            # --- Simple start: just use i ---
            # (A full search for the min-degree node in the component
            # is more robust but more complex. This is usually fine.)
            start_node = i 
            
            # --- Start BFS ---
            visited[start_node] = true
            resize!(queue, 0)
            push!(queue, start_node)

            while !isempty(queue)
                current_node = popfirst!(queue)
                
                # Add current node to the permutation
                perm_idx += 1
                permutation[perm_idx] = current_node

                # --- Cuthill-McKee part: Get & sort unvisited neighbors ---
                resize!(neighbor_buffer, 0)
                num_nb = degrees[current_node]
                
                if num_nb > 0
                    neighbor_slice = ptr[current_node] : (ptr[current_node] + num_nb - 1)
                    @inbounds for k in neighbor_slice
                        nb_idx = adj[k]
                        if !visited[nb_idx]
                            visited[nb_idx] = true # Mark as visited *when adding*
                            push!(neighbor_buffer, nb_idx)
                        end
                    end
                end
                
                # Sort neighbors by their degree (low to high)
                sort!(neighbor_buffer, by = i -> degrees[i])
                
                # Add them to the *end* of the queue
                append!(queue, neighbor_buffer)
            end
        end
    end
    
    @assert perm_idx == N "RCM permutation did not visit all nodes."

    # --- "Reverse" part (in-place) ---
    reverse!(permutation)
    
    return nothing
end

"""
    _build_connectivity_graph!(pg::ParticleGrid2D, system)

Populates the grid's neighbor graph (`num_neighbors`, `neighbor_pointers`, 
`neighbor_indices`) using a two-pass parallel neighbor search.
This is the minimum information needed for the RCM algorithm.
"""
function _build_connectivity_graph!(pg::ParticleGrid2D, system)
    N = pg.N
    
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
        pg.num_neighbors[i] = num_nb
        pg.neighbor_pointers[i] = total_neighbors + 1
        total_neighbors += num_nb
    end

    # --- PASS 2: Fill Neighbor Indices (Parallel) ---
    # Resize neighbor_indices array if needed
    if length(pg.neighbor_indices) < total_neighbors
        resize!(pg.neighbor_indices, total_neighbors)
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
            write_idx_i = pg.neighbor_pointers[i] + offset_i
            pg.neighbor_indices[write_idx_i] = j

            # Fill neighbor list for j
            offset_j = atomic_add!(atomic_offsets[j], 1)
            write_idx_j = pg.neighbor_pointers[j] + offset_j
            pg.neighbor_indices[write_idx_j] = i
            null
        end,
        0, system.box, system.cl; parallel = true
    )
    
    return nothing
end


"""
    updateNeighborsParallel!(pg, weightFunc; reorder_threshold=0.1)

Builds neighbor lists with an adaptive particle reordering strategy.
This function is now THREAD-SAFE and allocation-free.
"""
function updateNeighbors!(
    pg::ParticleGrid2D, 
    reorder_threshold::Float64 = 0.1 
)
    system = pg.neighbor_system
    weightFunc = pg.weight_func    

    CellListMap.update!(system, pg.positions)

    # --- PASS 1: COUNT NEIGHBORS (Thread-Safe & Allocation-Free) ---
    @inbounds for i in 1:pg.N; pg.atomic_counts_buffer[i][] = 0; end
    @inbounds for i in 1:pg.N; pg.atomic_seen_buffer[i][] = 0; end
    fill!(pg.new_permutation_buffer, 0) # Explicitly reset buffer
    atomic_counts = pg.atomic_counts_buffer
    atomic_seen = pg.atomic_seen_buffer
    new_order_counter = Atomic{Int}(0)

    map_pairwise!(
        (xi, xj, i, j, d2, null) -> begin
            if atomic_cas!(atomic_seen[i], 0, 1) == 0
                pg.new_permutation_buffer[i] = atomic_add!(new_order_counter, 1) + 1
            end
            if atomic_cas!(atomic_seen[j], 0, 1) == 0
                pg.new_permutation_buffer[j] = atomic_add!(new_order_counter, 1) + 1
            end
            
            atomic_add!(atomic_counts[i], 1)
            atomic_add!(atomic_counts[j], 1)
            null
        end,
        0, system.box, system.cl; parallel = true
    )
    # 2. Single loop to copy counts AND find the maximum
    max_so_far = 0 # Or typemin(Int)
    @inbounds for i in 1:pg.N
        # Get the value from the atomic
        count = atomic_counts[i][]
        
        # Copy it to the regular array
        pg.num_neighbors[i] = count
        
        # Check if it's the new maximum
        if count > max_so_far
            max_so_far = count
        end
    end

    # 3. Store the final maximum
    pg.max_nb = max_so_far

    
    # --- CRITICAL FIX: Handle isolated or missed particles ---
    # `map_pairwise!` does not guarantee visiting every particle. This loop
    # finds any particles that were missed (where the buffer is still 0) and
    # assigns them the remaining permutation slots, ensuring a valid permutation.
    final_counter_val = new_order_counter[]
    @inbounds for i in 1:pg.N
        if pg.new_permutation_buffer[i] == 0
            final_counter_val += 1
            pg.new_permutation_buffer[i] = final_counter_val
        end
    end
    

    # --- VALIDATION: Ensure the buffer is a valid permutation ---
    @assert final_counter_val == pg.N "Permutation counter did not reach N. Something is wrong."
    @assert isperm(pg.new_permutation_buffer) "new_permutation_buffer is not a valid permutation. It may contain zeros or duplicates."

    # # --- ADAPTIVE REORDERING DECISION ---
    # mismatches = 0
    # for i in 1:pg.N
    #     if pg.permutation[i] != pg.new_permutation_buffer[i]
    #         mismatches += 1
    #     end
    # end

    # if (mismatches / pg.N > reorder_threshold)# && !reordered
    #     pg.permutation[1] = -1 
    # end

    # --- PREPARE FOR PASS 2 ---
    total_neighbors = sum(pg.num_neighbors)
    resize!.((pg.neighbor_indices, pg.neighbor_xdistance, pg.neighbor_ydistance, pg.neighbor_weights), total_neighbors)
    
    pg.neighbor_pointers[1] = 1
    @inbounds for i in 1:pg.N
        pg.neighbor_pointers[i+1] = pg.neighbor_pointers[i] + pg.num_neighbors[i]
    end
    
    # --- PASS 2: FILL DATA (Thread-Safe & Allocation-Free) ---
    @inbounds for i in 1:pg.N; pg.atomic_offsets_buffer[i][] = 0; end
    atomic_offsets = pg.atomic_offsets_buffer
    map_pairwise!(
        (xi, xj, i, j, d2, null) -> begin

            dist_x = xj[1] - xi[1]
            dist_y = xj[2] - xi[2]
            if pg.bc == :periodic
                domainSizeX = pg.xmax - pg.xmin
                domainSizeY = pg.ymax - pg.ymin
                dist_x -= round(dist_x / domainSizeX) * domainSizeX
                dist_y -= round(dist_y / domainSizeY) * domainSizeY
            end
            weight = weightFunc(d2)

            # CORRECTED: Use robust atomic pattern
            offset_i = atomic_add!(atomic_offsets[i], 1)
            write_idx_i = pg.neighbor_pointers[i] + offset_i
            #println(write_idx_i)
            pg.neighbor_indices[write_idx_i]   = j
            pg.neighbor_xdistance[write_idx_i] = dist_x
            pg.neighbor_ydistance[write_idx_i] = dist_y
            pg.neighbor_weights[write_idx_i]   = weight

            # CORRECTED: Use robust atomic pattern
            
            offset_j = atomic_add!(atomic_offsets[j], 1) # -1 included bc of old value 
            #println(pg.neighbor_pointers[j],":", atomic_offsets[j][])
            write_idx_j = pg.neighbor_pointers[j] + offset_j
            pg.neighbor_indices[write_idx_j]   = i
            pg.neighbor_xdistance[write_idx_j] = -dist_x
            pg.neighbor_ydistance[write_idx_j] = -dist_y
            pg.neighbor_weights[write_idx_j]   = weight
            @assert (!isnan(dist_x) && !isnan(dist_y)) "$dist_x,$dist_y"
            null
        end,
        0, system.box, system.cl; parallel = true
    )
    #println(pg.neighbor_xdistance)
    #check_for_nans(pg)
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
    dist = pg.positions[j] - pg.positions[i]

    # 2. Apply periodic correction if necessary
    if pg.bc == :periodic
        domain_size = pg.xmax - pg.xmin
        # Correct the distance by the shortest wrap-around
        dist -= round(dist / domain_size) * domain_size
    end
    
    return dist
end

# """
#     updateNeighbors!(pg::ParticleGrid1D)

# Robust, pairwise (N^2) neighbor search. Does NOT require particles to be sorted.
# Also calculates volumes using the discovered neighbors.
# """
# function updateNeighbors!(pg::ParticleGrid1D)
#     N = pg.N
#     maxDist = pg.range_factor * pg.dx
#     weightFunc = pg.weight_func
    
#     # --- PASS 1: Count Neighbors ---
#     # We iterate N^2 to count valid neighbors.
#     total_neighbors = 0
#     max_nb = 0
    
#     for i in 1:N
#         count = 0
#         for j in 1:N
#             if i == j; continue; end # Skip self
            
#             # getDistance handles periodic wrapping automatically
#             dist = getDistance(pg, i, j)
            
#             if abs(dist) <= maxDist
#                 count += 1
#             end
#         end
        
#         pg.num_neighbors[i] = count
#         pg.neighbor_pointers[i] = total_neighbors + 1
#         total_neighbors += count
        
#         if count > max_nb
#             max_nb = count
#         end
#     end
#     # Set the end pointer (essential for iteration logic)
#     pg.neighbor_pointers[N+1] = total_neighbors + 1
#     pg.max_nb = max_nb
    
#     # --- Resize Buffers ---
#     if length(pg.neighbor_indices) < total_neighbors
#         # Grow with 25% buffer
#         new_cap = total_neighbors + (total_neighbors ÷ 4)
#         resize!(pg.neighbor_indices, new_cap)
#         resize!(pg.neighbor_xdistance, new_cap)
#         resize!(pg.neighbor_weights, new_cap)
#     end

#     # --- PASS 2: Fill Neighbor Data ---
#     for i in 1:N
#         # Start writing at the pointer for particle i
#         write_idx = pg.neighbor_pointers[i]
        
#         for j in 1:N
#             if i == j; continue; end
            
#             dist = getDistance(pg, i, j)
#             d_abs = abs(dist)
            
#             if d_abs <= maxDist
#                 d2 = d_abs^2
                
#                 pg.neighbor_indices[write_idx]   = j
#                 pg.neighbor_xdistance[write_idx] = dist
#                 pg.neighbor_weights[write_idx]   = weightFunc(d2)
                
#                 write_idx += 1
#             end
#         end
#     end
    
#     # Recalculate volumes using the new connectivity
#     determineVolumes!(pg)
    
#     return nothing
# end

# """
#     determineVolumes!(pg::ParticleGrid1D)

# Calculates Voronoi volumes using the neighbor list. 
# Robust to unsorted particles.
# """
# function determineVolumes!(pg::ParticleGrid1D)
#     N = pg.N
#     # Fallback if no neighbor is found (e.g. at boundaries or isolated)
#     default_vol = pg.dx 
    
#     for i in 1:N
#         # Find closest left and right neighbors from the neighbor list
#         closest_neg_dist = -Inf
#         closest_pos_dist = Inf
        
#         found_left  = false
#         found_right = false
        
#         start_ptr = pg.neighbor_pointers[i]
#         end_ptr   = start_ptr + pg.num_neighbors[i] - 1
        
#         for k in start_ptr:end_ptr
#             d = pg.neighbor_xdistance[k]
            
#             # Check Left Neighbor (negative distance)
#             if d < 0.0
#                 if d > closest_neg_dist
#                     closest_neg_dist = d
#                     found_left = true
#                 end
#             # Check Right Neighbor (positive distance)
#             elseif d > 0.0
#                 if d < closest_pos_dist
#                     closest_pos_dist = d
#                     found_right = true
#                 end
#             end
#         end
        
#         # Calculate Half-distances
#         # If a neighbor is missing (boundary), we assume dx/2 from the particle center.
#         # Alternatively, for non-periodic, you might use the distance to the wall.
        
#         dist_L = found_left  ? abs(closest_neg_dist) : (pg.dx)
#         dist_R = found_right ? abs(closest_pos_dist) : (pg.dx)
        
#         # Volume is the average of the gap sizes (1D Voronoi)
#         pg.volumes[i] = (dist_L + dist_R) / 2.0
#     end
#     return nothing
# end

"""
Finds all neighbors for particle `i` in a 1D grid within `maxDist`.
This is a helper function for `updateNeighbors!`.
"""
function _find_neighbors_1d(pg::ParticleGrid1D, i::Int, maxDist::Float64)
    N = pg.N
    positions = pg.positions
    pos_i = positions[i]
    
    # Pre-allocate a reasonable number of neighbors
    neighbor_list = Vector{Int}()
    sizehint!(neighbor_list, 2 * ceil(Int, maxDist / pg.dx) + 2)

    if pg.bc == :periodic
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

"""
(1D Implementation) Populates the flat neighbor buffers.
This version is serial, fast, and allocation-free (after the first setup).
"""
function updateNeighbors!(pg::ParticleGrid1D)
    N = pg.N
    maxDist = pg.range_factor * pg.dx
    weightFunc = pg.weight_func
    
    # --- PASS 1: Count Neighbors (Serial) ---
    max_nb = 0
    total_neighbors = 0
    for i in 1:N
        # Find neighbors using the simple sorted search
        # Note: This helper allocates a temporary list,
        # which is unavoidable for this pass.
        num_nb = length(_find_neighbors_1d(pg, i, maxDist)) 
        pg.num_neighbors[i] = num_nb
        pg.neighbor_pointers[i] = total_neighbors + 1
        total_neighbors += num_nb
        max_nb = max(max_nb, num_nb)
    end
    pg.neighbor_pointers[N+1] = total_neighbors + 1
    pg.max_nb = max_nb
    
    # --- Resize flat buffers ---
    if length(pg.neighbor_indices) < total_neighbors
        n = total_neighbors + total_neighbors ÷ 4
        resize!(pg.neighbor_indices, n)
        resize!(pg.neighbor_xdistance, n)
        resize!(pg.neighbor_weights, n)
    end

    # --- PASS 2: Fill Neighbor Data (Serial) ---
    # We use a helper array to track the current write offset for each particle
    offset_counts = zeros(Int, N) 
    
    for i in 1:N
        pos_i = pg.positions[i]
        
        # Find neighbors again (this is fast in 1D)
        neighbor_list = _find_neighbors_1d(pg, i, maxDist)
        
        for j in neighbor_list
            offset = offset_counts[i]
            write_idx = pg.neighbor_pointers[i] + offset
            
            dist_x = getDistance(pg, i, j) # Handles periodicity
            d2 = dist_x^2

            pg.neighbor_indices[write_idx] = j
            pg.neighbor_xdistance[write_idx] = dist_x
            pg.neighbor_weights[write_idx] = weightFunc(d2)
            
            offset_counts[i] += 1
        end
    end
    determineVolumes!(pg)
    return nothing
end

"""
    sort_1d_particles!(pg::ParticleGrid1D)

Sorts the persistent state arrays (positions, rhos, curvatures, etc.) based on
position x_i < x_j for i < j. This is crucial for moving grids (Lagrangian
or semi-Lagrangian) to ensure the 1D neighbor search and boundary handling
remain valid.

Note: This function invalidates the current neighbor graph. You must call 
`updateNeighbors!(pg)` immediately after.
"""
function sort_1d_particles!(pg::ParticleGrid1D)
    # 1. Determine the permutation that sorts the positions
    # sortperm is robust and handles the indices logic for us
    range = pg.N+1:length(pg.positions)
    p = [sortperm(pg.positions[1:pg.N]) ; collect(range) ]

    # Optimization: If already sorted (common in small time steps), exit early
    if issorted(p)
        return nothing
    end

    # 2. Permute all persistent state vectors in-place using the permutation `p`.
    # Base.permute! handles the cycles efficiently.
    Base.permute!(pg.positions, p)
    Base.permute!(pg.rhos, p)
    Base.permute!(pg.curvatures, p)
    Base.permute!(pg.is_boundary, p)
    Base.permute!(pg.volumes, p)
    Base.permute!(pg.mood_events, p)

    # Note: We do not permute neighbor buffers (indices/weights/pointers)
    # because those are meaningless after a position swap and must be 
    # totally rebuilt by updateNeighbors!().
    
    return nothing
end

"""
Calculates the 1D 'volume' (length of the Voronoi cell) for each particle.
"""
function determineVolumes!(particleGrid::ParticleGrid1D)
    N = particleGrid.N
    if N == 0; return; end

    positions = particleGrid.positions
    volumes = particleGrid.volumes
    
    if particleGrid.bc == :periodic
        for i in 1:N
            prev_idx = mod1(i - 1, N)
            next_idx = mod1(i + 1, N)
            # Use getDistance to correctly handle wrapping for edge particles
            deltaPosL = abs(getDistance(particleGrid, i, prev_idx))
            deltaPosR = abs(getDistance(particleGrid, i, next_idx))
            volumes[i] = (deltaPosL + deltaPosR) / 2.0
        end
    else
        # For non-periodic, only calculate for interior points
        for i in particleGrid.interior_indices
            volumes[i] = (positions[i+1] - positions[i-1]) / 2.0
        end
    end
    return
end

"""
(1D Implementation) Updates ghost cell values in the provided `rhos_buffer`.

This operates on a buffer (like `rhoInit` or `rhos` in the timestepper)
to avoid modifying the grid's state mid-step.
"""
function apply_boundary_conditions!(particleGrid::ParticleGrid1D, rhos_buffer::AbstractVector)
    bc = particleGrid.bc

    if bc == :periodic
        return # Periodic boundaries have no ghost cells to update
    
    elseif bc == :fixed_dirichlet
        # Copy the grid's "permanent" rho values into the buffer's ghost cells.
        # This is for fixed inflow/wall boundaries.
        @inbounds for ghost_idx in 1:particleGrid.N
            if particleGrid.is_boundary[ghost_idx]
                # This reads from the grid's persistent state (particleGrid.rhos)
                # and writes to the temporary buffer (rhos_buffer).
                rhos_buffer[ghost_idx] = particleGrid.rhos[ghost_idx]
            end
        end

    elseif bc == :outflow
        # Copy the value from the nearest interior particle (at the edge)
        # into all ghost cells on that side. This uses the sorted nature of the 1D grid.
        interior = particleGrid.interior_indices
        if isempty(interior); return; end
        
        first_interior_idx = first(interior)
        last_interior_idx  = last(interior)

        # Read the boundary values *from the buffer* we are currently working on.
        val_at_left_boundary  = rhos_buffer[first_interior_idx]
        val_at_right_boundary = rhos_buffer[last_interior_idx]

        # Write those values into the ghost cell regions *of the buffer*.
        rhos_buffer[1:(first_interior_idx-1)] .= val_at_left_boundary
        rhos_buffer[(last_interior_idx+1):end] .= val_at_right_boundary
    end
    
    return nothing
end


"""
    apply_boundary_conditions!(particleGrid::ParticleGrid2D, rhos_buffer::AbstractVector)

Updates ghost cell values in the provided `rhos_buffer` for `:outflow`
boundary conditions by finding the closest interior neighbor. This operates
on a buffer to avoid modifying the grid's state mid-step.
"""
function apply_boundary_conditions!(particleGrid::ParticleGrid2D, rhos_buffer::AbstractVector)
    if particleGrid.bc == :periodic; return; 
    elseif particleGrid.bc == :fixed_dirichlet
        for ghost_idx in 1:particleGrid.N
            if particleGrid.is_boundary[ghost_idx]
                rhos_buffer[ghost_idx] = particleGrid.rhos[ghost_idx]
            end
        end
    elseif particleGrid.bc == :outflow
    # Find all ghost particles by checking the is_boundary flag
    @inbounds for ghost_idx in 1:particleGrid.N
        if particleGrid.is_boundary[ghost_idx]
            num_nb = particleGrid.num_neighbors[ghost_idx]
            if num_nb == 0; continue; end

            start_idx = particleGrid.neighbor_pointers[ghost_idx]
            
            min_dist_sq = Inf
            closest_interior_neighbor_idx = -1

            # Search through this ghost particle's neighbors
            @inbounds for k in 1:num_nb
                global_idx = start_idx + k - 1
                neighbor_idx = particleGrid.neighbor_indices[global_idx]
                
                # Check if the neighbor is an interior particle
                if !particleGrid.is_boundary[neighbor_idx]
                    dx = particleGrid.neighbor_xdistance[global_idx]
                    dy = particleGrid.neighbor_ydistance[global_idx]
                    dist_sq = dx^2 + dy^2

                    if dist_sq < min_dist_sq
                        min_dist_sq = dist_sq
                        closest_interior_neighbor_idx = neighbor_idx
                    end
                end
            end
            # Copy the value from the closest neighbor *from the buffer*
            if closest_interior_neighbor_idx != -1
                rhos_buffer[ghost_idx] = rhos_buffer[closest_interior_neighbor_idx]
#            else
#                rhos_buffer[ghost_idx] = particleGrid.rhos[ghost_idx]
            end
        end
    end
end
end
"""
    getTimeStep(particleGrid::ParticleGrid1D, eq::LinearAdvection{1})

Return the maximum time step for the first-order upwind method,
using the pre-calculated neighbor buffers.
"""
function getTimeStep(particleGrid::ParticleGrid1D, eq::LinearAdvection{1})
    dtMax = Inf
    
    # It is assumed updateNeighbors!(particleGrid) has already been called.
    
    vel = velocity(eq, 0.0)

    # Access the flat buffers once
    neighbor_xdistance_full = particleGrid.neighbor_xdistance
    neighbor_weights_full = particleGrid.neighbor_weights

    for particleIndex in particleGrid.interior_indices
        num = 0.0
        denum = 0.0
        
        # --- NEW: Use flat buffer loop ---
        nb_slice = getNBSlice(particleGrid, particleIndex)
        
        @inbounds for k in nb_slice
            dx = neighbor_xdistance_full[k]
            w = neighbor_weights_full[k]
            
            # Upwind condition
            if ((vel >= 0.0) && (dx <= 0.0)) || ((vel <= 0.0) && (dx >= 0.0))
                num += w * dx
                denum += w * dx * dx
            end
        end
        # --- End new loop ---

        if abs(vel * num) > 1e-14
            dtMax = min(-denum / (vel * num), dtMax)
        end
    end
    return dtMax
end
function getTimeStep(particleGrid::ParticleGrid2D{S}, eq::LinearAdvection{2}) where S
    dtMax = Inf
    # Note: It's assumed that `updateNeighbors!` has already been called
    # and has populated all the neighbor_* buffers, including neighbor_weights.

    vel = eq.vel

    for particleIndex in 1:particleGrid.N
        if particleGrid.is_boundary[particleIndex]; continue end
        num_nb = particleGrid.num_neighbors[particleIndex]
        if num_nb == 0; continue; end
        
        start_idx = particleGrid.neighbor_pointers[particleIndex]

        # --- 1. First Pass: Calculate the least-squares matrix A ---
        # This loop reads directly from the global grid buffers. No allocations.
        A11 = 0.0; A12 = 0.0; A22 = 0.0
        @inbounds for k in 1:num_nb
            global_idx = start_idx + k - 1
            
            # Read pre-calculated values directly from the grid
            dx = particleGrid.neighbor_xdistance[global_idx]
            dy = particleGrid.neighbor_ydistance[global_idx]
            w  = particleGrid.neighbor_weights[global_idx]

            # Accumulate for the A matrix
            A11 += w * dx * dx
            A12 += w * dx * dy
            A22 += w * dy * dy
        end

        D = A11 * A22 - (A12^2)
        if abs(D) < 1e-14; continue; end

        # --- 2. Second Pass: Calculate sumCij using the A matrix ---
        # This loop also reads directly from the global grid buffers. No allocations.
        sumCij = 0.0
        @inbounds for k in 1:num_nb
            global_idx = start_idx + k - 1

            # Read pre-calculated values again
            deltaX = particleGrid.neighbor_xdistance[global_idx]
            deltaY = particleGrid.neighbor_ydistance[global_idx]
            w      = particleGrid.neighbor_weights[global_idx]
            
            # Solve 2x2 LS system for coefficients
            coeff_x = (A22 * w * deltaX - A12 * w * deltaY) / D
            coeff_y = (A11 * w * deltaY - A12 * w * deltaX) / D
            
            # Compute adapted coefficients for positivity
            angle = atan(deltaY, deltaX)
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

include("ParticleManagement.jl")

end  # module ParticleGrids