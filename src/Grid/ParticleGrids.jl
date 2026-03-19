include("MLSWeightFunctions.jl")
include("GridMovement.jl")
include("ParticleManagement.jl")

# --- 1. Unified Position Accessors (No more reinterpret hacks!) ---
@inline get_positions(pg::ParticleGrid) = pg.core.positions
@inline get_positions(sb::SharedBuffers) = sb.pos_buffer

# --- 2. Unified Neighbor Accessors ---
@inline get_weights(pg::ParticleGrid)   = pg.neighbor.weights
@inline get_distances(pg::ParticleGrid) = pg.neighbor.distances # Returns Space{D}
@inline get_neighbors(pg::ParticleGrid) = pg.neighbor.indices
# =========================================================================
# UNIFIED DISTANCE CALCULATIONS 
# =========================================================================

# Euclidean version: Just pass the raw array
@inline function getEuclideanDistance(pos::AbstractVector, i::Int, j::Int)
    @inbounds begin
        return pos[j] - pos[i]
    end
end

# Periodic version: Pass the raw array AND the pre-calculated domain size (L)
@inline function getPeriodicDistance(pos::AbstractVector, i::Int, j::Int, L)
    @inbounds begin
        dist = pos[j] - pos[i]
        # Perfectly type-stable, unrolled periodic wrapping for any dimension
        return map((d, l) -> d > 0.5 * l ? d - l : (d < -0.5 * l ? d + l : d), dist, L)
    end
end

# =========================================================================
# FAST SPATIAL HASHING (Coordinates -> 1D Bin Index)
# =========================================================================

# 1D implementation
@inline function get_flat_bin_index(pos::Space{1}, mins::Space{1}, bin_size::Float64, dims::NTuple{1, Int})
    idx = floor(Int, (pos[1] - mins[1]) / bin_size) + 1
    return clamp(idx, 1, dims[1])
end

# 2D implementation (Standard Column-Major Flattening)
@inline function get_flat_bin_index(pos::Space{2}, mins::Space{2}, bin_size::Float64, dims::NTuple{2, Int})
    idx_x = floor(Int, (pos[1] - mins[1]) / bin_size) + 1
    idx_y = floor(Int, (pos[2] - mins[2]) / bin_size) + 1
    
    # Clamp to domain safely
    cx = clamp(idx_x, 1, dims[1])
    cy = clamp(idx_y, 1, dims[2])
    
    return cx + (cy - 1) * dims[1]
end

# Fallback for D-Dimensions (3D+)
@inline function get_flat_bin_index(pos::Space{D}, mins::Space{D}, bin_size::Float64, dims::NTuple{D, Int}) where {D}
    cartesian = ntuple(Val(D)) do d
        clamp(floor(Int, (pos[d] - mins[d]) / bin_size) + 1, 1, dims[d])
    end
    return LinearIndices(dims)[cartesian...]
end

# =========================================================================
# PERIODIC BINS CONSTRUCTOR
# =========================================================================
function GlobalBins(
    mins_tot::NTuple{D, Real}, maxs_tot::NTuple{D, Real}, 
    mins_interior::NTuple{D, Real}, maxs_interior::NTuple{D, Real},
    R::Real, r::Real, max_particles::Int, ::Val{:periodic}
) where {D}
    
    mins_t = Space{D}(mins_tot...)
    maxs_t = Space{D}(maxs_tot...)

    domain_size = maxs_t .- mins_t
    coarse_dims = ntuple(d -> ceil(Int, domain_size[d] / R), Val(D))
    fine_dims   = ntuple(d -> ceil(Int, domain_size[d] / r), Val(D))

    total_coarse_bins = prod(coarse_dims)
    total_fine_bins   = prod(fine_dims)

    head = zeros(Int, total_coarse_bins)
    next = zeros(Int, ceil(Int, max_particles * 1.25))
    
    fine_occ  = zeros(Bool, total_fine_bins)
    
    # Fast path: All bins are interior (1) for periodic boundaries
    fine_type = ones(UInt8, total_fine_bins) 

    return GlobalBins{D, :periodic}(
        mins_t, maxs_t, 
        Float64(R), coarse_dims, head, next,
        Float64(r), fine_dims, fine_occ, fine_type
    )
end

# =========================================================================
# NON-PERIODIC BINS CONSTRUCTOR
# =========================================================================
function GlobalBins(
    mins_tot::NTuple{D, Real}, maxs_tot::NTuple{D, Real}, 
    mins_interior::NTuple{D, Real}, maxs_interior::NTuple{D, Real},
    R::Real, r::Real, max_particles::Int, ::Val{BC}
) where {D, BC}
    
    mins_t = Space{D}(mins_tot...)
    maxs_t = Space{D}(maxs_tot...)
    mins_i = Space{D}(mins_interior...)
    maxs_i = Space{D}(maxs_interior...)

    domain_size = maxs_t .- mins_t
    coarse_dims = ntuple(d -> ceil(Int, domain_size[d] / R), Val(D))
    fine_dims   = ntuple(d -> ceil(Int, domain_size[d] / r), Val(D))

    total_coarse_bins = prod(coarse_dims)
    total_fine_bins   = prod(fine_dims)

    head = zeros(Int, total_coarse_bins)
    next = zeros(Int, ceil(Int, max_particles * 1.25))
    
    fine_occ  = zeros(Bool, total_fine_bins)
    fine_type = zeros(UInt8, total_fine_bins)

    for (flat_idx, I) in enumerate(CartesianIndices(fine_dims))
        bin_center = ntuple(Val(D)) do d
            mins_t[d] + (I[d] - 0.5) * r
        end
        
        is_interior = all(1:D) do d
            bin_center[d] >= mins_i[d] && bin_center[d] <= maxs_i[d]
        end
        
        is_in_domain = all(1:D) do d
            bin_center[d] >= mins_t[d] && bin_center[d] <= maxs_t[d]
        end

        if is_interior
            fine_type[flat_idx] = 1 # Interior
        elseif is_in_domain
            fine_type[flat_idx] = 2 # Ghost/Boundary
        else
            fine_type[flat_idx] = 0 # OutOfBounds
        end
    end

    return GlobalBins{D, BC}(
        mins_t, maxs_t, 
        Float64(R), coarse_dims, head, next,
        Float64(r), fine_dims, fine_occ, fine_type
    )
end

# =========================================================================
# GENERALIZED D-DIMENSIONAL GRID GENERATOR
# =========================================================================

function createParticleGrid(
    mins::NTuple{D, Real}, maxs::NTuple{D, Real}, Ns_interior::NTuple{D, Integer}, 
    bc::Symbol, interp_range_factor::Real;
    M::Int = 1, randomness::Tuple = ntuple(i->0.0, D), 
    rng = Random.default_rng(), merge_factor = 0.3, split_factor = 1., 
    weight_func = exponentialWeightFunction(1.,1.), km = nothing, mover = NoGridMover()
) where {D}
    
    km = isnothing(km) ? Kin2Macro(1:M) : km  
    N_ghost::Int = bc == :periodic ? 0 : ceil(Int, interp_range_factor)
    
    if bc == :periodic
        @assert N_ghost == 0 "Periodic grids do not use ghost cells."
        Ns_total = Ns_interior
        dxs = (maxs .- mins) ./ max.(Ns_interior, 1.0)
    else
        @assert N_ghost >= 0 "N_ghost must be non-negative."
        Ns_total = Ns_interior .+ 2*N_ghost
        dxs = (maxs .- mins) ./ max.(Ns_interior .- 1, 1.0)
    end

    N = prod(Ns_total)
    N_interior_total = prod(Ns_interior)
    N_ghost_total = N - N_interior_total

    # Cast to Static Vectors for type-stable math
    mins_f = Space{D}(mins...)
    maxs_f = Space{D}(maxs...)
    dxs_f  = Space{D}(dxs...)
    rand_f = Space{D}(randomness...)

    mins_tot = mins_f .- N_ghost .* dxs_f
    maxs_tot = maxs_f .+ N_ghost .* dxs_f
    
    max_dx = maximum(dxs_f)
    R = interp_range_factor *  max_dx
    r = split_factor * max_dx
    a = merge_factor * max_dx
    regular = all(==(0.0), randomness)

    positions = Vector{Space{D}}(undef, N)
    is_boundary = zeros(Bool, N)

    # ---------------------------------------------------------
    # D-Dimensional Placement Loop using CartesianIndices
    # ---------------------------------------------------------
    for (i, I) in enumerate(CartesianIndices(Ns_total))
        pos_tuple = ntuple(Val(D)) do d
            idx = I[d]
            
            if bc == :periodic
                return mins_f[d] + dxs_f[d]*(idx - 0.5) + rand_f[d]*(rand(rng, Float64)*2 - 1)
            else
                if idx <= N_ghost
                    return mins_f[d] - (N_ghost - idx + 1) * dxs_f[d]
                elseif idx > Ns_interior[d] + N_ghost
                    return maxs_f[d] + (idx - (Ns_interior[d] + N_ghost)) * dxs_f[d]
                else
                    # Keep single-particle domains perfectly centered
                    base = Ns_interior[d] == 1 ? (mins_f[d] + maxs_f[d]) / 2.0 : mins_f[d] + (idx - N_ghost - 1) * dxs_f[d]
                    return base + rand_f[d] * (rand(rng, Float64) * 2 - 1)
                end
            end
        end
        
        positions[i] = Space{D}(pos_tuple)
        
        if bc != :periodic
            is_boundary[i] = any(d -> I[d] <= N_ghost || I[d] > Ns_interior[d] + N_ghost, 1:D)
        end
    end

    # ---------------------------------------------------------
    # Struct Instantiation
    # ---------------------------------------------------------
    meta = GridMetadata{D}(
        N, N_interior_total, N_ghost_total, mins_tot, maxs_tot, mins_f, maxs_f, 
        R, r, a, dxs_f, regular, bc, Float64(interp_range_factor), 0
    )

    core = ParticleGridCore{D}(positions, is_boundary, zeros(Float64, N))
    shared = SharedBuffers{D, M}(zeros(State{M}, N), similar(positions), zeros(Bool,N), zeros(Int, N))

    reorder = ReorderData{D}(collect(1:N), collect(1:N), zeros(Int, N), zeros(Bool,N))
    bins = GlobalBins(
        Tuple(mins_tot), Tuple(maxs_tot), 
        Tuple(mins_f), Tuple(maxs_f), 
        R, r, N, Val(bc)
    )

    neighbors = NeighborData{D, typeof(weight_func)}(
        weight_func, fill(1:0, N + 1), Int[], 
        Vector{Float64}(undef,0), Vector{SVector{D,Float64}}(undef, 0), 
        zeros(Int, N), zeros(Int, N)
    )

    pg = ParticleGrid{D, M, typeof(weight_func), typeof(mover), bc}(
        meta, core, shared, neighbors, reorder, bins, km, mover,
        zeros(State{M}, N), zeros(SVector{M, Bool}, N), zeros(State{M}, N)
    )

    if D > 1; pg.neighbor(pg); end # Pre-warm CellListMap
    pg.reorder(pg)
    pg.neighbor(pg)
    
    return pg
end

# =========================================================================
# BACKWARDS COMPATIBILITY WRAPPERS
# =========================================================================

# 1D Wrapper
function createParticleGrid(
    ::Val{1}, xmin::Real, xmax::Real, N_interior::Integer, bc::Symbol,
    interp_range_factor::Real;
    randomness::Real = 0.0, kwargs...
)
    return createParticleGrid(
        Space{1}(Float64(xmin)), Space{1}(Float64(xmax)), (Int(N_interior),), 
        bc, Float64(interp_range_factor);
        randomness=Space{1}(Float64(randomness)), kwargs...
    )
end

# 2D Wrapper
function createParticleGrid(
    ::Val{2}, xmin::Real, xmax::Real, ymin::Real, ymax::Real, 
    Nx_interior::Int, Ny_interior::Int, bc::Symbol, interp_range_factor::Real;
    randomness::NTuple{2, Float64} = (0.0, 0.0), kwargs...
)
    return createParticleGrid(
        Space{2}(Float64(xmin), Float64(ymin)), Space{2}(Float64(xmax), Float64(ymax)), 
        (Int(Nx_interior), Int(Ny_interior)), 
        bc, Float64(interp_range_factor);
        randomness=Space{2}(Float64.(randomness)), kwargs...
    )
end

# Extractor simply returns the cached UnitRange
@inline function getNBSlice(pg::ParticleGrid, p_idx::Int)
    return pg.neighbor.ranges[p_idx]
end

sort_particles!(pg::ParticleGrid) = pg.reorder(pg) 

function (rd::ReorderData{D})(pg::ParticleGrid{D, M, WF}) where {D, M, WF}
    N = pg.meta.N
    visited = rd.seen_buffer
    fill!(visited, false)
    perm_idx = 0
    queue = Int[]
    neighbor_buffer = Int[]

    for i in 1:N 
        if !visited[i]
            start_node = i
            visited[start_node] = true
            resize!(queue, 0)
            push!(queue, start_node)

            while !isempty(queue)
                current_node = popfirst!(queue)
                perm_idx += 1
                rd.permutation[perm_idx] = current_node

                resize!(neighbor_buffer, 0)
                nb_slice = pg.neighbor.ranges[current_node]
                
                if !isempty(nb_slice)
                    @inbounds for k in nb_slice
                        nb_idx = pg.neighbor.indices[k]
                        if !visited[nb_idx]
                            visited[nb_idx] = true 
                            push!(neighbor_buffer, nb_idx)
                        end
                    end
                end
                
                # Sort neighbors by their degree (length of their UnitRange)
                sort!(neighbor_buffer, by = idx -> length(pg.neighbor.ranges[idx]))
                append!(queue, neighbor_buffer)
            end
        end
    end
    
    reverse!(rd.permutation)

    copyto!(pg.shared.pos_buffer, pg.core.positions)
    copyto!(pg.shared.bit_buffer, pg.core.is_boundary)
    copyto!(pg.shared.rho_buffer, pg.rhos)

    @batch for i in 1:N
        src_idx = rd.permutation[i]
        pg.core.positions[i]   = pg.shared.pos_buffer[src_idx]
        pg.core.is_boundary[i] = pg.shared.bit_buffer[src_idx]
        pg.rhos[i]             = pg.shared.rho_buffer[src_idx] # Single vector copy!
    end

    @batch for i in 1:N
        rd.inv_permutation[rd.permutation[i]] = i
    end
    return nothing
end

updateNeighbors!(pg::ParticleGrid) = pg.neighbor(pg)

function (nd::NeighborData{1, WF})(pg::ParticleGrid{1, M, WF}) where {M, WF}
    N = pg.meta.N
    maxDist = pg.meta.range_factor * pg.meta.dx[1]
    weightFunc = nd.weight_func
    
    max_nb = 0
    total_neighbors = 0
    for i in 1:N
        num_nb = length(_find_neighbors_1d(pg, i, maxDist))
        pg.neighbor.ranges[i] = (total_neighbors + 1):(total_neighbors + num_nb)
        total_neighbors += num_nb
        max_nb = max(max_nb, num_nb)
    end
    pg.neighbor.ranges[N+1] = (total_neighbors + 1):total_neighbors
    pg.meta.max_nb = max_nb
    
    current_capacity = length(pg.neighbor.indices)
    if total_neighbors > current_capacity
        new_capacity = ceil(Int, total_neighbors * 1.25)
        resize!(pg.neighbor.indices, new_capacity)
        resize!(pg.neighbor.weights, new_capacity)
        resize!(pg.neighbor.distances, new_capacity)
    end

    offset_counts = zeros(Int, N) 
    for i in 1:N
        neighbor_list = _find_neighbors_1d(pg, i, maxDist)
        for j in neighbor_list
            offset = offset_counts[i]
            write_idx = pg.neighbor.ranges[i].start + offset
            
            dist_x = getDistance(pg, i, j)[1] 
            d2 = dist_x^2

            pg.neighbor.indices[write_idx] = j
            pg.neighbor.weights[write_idx] = weightFunc(d2)
            pg.neighbor.distances[write_idx] = SVector{1,Float64}(dist_x)
            
            offset_counts[i] += 1
        end
    end
    determineVolumes!(pg) 
    return nothing
end

function _find_neighbors_1d(pg::ParticleGrid1D, i::Int, maxDist::Float64)
    N = pg.meta.N
    positions = get_positions(pg)
    pos_i = positions[i][1]
    
    neighbor_list = Vector{Int}()
    sizehint!(neighbor_list, 2 * ceil(Int, maxDist / pg.meta.dx[1]) + 2)

    if pg.meta.bc == :periodic
        for j_offset in 1:div(N, 2)
            j = mod1(i - j_offset, N)
            dist = abs(getDistance(pg, i, j)[1])
            if dist <= maxDist; push!(neighbor_list, j)
            else; break; end
        end
        for j_offset in 1:div(N, 2)
            j = mod1(i + j_offset, N)
            dist = abs(getDistance(pg, i, j)[1])
            if dist <= maxDist; push!(neighbor_list, j)
            else; break; end
        end
    else 
        for j in (i-1):-1:1
            if abs(positions[j][1] - pos_i) <= maxDist; push!(neighbor_list, j)
            else; break; end
        end
        for j in (i+1):N
            if abs(positions[j][1] - pos_i) <= maxDist; push!(neighbor_list, j)
            else; break; end
        end
    end
    return neighbor_list
end

# =========================================================================
# GLOBAL BIN BUILDING (Dispatched)
# =========================================================================

# User-facing wrapper
build_global_bins!(pg::ParticleGrid) = _build_global_bins!(pg, Val(pg.meta.bc))

# PERIODIC (No boundary checks needed)
function _build_global_bins!(pg::ParticleGrid{D}, ::Val{:periodic}) where {D}
    N = pg.meta.N
    pos = get_positions(pg)
    bins = pg.bins
    
    fill!(bins.head, 0)
    fill!(bins.fine_occupation, false)
    
    @inbounds for i in 1:N
        p_pos = pos[i]
        
        # --- FINE GRID ---
        fine_idx = get_flat_bin_index(p_pos, bins.mins, bins.fine_size, bins.fine_dims)
        bins.fine_occupation[fine_idx] = true
        pg.core.is_boundary[i] = false # Everything is interior
        
        # --- COARSE GRID ---
        coarse_idx = get_flat_bin_index(p_pos, bins.mins, bins.coarse_size, bins.coarse_dims)
        bins.next[i] = bins.head[coarse_idx]
        bins.head[coarse_idx] = i
    end
    return nothing
end

# NON-PERIODIC (Reads pre-computed fine_type for instant boundary flagging)
function _build_global_bins!(pg::ParticleGrid{D}, ::Val{BC}) where {D, BC}
    N = pg.meta.N
    pos = get_positions(pg)
    bins = pg.bins
    
    fill!(bins.head, 0)
    fill!(bins.fine_occupation, false)
    
    @inbounds for i in 1:N
        p_pos = pos[i]
        
        # --- FINE GRID ---
        fine_idx = get_flat_bin_index(p_pos, bins.mins, bins.fine_size, bins.fine_dims)
        
        bin_type = bins.fine_type[fine_idx]
        if bin_type == 0 # OutOfBounds
            pg.core.is_boundary[i] = true
        else
            pg.core.is_boundary[i] = (bin_type == 2) # True if Ghost, False if Interior
            bins.fine_occupation[fine_idx] = true
        end
        
        # --- COARSE GRID ---
        coarse_idx = get_flat_bin_index(p_pos, bins.mins, bins.coarse_size, bins.coarse_dims)
        bins.next[i] = bins.head[coarse_idx]
        bins.head[coarse_idx] = i
    end
    return nothing
end
# =========================================================================
# PERIODIC NEIGHBOR SEARCH (Branchless & Type-Stable)
# =========================================================================
function (nd::NeighborData{D, WF})(pg::ParticleGrid{D, M, WF, GM, :periodic}) where {D, M, WF, GM}
    build_global_bins!(pg)

    N = pg.meta.N
    pos = get_positions(pg)
    R_sq = pg.meta.R^2
    weightFunc = nd.weight_func
    bins = pg.bins
    
    # Pre-compute domain size for periodic distance
    L = pg.meta.maxs .- pg.meta.mins

    coarse_dims = bins.coarse_dims
    ci = CartesianIndices(coarse_dims)
    li = LinearIndices(coarse_dims)
    window = CartesianIndices(ntuple(_ -> -1:1, Val(D)))

    # --- PASS 1: COUNTING ---
    counts = nd.counts
    fill!(counts, 0)
    
    @batch for i in 1:N
        bin_idx = get_flat_bin_index(pos[i], bins.mins, bins.coarse_size, coarse_dims)
        cart_idx = ci[bin_idx]
        
        c = 0
        for offset in window
            nb_cart = cart_idx + offset
            
            # Pure Tuple mapping: perfectly type-stable, branchless periodic wrapping
            wrapped_cart = map((nc, cd) -> mod1(nc, cd), Tuple(nb_cart), coarse_dims)
            nb_bin_idx = li[CartesianIndex(wrapped_cart)]
            
            j = bins.head[nb_bin_idx]
            while j > 0
                if i != j
                    dist = getPeriodicDistance(pos, i, j, L)
                    d2 = sum(abs2, dist)
                    if d2 <= R_sq
                        c += 1
                    end
                end
                j = bins.next[j]
            end
        end
        counts[i] = c
    end

    # --- SEQUENTIAL PREFIX SUM ---
    starts = pg.shared.int_buffer 
    max_so_far = 0 
    current_ptr = 1
    
    @inbounds for i in 1:N
        c = counts[i]
        pg.neighbor.ranges[i] = current_ptr:(current_ptr + c - 1)
        starts[i] = current_ptr
        current_ptr += c
        if c > max_so_far; max_so_far = c; end
    end
    
    pg.meta.max_nb = max_so_far
    pg.neighbor.ranges[N + 1] = current_ptr:(current_ptr - 1)

    total_neighbors = current_ptr - 1
    ensure_capacity!(nd, total_neighbors) 

    # --- PASS 2: WRITING ---
    indices = pg.neighbor.indices
    weights = pg.neighbor.weights
    distances = pg.neighbor.distances
    
    offsets = nd.offsets
    fill!(offsets, 0)

    @batch for i in 1:N
        bin_idx = get_flat_bin_index(pos[i], bins.mins, bins.coarse_size, coarse_dims)
        cart_idx = ci[bin_idx]
        
        for offset in window
            nb_cart = cart_idx + offset
            
            wrapped_cart = map((nc, cd) -> mod1(nc, cd), Tuple(nb_cart), coarse_dims)
            nb_bin_idx = li[CartesianIndex(wrapped_cart)]
            
            j = bins.head[nb_bin_idx]
            while j > 0
                if i != j
                    dist = getPeriodicDistance(pos, i, j, L)
                    d2 = sum(abs2, dist)
                    
                    if d2 <= R_sq
                        @inbounds begin
                            write_idx = starts[i] + offsets[i]
                            offsets[i] += 1
                            
                            indices[write_idx]   = j
                            weights[write_idx]   = weightFunc(d2) 
                            distances[write_idx] = dist
                        end
                    end
                end
                j = bins.next[j]
            end
        end
    end
    return nothing
end

# =========================================================================
# NON-PERIODIC NEIGHBOR SEARCH (Branchless Bounds Checking)
# =========================================================================
function (nd::NeighborData{D, WF})(pg::ParticleGrid{D, M, WF, GM, BC}) where {D, M, WF, GM, BC}
    build_global_bins!(pg)

    N = pg.meta.N
    pos = get_positions(pg)
    R_sq = pg.meta.R^2
    weightFunc = nd.weight_func
    bins = pg.bins
    
    coarse_dims = bins.coarse_dims
    ci = CartesianIndices(coarse_dims)
    li = LinearIndices(coarse_dims)
    window = CartesianIndices(ntuple(_ -> -1:1, Val(D)))

    # --- PASS 1: COUNTING ---
    counts = nd.counts
    fill!(counts, 0)
    
    @batch for i in 1:N
        bin_idx = get_flat_bin_index(pos[i], bins.mins, bins.coarse_size, coarse_dims)
        cart_idx = ci[bin_idx]
        
        c = 0
        for offset in window
            nb_cart = cart_idx + offset
            
            # Fast, native boundary check
            if !checkbounds(Bool, li, nb_cart)
                continue
            end
            nb_bin_idx = li[nb_cart]
            
            j = bins.head[nb_bin_idx]
            while j > 0
                if i != j
                    dist = getEuclideanDistance(pos, i, j)
                    d2 = sum(abs2, dist)
                    if d2 <= R_sq
                        c += 1
                    end
                end
                j = bins.next[j]
            end
        end
        counts[i] = c
    end

    # --- SEQUENTIAL PREFIX SUM ---
    starts = pg.shared.int_buffer 
    max_so_far = 0 
    current_ptr = 1
    
    @inbounds for i in 1:N
        c = counts[i]
        pg.neighbor.ranges[i] = current_ptr:(current_ptr + c - 1)
        starts[i] = current_ptr
        current_ptr += c
        if c > max_so_far; max_so_far = c; end
    end
    
    pg.meta.max_nb = max_so_far
    pg.neighbor.ranges[N + 1] = current_ptr:(current_ptr - 1)

    total_neighbors = current_ptr - 1
    ensure_capacity!(nd, total_neighbors) 

    # --- PASS 2: WRITING ---
    indices = pg.neighbor.indices
    weights = pg.neighbor.weights
    distances = pg.neighbor.distances
    
    offsets = nd.offsets
    fill!(offsets, 0)

    @batch for i in 1:N
        bin_idx = get_flat_bin_index(pos[i], bins.mins, bins.coarse_size, coarse_dims)
        cart_idx = ci[bin_idx]
        
        for offset in window
            nb_cart = cart_idx + offset
            
            if !checkbounds(Bool, li, nb_cart)
                continue
            end
            nb_bin_idx = li[nb_cart]
            
            j = bins.head[nb_bin_idx]
            while j > 0
                if i != j
                    dist = getEuclideanDistance(pos, i, j) 
                    d2 = sum(abs2, dist)
                    
                    if d2 <= R_sq
                        @inbounds begin
                            write_idx = starts[i] + offsets[i]
                            offsets[i] += 1
                            
                            indices[write_idx]   = j
                            weights[write_idx]   = weightFunc(d2) 
                            distances[write_idx] = dist
                        end
                    end
                end
                j = bins.next[j]
            end
        end
    end
    return nothing
end

reorder_particles!(pg::ParticleGrid) = pg.reorder(pg)

function (rd::ReorderData{1})(pg::ParticleGrid{1, M, WF}) where {M, WF}
    N = pg.meta.N
    range = (N + 1):length(pg.core.positions)
    p = [sortperm(pg.core.positions[1:N]); collect(range)]
    if issorted(p); return nothing; end
    
    Base.permute!(pg.core.positions, p)
    Base.permute!(pg.core.is_boundary, p)
    Base.permute!(pg.rhos, p)          # Native permutation!
    Base.permute!(pg.curvatures, p)    # Native permutation!
    Base.permute!(pg.mood_events, p)   # Native permutation!
    
    return nothing
end

function apply_boundary_conditions!(pg::ParticleGrid{D, M}, rhos_buffer::AbstractVector{State{M}}) where {D, M}
    bc = pg.meta.bc
    
    # 1. Periodic needs no manual overriding; neighbors wrap automatically
    if bc == :periodic
        return nothing
    end

    # 2. Fixed Dirichlet: Reset boundaries to their initial states
    if bc == :fixed_dirichlet
        @inbounds for i in 1:pg.meta.N
            if pg.core.is_boundary[i]
                rhos_buffer[i] = pg.rhos[i] 
            end
        end
        return nothing
    end
    
    # 3. Outflow (Zero-Gradient): Adopt the state of the closest interior neighbor
    if bc == :outflow
        dist_vec = get_distances(pg)
        
        @inbounds for i in 1:pg.meta.N
            if pg.core.is_boundary[i]
                nb_slice = pg.neighbor.ranges[i]
                
                closest_j = -1
                min_dist_sq = Inf
                
                # Search the local support domain for the nearest interior particle
                for k in nb_slice
                    j = pg.neighbor.indices[k]
                    
                    if !pg.core.is_boundary[j] # Ensure it's an interior particle!
                        dx = dist_vec[k]       # dx is natively a Space{D}
                        d2 = sum(abs2, dx)     # Fast squared distance
                        
                        if d2 < min_dist_sq
                            min_dist_sq = d2
                            closest_j = j
                        end
                    end
                end
                
                if closest_j != -1
                    rhos_buffer[i] = rhos_buffer[closest_j]
                else
                    # Fallback in case the support domain is too small to see the interior
                    rhos_buffer[i] = pg.rhos[i]
                end
            end
        end
    end
    
    return nothing
end

determineVolumes!(pg) = return

function determineVolumes!(pg::ParticleGrid1D)
    N = pg.meta.N

    if N == 0; return; end
    positions = get_positions(pg)
    volumes = pg.core.volumes
    
    if pg.meta.bc == :periodic
        for i in 1:N
            prev_idx = mod1(i - 1, N)
            next_idx = mod1(i + 1, N)
            deltaPosL = abs(getDistance(pg, i, prev_idx)[1])
            deltaPosR = abs(getDistance(pg, i, next_idx)[1])
            volumes[i] = (deltaPosL + deltaPosR) / 2.0
        end
    else
        for i in 1:N
            if pg.core.is_boundary[i]; continue end
            volumes[i] = (positions[i+1][1] - positions[i-1][1]) / 2.0
        end
    end
    return
end

function findLocalExtrema(rho_i::Float64, nb_slice::UnitRange{Int}, neighbor_fs::AbstractVector{Float64})::Tuple{Float64, Float64}
    minU = maxU = rho_i
    @inbounds for k in nb_slice 
        rho_j = neighbor_fs[k]
        minU = min(minU, rho_j)
        maxU = max(maxU, rho_j)
    end
    return (minU, maxU)
end

function findLocalExtremaAbs(curve_i::Float64, nb_slice::UnitRange{Int}, neighbor_indices_full::Vector{Int}, curveVec::AbstractVector{Float64})::Tuple{Float64, Float64, Float64, Float64}
    mini = maxi = curve_i
    minAbs = maxAbs = abs(curve_i)
    
    @inbounds for k in nb_slice
        j = neighbor_indices_full[k]
        curve_j = curveVec[j]
        abs_curve_j = abs(curve_j)
        mini = min(mini, curve_j); maxi = max(maxi, curve_j)
        minAbs = min(minAbs, abs_curve_j); maxAbs = max(maxAbs, abs_curve_j)
    end
    return (mini, maxi, minAbs, maxAbs)
end

function findLocalExtremaAbs(
    curve_xx_i::Float64, curve_yy_i::Float64, nb_slice::UnitRange{Int}, neighbor_indices_full::Vector{Int}, 
    curve_xx_Vec::AbstractVector{Float64}, curve_yy_Vec::AbstractVector{Float64}
)::NTuple{8, Float64}
    mini1 = maxi1 = curve_xx_i
    mini2 = maxi2 = curve_yy_i
    minAbs1 = maxAbs1 = abs(curve_xx_i)
    minAbs2 = maxAbs2 = abs(curve_yy_i)
    
    @inbounds for k in nb_slice
        j = neighbor_indices_full[k] 
        curve_xx_j = curve_xx_Vec[j]; curve_yy_j = curve_yy_Vec[j]
        abs_curve_xx_j = abs(curve_xx_j); abs_curve_yy_j = abs(curve_yy_j)

        mini1 = min(mini1, curve_xx_j); maxi1 = max(maxi1, curve_xx_j)
        minAbs1 = min(minAbs1, abs_curve_xx_j); maxAbs1 = max(maxAbs1, abs_curve_xx_j)
        
        mini2 = min(mini2, curve_yy_j); maxi2 = max(maxi2, curve_yy_j)
        minAbs2 = min(minAbs2, abs_curve_yy_j); maxAbs2 = max(maxAbs2, abs_curve_yy_j)
    end
    return (mini1, maxi1, minAbs1, maxAbs1, mini2, maxi2, minAbs2, maxAbs2)
end

@inline function getTimeStep(pg::ParticleGrid{D}, eq) where {D}
    dtMax = Inf
    
    # Extract base wave speeds safely into an SVector. 
    # runSimulation.jl guarantees eq_for_dt is a LinearAdvection object.
    vel = Space{D}(ntuple(d -> eq.vel[d][1], Val(D)))

    w_vec = get_weights(pg)
    dist_vec = get_distances(pg)

    for i in 1:pg.meta.N
        if pg.core.is_boundary[i]
            continue
        end
        
        nb_slice = pg.neighbor.ranges[i]
        if isempty(nb_slice)
            continue
        end

        # 1. Build the MLS Matrix N_s (Exactly like your Interpolator!)
        N_s = @SMatrix zeros(Float64, D, D)
        @inbounds for k in nb_slice
            w  = w_vec[k]
            dx = dist_vec[k] # This is already Space{D}
            N_s += w * (dx * dx')
        end

        if abs(det(N_s)) < 1e-14
            continue
        end

        # 2. Compile-time analytic inversion using StaticArrays
        inv_N_s = inv(N_s)

        # 3. Accumulate the stability condition
        sum_c = 0.0
        @inbounds for k in nb_slice
            w  = w_vec[k]
            dx = dist_vec[k]
            
            # C_k is the effective MLS shape function vector
            C_k = inv_N_s * (w * dx)
            
            # Evaluate the upwind contribution: dot(velocity, shape_gradient)
            c_k = dot(vel, C_k)
            if c_k < 0.0
                sum_c -= c_k
            end
        end

        if sum_c > 1e-14
            # The factor D automatically scales CFL for 1D (D=1) and 2D (D=2)!
            dt_i = 1.0 / (D * sum_c)
            dtMax = min(dt_i, dtMax)
        end
    end
    return dtMax
end

# =========================================================================
# OVERLOADED CAPACITY MANAGERS
# =========================================================================

"""
    ensure_capacity!(rd::ReorderData, req_capacity::Int)
"""
@inline function ensure_capacity!(rd::ReorderData, req_capacity::Int)
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
@inline function ensure_capacity!(core::ParticleGridCore, req_capacity::Int)
    if length(core.positions) < req_capacity
        new_cap = ceil(Int, req_capacity * 1.25)
        
        resize!(core.positions, new_cap)
        resize!(core.is_boundary, new_cap)
        resize!(core.volumes, new_cap)
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
    
Note: This only resizes the arrays mapped to the number of PARTICLES (N). 
The `indices` and `data` arrays map to the number of NEIGHBORS, which scales 
differently and must be resized separately during the neighbor search.
"""
@inline function ensure_capacity!(nd::NeighborData{D}, req_neighbors::Int) where {D}
    if length(nd.indices) < req_neighbors
        new_cap = ceil(Int, req_neighbors * 1.25)
        resize!(nd.indices, new_cap)
        resize!(nd.weights, new_cap)
        resize!(nd.distances, new_cap)
    end
    return nothing
end

@inline function ensure_capacity!(sb::SharedBuffers{D, M}, req_capacity::Int) where {D, M}
    if length(sb.pos_buffer) < req_capacity
        new_cap = ceil(Int, req_capacity * 1.25)
        
        resize!(sb.pos_buffer, new_cap)
        resize!(sb.bit_buffer, new_cap)
        resize!(sb.int_buffer, new_cap)
        resize!(sb.rho_buffer, new_cap)
    end
    return nothing
end

@inline function ensure_capacity!(pg::ParticleGrid, req_capacity::Int)
    if length(pg.rhos) < req_capacity
        new_cap = ceil(Int, req_capacity * 1.25)
        
        # Native resize! now works for everything
        resize!(pg.rhos, new_cap)
        resize!(pg.mood_events, new_cap)
        resize!(pg.curvatures, new_cap)
    end
    
    # Safely delegate down the chain
    ensure_capacity!(pg.core, req_capacity)
    ensure_capacity!(pg.shared, req_capacity)
    ensure_capacity!(pg.reorder, req_capacity)
    ensure_capacity!(pg.neighbor, req_capacity)
    ensure_capacity!(pg.bins, req_capacity)
    
    return nothing
end