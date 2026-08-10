include("MLSWeightFunctions.jl")
include("GridMovement.jl")
#include("ParticleManagement.jl")

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

@inline function getPeriodicDistance(pos::AbstractVector{Space{D, T}}, i::Int, j::Int, L::Space{D, T}, invL::Space{D, T}) where {D, T}
    @inbounds begin
        p_i = pos[i]
        p_j = pos[j]
        
        return Space{D, T}(ntuple(Val(D)) do d
            dx = p_j[d] - p_i[d]
            
            # Replaced division with multiplication! (FMA friendly)
            dx - L[d] * round(dx * invL[d]) 
        end)
    end
end

# =========================================================================
# UNIFIED DISTANCE DISPATCHER
# =========================================================================
@inline _get_dist(pos, i, j, L, L_inv, ::Val{:periodic}) = getPeriodicDistance(pos, i, j, L, L_inv)
@inline _get_dist(pos, i, j, L, L_inv, ::Val) = getEuclideanDistance(pos, i, j)

# =========================================================================
# FAST SPATIAL HASHING (Coordinates -> 1D Bin Index)
# =========================================================================

@inline function get_flat_bin_index(pos::Space{1, T}, mins::Space{1, T}, bin_size::Space{1, T}, dims::NTuple{1, Int}) where {T}
    idx = floor(Int, (pos[1] - mins[1]) / bin_size[1]) + 1
    return clamp(idx, 1, dims[1])
end

@inline function get_flat_bin_index(pos::Space{2, T}, mins::Space{2, T}, bin_size::Space{2, T}, dims::NTuple{2, Int}) where {T}
    idx_x = floor(Int, (pos[1] - mins[1]) / bin_size[1]) + 1
    idx_y = floor(Int, (pos[2] - mins[2]) / bin_size[2]) + 1
    return clamp(idx_x, 1, dims[1]) + (clamp(idx_y, 1, dims[2]) - 1) * dims[1]
end

@inline function get_flat_bin_index(pos::Space{D, T}, mins::Space{D, T}, bin_size::Space{D, T}, dims::NTuple{D, Int}) where {D, T}
    cartesian = ntuple(Val(D)) do d
        clamp(floor(Int, (pos[d] - mins[d]) / bin_size[d]) + 1, 1, dims[d])
    end
    return LinearIndices(dims)[cartesian...]
end
# =========================================================================
# PERIODIC BINS CONSTRUCTOR (Perfect Tiling)
# =========================================================================
function GlobalBins(
    ::Type{T}, mins_tot::NTuple{D, Real}, maxs_tot::NTuple{D, Real}, 
    mins_interior::NTuple{D, Real}, maxs_interior::NTuple{D, Real},
    R::Real, r::Real, max_particles::Int, buffer::Vector{SVector{N_OFF, Int}}, ::Val{:periodic}
) where {D, N_OFF, T}
    
    mins_t = Space{D, T}(mins_tot...)
    maxs_t = Space{D, T}(maxs_tot...)
    domain_size = maxs_t .- mins_t

    coarse_dims = ntuple(d -> max(1, floor(Int, domain_size[d] / R)), Val(D))
    fine_dims   = ntuple(d -> max(1, floor(Int, domain_size[d] / r)), Val(D))

    coarse_size = domain_size ./ coarse_dims
    fine_size   = domain_size ./ fine_dims

    total_coarse_bins = prod(coarse_dims)
    total_fine_bins   = prod(fine_dims)

    head = zeros(Int, total_coarse_bins)
    next = zeros(Int, ceil(Int, max_particles * 1.25))
    
    fine_occ  = zeros(Bool, total_fine_bins)
    fine_type = ones(UInt8, total_fine_bins) 

    return GlobalBins{D, T, :periodic, N_OFF}(
        mins_t, maxs_t, 
        coarse_size, coarse_dims, head, next,
        fine_size, fine_dims, fine_occ, fine_type, buffer
    )
end


function GlobalBins(
    ::Type{T}, mins_tot::NTuple{D, Real}, maxs_tot::NTuple{D, Real}, 
    mins_interior::NTuple{D, Real}, maxs_interior::NTuple{D, Real},
    R::Real, r::Real, max_particles::Int, buffer::Vector{SVector{N_OFF, Int}}, ::Val{BC}
) where {D, BC, N_OFF, T}
    
    mins_t = Space{D, T}(mins_tot...)
    maxs_t = Space{D, T}(maxs_tot...)
    mins_i = Space{D, T}(mins_interior...)
    maxs_i = Space{D, T}(maxs_interior...)

    domain_size = maxs_t .- mins_t
    coarse_dims = ntuple(d -> ceil(Int, domain_size[d] / R), Val(D))
    fine_dims   = ntuple(d -> ceil(Int, domain_size[d] / r), Val(D))

    coarse_size = Space{D, T}(ntuple(_ -> T(R), Val(D)))
    fine_size   = Space{D, T}(ntuple(_ -> T(r), Val(D)))

    total_coarse_bins = prod(coarse_dims)
    total_fine_bins   = prod(fine_dims)

    head = zeros(Int, total_coarse_bins)
    next = zeros(Int, ceil(Int, max_particles * 1.25))
    
    fine_occ  = zeros(Bool, total_fine_bins)
    fine_type = zeros(UInt8, total_fine_bins)

    for (flat_idx, I) in enumerate(CartesianIndices(fine_dims))
        bin_center = ntuple(Val(D)) do d
            mins_t[d] + (I[d] - T(0.5)) * fine_size[d]
        end
        
        is_interior = all(1:D) do d
            bin_center[d] >= mins_i[d] && bin_center[d] <= maxs_i[d]
        end
        
        is_in_domain = all(1:D) do d
            bin_center[d] >= mins_t[d] && bin_center[d] <= maxs_t[d]
        end

        if is_interior
            fine_type[flat_idx] = 1 
        elseif is_in_domain
            fine_type[flat_idx] = 2 
        else
            fine_type[flat_idx] = 0 
        end
    end

    return GlobalBins{D, T, BC, N_OFF}(
        mins_t, maxs_t, 
        coarse_size, coarse_dims, head, next,
        fine_size, fine_dims, fine_occ, fine_type, buffer
    )
end

function createParticleGrid(
    ::Type{T}, mins::NTuple{D, Real}, maxs::NTuple{D, Real}, Ns_interior::NTuple{D, Integer}, 
    bc::Symbol, interp_range_factor::Real;
    M::Int = 1, randomness::Tuple = ntuple(i->zero(T), D), 
    rng = Random.default_rng(), merge_factor = T(0.3), split_factor = one(T), 
    weight_func = ExponentialWeightFunction(one(T), one(T)), mover = NoGridMover()
) where {D, T}
    
    N_ghost::Int = bc == :periodic ? 0 : ceil(Int, interp_range_factor)
    
    if bc == :periodic
        @assert N_ghost == 0 "Periodic grids do not use ghost cells."
        Ns_total = Ns_interior
        dxs = (maxs .- mins) ./ max.(Ns_interior, T(1.0))
    else
        @assert N_ghost >= 0 "N_ghost must be non-negative."
        Ns_total = Ns_interior .+ 2*N_ghost
        dxs = (maxs .- mins) ./ max.(Ns_interior .- 1, T(1.0))
    end

    N = prod(Ns_total)
    N_interior_total = prod(Ns_interior)
    N_ghost_total = N - N_interior_total

    mins_f = Space{D, T}(mins...)
    maxs_f = Space{D, T}(maxs...)
    dxs_f  = Space{D, T}(dxs...)
    rand_f = Space{D, T}(randomness...)

    mins_tot = mins_f .- N_ghost .* dxs_f
    maxs_tot = maxs_f .+ N_ghost .* dxs_f
    
    max_dx = maximum(dxs_f)
    R = T(interp_range_factor) * max_dx
    r = T(split_factor) * max_dx
    a = T(merge_factor) * max_dx
    regular = all(==(zero(T)), randomness)

    positions = Vector{Space{D, T}}(undef, N)
    is_boundary = zeros(Bool, N)

    for (i, I) in enumerate(CartesianIndices(Ns_total))
        pos_tuple = ntuple(Val(D)) do d
            idx = I[d]
            
            if bc == :periodic
                return mins_f[d] + dxs_f[d]*(idx - T(0.5)) + rand_f[d]*(rand(rng, T)*2 - 1)
            else
                if idx <= N_ghost
                    return mins_f[d] - (N_ghost - idx + 1) * dxs_f[d]
                elseif idx > Ns_interior[d] + N_ghost
                    return maxs_f[d] + (idx - (Ns_interior[d] + N_ghost)) * dxs_f[d]
                else
                    base = Ns_interior[d] == 1 ? (mins_f[d] + maxs_f[d]) / T(2.0) : mins_f[d] + (idx - N_ghost - 1) * dxs_f[d]
                    return base + rand_f[d] * (rand(rng, T) * 2 - 1)
                end
            end
        end
        
        positions[i] = Space{D, T}(pos_tuple)
        if bc != :periodic
            is_boundary[i] = any(d -> I[d] <= N_ghost || I[d] > Ns_interior[d] + N_ghost, 1:D)
        end
    end

    L = maxs_f - mins_f
    meta = GridMetadata{D, T}(
        N, N_interior_total, N_ghost_total, mins_tot, maxs_tot, mins_f, maxs_f, L, one(T) ./ L,
        R, r, a, dxs_f, regular, bc, T(interp_range_factor), 0
    )

    core = ParticleGridCore{D, T}(positions, is_boundary, zeros(T, N))
    shared = SharedBuffers{D, M, T}(zeros(State{M, T}, N), similar(positions), zeros(T, N), zeros(Bool, N), zeros(Int, N))
    reorder = ReorderData{D}(collect(1:N), collect(1:N), zeros(Int, N), zeros(Bool, N))

    N_OFF = get_n_offsets(Val(D))
    bins = GlobalBins(
        T, Tuple(mins_tot), Tuple(maxs_tot), Tuple(mins_f), Tuple(maxs_f), 
        R, r, N, Vector{SVector{N_OFF, Int}}(undef, 0), Val(bc), 
    )

    neighbors = NeighborData{D, T, typeof(weight_func)}(
        weight_func, fill(1:0, N + 1), Int[], 
        Vector{T}(undef, 0), Vector{Space{D, T}}(undef, 0), 
        zeros(Int, N), zeros(Int, N)
    )

    pg = ParticleGrid{D, M, T, typeof(weight_func), typeof(mover), bc, N_OFF}(
        meta, core, shared, neighbors, reorder, bins, mover,
        zeros(State{M, T}, N), zeros(SVector{M, Bool}, N), zeros(State{M, T}, N)
    )

    if D > 1; pg.neighbor(pg); end 
    pg.reorder(pg)
    pg.neighbor(pg)
    
    return pg
end

# Extractor simply returns the cached UnitRange
@inline function getNBSlice(pg::ParticleGrid, p_idx::Int)
    return pg.neighbor.ranges[p_idx]
end

sort_particles!(pg::ParticleGrid) = pg.reorder(pg) 

# function (rd::ReorderData{D})(pg::ParticleGrid{D, M, WF}) where {D, M, WF}
#     N = pg.meta.N
    
#     # 1. Lexicographical Spatial Sort (e.g., sort by X, then by Y)
#     # This guarantees particles physically next to each other are adjacent in memory
#     pos = pg.core.positions
#     p = sortperm(view(pos, 1:N), by = p -> D == 1 ? p[1] : (p[1], p[2]))
    
#     # Check if already mostly sorted to save time
#     if issorted(p); return nothing; end
    
#     # 2. Native Julia In-Place Permutations
#     Base.permute!(pg.core.positions, p)
#     Base.permute!(pg.core.is_boundary, p)
#     Base.permute!(pg.rhos, p)          
#     Base.permute!(pg.curvatures, p)    
#     Base.permute!(pg.mood_events, p)   
    
#     return nothing
# end
# =========================================================================
# MORTON Z-ORDER CURVE GENERATORS (For 2D and 3D Spatial Hashing)
# =========================================================================

# Expands a 16-bit integer by inserting a 0 bit after every bit
@inline function expand_bits_2D(w::UInt32)
    w &= 0x0000ffff
    w = (w | (w << 8)) & 0x00FF00FF
    w = (w | (w << 4)) & 0x0F0F0F0F
    w = (w | (w << 2)) & 0x33333333
    w = (w | (w << 1)) & 0x55555555
    return w
end

@inline morton_2D(x::UInt32, y::UInt32) = expand_bits_2D(x) | (expand_bits_2D(y) << 1)

# Expands a 10-bit integer by inserting two 0 bits after every bit
@inline function expand_bits_3D(w::UInt32)
    w &= 0x000003ff
    w = (w | (w << 16)) & 0xFF0000FF
    w = (w | (w <<  8)) & 0x0300F00F
    w = (w | (w <<  4)) & 0x030C30C3
    w = (w | (w <<  2)) & 0x09249249
    return w
end

@inline morton_3D(x::UInt32, y::UInt32, z::UInt32) = expand_bits_3D(x) | (expand_bits_3D(y) << 1) | (expand_bits_3D(z) << 2)


# =========================================================================
# PROXIMITY-OPTIMIZED SPATIAL REORDERING
# =========================================================================

function (rd::ReorderData{D})(pg::ParticleGrid{D, M, WF}) where {D, M, WF}
    return
    N = pg.meta.N
    if N <= 1; return nothing; end
    
    pos = pg.core.positions
    p = rd.permutation # Alias the existing pre-allocated buffer
    
    # 1. Update permutation buffer to current range
    for i in 1:N
        p[i] = i
    end
    
    # 2. Extract domain boundaries for normalization
    mins = pg.meta.mins
    extents = pg.meta.maxs .- mins
    
    # 3. Sort using the Morton Curve
    if D == 1
        # 1D is naturally perfectly local
        sort!(view(p, 1:N), by = i -> pos[i][1], alg=QuickSort)
        
    elseif D == 2
        sort!(view(p, 1:N), by = i -> begin
            # Normalize to 16-bit integers
            nx = UInt32(clamp(floor(((pos[i][1] - mins[1]) / extents[1]) * 65535.0), 0, 65535))
            ny = UInt32(clamp(floor(((pos[i][2] - mins[2]) / extents[2]) * 65535.0), 0, 65535))
            morton_2D(nx, ny)
        end, alg=QuickSort)
        
    else # D == 3
        sort!(view(p, 1:N), by = i -> begin
            # Normalize to 10-bit integers
            nx = UInt32(clamp(floor(((pos[i][1] - mins[1]) / extents[1]) * 1023.0), 0, 1023))
            ny = UInt32(clamp(floor(((pos[i][2] - mins[2]) / extents[2]) * 1023.0), 0, 1023))
            nz = UInt32(clamp(floor(((pos[i][3] - mins[3]) / extents[3]) * 1023.0), 0, 1023))
            morton_3D(nx, ny, nz)
        end, alg=QuickSort)
    end
    
    # Check if already mostly sorted to prevent unnecessary memory writes
    if issorted(view(p, 1:N)); return nothing; end
    
    # 4. Native Julia In-Place Permutations
    Base.permute!(pg.core.positions, p)
    Base.permute!(pg.core.is_boundary, p)
    Base.permute!(pg.rhos, p)          
    Base.permute!(pg.curvatures, p)    
    Base.permute!(pg.mood_events, p)   
    
    # 5. Optional: Update volumes if they exist
    if length(pg.core.volumes) >= N
        Base.permute!(pg.core.volumes, p)
    end
    
    return nothing
end
updateNeighbors!(pg::ParticleGrid) = pg.neighbor(pg)

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
    
    # --- ADD THIS LINE HERE TOO! ---
    update_bin_neighbors!(bins)
    
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
    # --- ADD THIS LINE! ---
    update_bin_neighbors!(bins)
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
# PERIODIC BIN NEIGHBOR UPDATE
# =========================================================================
function update_bin_neighbors!(bins::GlobalBins{D, T, :periodic, N_OFF}) where {D, T, N_OFF}
    coarse_dims = bins.coarse_dims
    ci = CartesianIndices(coarse_dims)
    li = LinearIndices(coarse_dims)
    window = CartesianIndices(ntuple(_ -> -1:1, Val(D)))
    
    num_bins = prod(coarse_dims)

    # Resize the outer vector if necessary
    if length(bins.bin_neighbors) != num_bins
        resize!(bins.bin_neighbors, num_bins)
    end

    for b in 1:num_bins
        cart_idx = ci[b]
        
        bins.bin_neighbors[b] = SVector{N_OFF, Int}(ntuple(Val(N_OFF)) do idx
            offset = window[idx]
            nb_cart = cart_idx + offset
            
            wrapped_cart = map((nc, cd) -> mod1(nc, cd), Tuple(nb_cart), coarse_dims)
            return li[CartesianIndex(wrapped_cart)]
        end)
    end
    
    return nothing
end

# =========================================================================
# NON-PERIODIC BIN NEIGHBOR UPDATE (With Dummy Sink Bin)
# =========================================================================
function update_bin_neighbors!(bins::GlobalBins{D, T, BC, N_OFF}) where {D, T, BC, N_OFF}
    coarse_dims = bins.coarse_dims
    ci = CartesianIndices(coarse_dims)
    li = LinearIndices(coarse_dims)
    window = CartesianIndices(ntuple(_ -> -1:1, Val(D)))
    
    num_bins = prod(coarse_dims)
    dummy_bin = num_bins + 1 # The safe sink bin for out-of-bounds offsets

    # Resize vectors to include the dummy bin
    if length(bins.bin_neighbors) != num_bins
        resize!(bins.bin_neighbors, num_bins)
        if length(bins.head) < dummy_bin
            resize!(bins.head, dummy_bin)
        end
    end

    for b in 1:num_bins
        cart_idx = ci[b]
        
        bins.bin_neighbors[b] = SVector{N_OFF, Int}(ntuple(Val(N_OFF)) do idx
            offset = window[idx]
            nb_cart = cart_idx + offset
            
            if checkbounds(Bool, li, nb_cart)
                return li[nb_cart]
            else
                return dummy_bin # Out-of-bounds points to the empty sink!
            end
        end)
    end
    
    return nothing
end
# =========================================================================
# UNIFIED NEIGHBOR SEARCH (Branchless & Type-Stable)
# =========================================================================
function (nd::NeighborData{D, WF})(pg::ParticleGrid{D, M, WF, GM, BC, N_OFF}) where {D, M, WF, GM, BC, N_OFF}
    build_global_bins!(pg) # (Ensures update_bin_neighbors! and dummy bin are configured)

    N = pg.meta.N
    pos = get_positions(pg)
    R_sq = pg.meta.R^2
    weightFunc = nd.weight_func
    
    bins = pg.bins
    L = pg.meta.L
    L_inv = pg.meta.L_inv
    coarse_dims = bins.coarse_dims
    
    bin_neighbors = bins.bin_neighbors
    bc_val = Val(BC) # Extracted for distance dispatch

    # --- PASS 1: COUNTING ---
    counts = nd.counts
    fill!(counts, 0)
    
    @batch for i in 1:N
        bin_idx = get_flat_bin_index(pos[i], bins.mins, bins.coarse_size, coarse_dims)
        
        c = 0
        @inbounds for nb_bin_idx in bin_neighbors[bin_idx]
            j = bins.head[nb_bin_idx]
            while j > 0
                if i != j
                    # Dispatch dynamically routes to periodic or euclidean natively
                    dist = _get_dist(pos, i, j, L, L_inv, bc_val)
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
        
        @inbounds for nb_bin_idx in bin_neighbors[bin_idx]
            j = bins.head[nb_bin_idx]
            while j > 0
                if i != j
                    dist = _get_dist(pos, i, j, L, L_inv, bc_val)
                    d2 = sum(abs2, dist)
                    
                    if d2 <= R_sq
                        write_idx = starts[i] + offsets[i]
                        offsets[i] += 1
                        
                        indices[write_idx]   = j
                        weights[write_idx]   = weightFunc(d2) 
                        distances[write_idx] = dist
                    end
                end
                j = bins.next[j]
            end
        end
    end
    return nothing
end

reorder_particles!(pg::ParticleGrid) = pg.reorder(pg)

function apply_boundary_conditions!(pg::ParticleGrid{D, M, T}, rhos_buffer::AbstractVector{State{M, T}}) where {D, M, T}
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
    

# 3. Outflow (Zero-Divergence): Adopt the state of the closest interior neighbor
    if bc == :outflow
        dist_vec = get_distances(pg)
        
        # Use int_buffer to track the "generation" of the update to prevent directional bias
        # 0 = unresolved. 
        status = pg.shared.int_buffer 
        fill!(status, 0)
        
        # Generation 1: All interior particles are valid initial donors
        @inbounds for i in 1:pg.meta.N
            if !pg.core.is_boundary[i]
                status[i] = 1 
            end
        end
        
        # Symmetrically propagate the boundary condition outwards
        for pass in 1:5 # 5 passes is enough to clear thick ghost layers
            all_resolved = true
            
            @inbounds for i in 1:pg.meta.N
                if status[i] == 0
                    nb_slice = pg.neighbor.ranges[i]
                    closest_j = -1
                    min_dist_sq = Inf
                    
                    for k in nb_slice
                        j = pg.neighbor.indices[k]
                        
                        # A particle can ONLY copy from a donor resolved in a PREVIOUS pass!
                        # This prevents 1-to-N loop indexing from creating directional bias.
                        if status[j] > 0 && status[j] <= pass
                            d2 = sum(abs2, dist_vec[k])
                            if d2 < min_dist_sq
                                min_dist_sq = d2
                                closest_j = j
                            end
                        end
                    end
                    
                    if closest_j != -1
                        rhos_buffer[i] = rhos_buffer[closest_j]
                        # Mark as resolved for the NEXT generation
                        status[i] = pass + 1 
                    else
                        all_resolved = false
                    end
                end
            end
            
            if all_resolved; break; end
        end
        
        # Ultimate fallback for completely orphaned particles (safety net)
        @inbounds for i in 1:pg.meta.N
            if status[i] == 0
                rhos_buffer[i] = pg.rhos[i]
            end
        end
    end
    
    return nothing
end

determineVolumes!(pg) = return

function determineVolumes!(pg::ParticleGrid{1, M, T, WF, GM, BC}) where {M, WF, GM, BC, T}
    N = pg.meta.N
    if N == 0; return; end
    
    positions = get_positions(pg)
    volumes = pg.core.volumes
    
    if BC == :periodic
        L = pg.meta.L
        L_inv = pg.meta.L_inv
        for i in 1:N
            prev_idx = mod1(i - 1, N)
            next_idx = mod1(i + 1, N)
            # Use the generalized periodic distance function
            deltaPosL = abs(getPeriodicDistance(positions, prev_idx, i, L, L_inv)[1])
            deltaPosR = abs(getPeriodicDistance(positions, i, next_idx, L, L_inv)[1])
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
# =========================================================================
# EXACT GEOMETRIC CFL CALCULATION FOR HIGH-ORDER MLS
# =========================================================================

@inline _get_Lambda(eq, U_i, ::Val{D}) where {D} = Space{D}(ntuple(d -> max_eigenvalue(eq, U_i, d), Val(D)))

@inline _extract_order(::MUSCL{D, M, B_LEN, MAX_ORDER}) where {D, M, B_LEN, MAX_ORDER} = MAX_ORDER
@inline _extract_order(g::UpwindDivergence) = g.order
@inline _extract_order(g::CentralDivergence) = g.order
@inline _extract_order(g::WENO) = g.order
@inline _extract_order(::Any) = 2 # Fallback to Linear

@generated function _compute_cfl_coeffs(::Val{D}, ::Val{IO}, ::Val{B_LEN}, nb_slice, dist_vec, w_vec, invL, ::Type{T}) where {D, IO, B_LEN, T}
    quote
        N_s = zero(SMatrix{B_LEN, B_LEN, T, B_LEN * B_LEN})
        
        @inbounds for idx in 1:length(nb_slice)
            k = nb_slice[idx]
            p_s = build_basis(Val(IO), dist_vec[k] * invL)
            N_s += w_vec[k] * (p_s * p_s')
        end
        
        if abs(det(N_s)) < T(1e-14)
            return zero(SVector{D, T})
        end
        
        inv_N_s = inv(N_s)
        sum_c_local = zero(MVector{D, T})
        
        @inbounds for idx in 1:length(nb_slice)
            k = nb_slice[idx]
            w = w_vec[k]
            p_s = build_basis(Val(IO), dist_vec[k] * invL)
            
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

@generated function dispatch_cfl(order::Int, ::Val{D}, nb_slice, dist_vec, w_vec, invL, ::Type{T}) where {D, T}
    expr = :(zero(SVector{D, T}))
    for o in 5:-1:2
        IO = o - 1
        B_LEN = typeof(basis_length(Val(D), Val(IO))).parameters[1]
        call = :(_compute_cfl_coeffs(Val($D), Val($IO), Val($B_LEN), nb_slice, dist_vec, w_vec, invL, T))
        expr = :(order == $o ? $call : $expr)
    end
    B_LEN_1 = typeof(basis_length(Val(D), Val(1))).parameters[1]
    call_fallback = :(_compute_cfl_coeffs(Val($D), Val(1), Val($B_LEN_1), nb_slice, dist_vec, w_vec, invL, T))
    expr = :(order <= 2 ? $call_fallback : $expr)
    return expr
end

@inline function getTimeStep(pg::ParticleGrid{D, M, T}, eq::HyperbolicPDE, main_grad) where {D, M, T}
    dt_buffer = pg.shared.float_buffer
    fill!(dt_buffer, T(Inf))

    w_vec = get_weights(pg)
    dist_vec = get_distances(pg)

    valD = Val(D)
    dim = D 
    invL = one(T) / minimum(pg.meta.dx)
    
    p_order = _extract_order(main_grad)

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

        sum_c_vec = dispatch_cfl(p_order, valD, nb_slice, dist_vec, w_vec, invL, T)
        sum_c = dot(Lambda, sum_c_vec)

        if sum_c > T(1e-14)
            dt_buffer[i] = one(T) / (dim * sum_c)
        end
    end
    
    return minimum(view(dt_buffer, 1:pg.meta.N))
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

@inline function ensure_capacity!(sb::SharedBuffers{D, M, T}, req_capacity::Int) where {D, M, T}
    if length(sb.pos_buffer) < req_capacity
        new_cap = ceil(Int, req_capacity * 1.25)
        
        resize!(sb.pos_buffer, new_cap)
        resize!(sb.bit_buffer, new_cap)
        resize!(sb.float_buffer, new_cap)
        resize!(sb.int_buffer, new_cap)
        resize!(sb.rho_buffer, new_cap)
    end
    return nothing
end

@inline function ensure_capacity!(pg::ParticleGrid, req_capacity::Int)
    if length(pg.rhos) < req_capacity
        new_cap = ceil(Int, req_capacity * 1.25)
        
        resize!(pg.rhos, new_cap)
        resize!(pg.mood_events, new_cap)
        resize!(pg.curvatures, new_cap)
    end
    
    ensure_capacity!(pg.core, req_capacity)
    ensure_capacity!(pg.shared, req_capacity)
    ensure_capacity!(pg.reorder, req_capacity)
    ensure_capacity!(pg.neighbor, req_capacity)
    ensure_capacity!(pg.bins, req_capacity)
    
    return nothing
end