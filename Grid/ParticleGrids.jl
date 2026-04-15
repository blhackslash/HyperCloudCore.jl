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
    weight_func = exponentialWeightFunction(1.,1.), mover = NoGridMover()
) where {D}
    
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
        meta, core, shared, neighbors, reorder, bins, mover,
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

function determineVolumes!(pg::ParticleGrid{1, M, WF, GM, BC}) where {M, WF, GM, BC}
    N = pg.meta.N
    if N == 0; return; end
    
    positions = get_positions(pg)
    volumes = pg.core.volumes
    
    if BC == :periodic
        L = pg.meta.maxs .- pg.meta.mins
        for i in 1:N
            prev_idx = mod1(i - 1, N)
            next_idx = mod1(i + 1, N)
            # Use the generalized periodic distance function
            deltaPosL = abs(getPeriodicDistance(positions, prev_idx, i, L)[1])
            deltaPosR = abs(getPeriodicDistance(positions, i, next_idx, L)[1])
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
@inline function getTimeStep(pg::ParticleGrid{D, M}, eq::HyperbolicPDE) where {D, M}
    dtMax = Inf

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

        # 1. Get the local macroscopic/kinetic state
        U_i = pg.rhos[i]

        # 2. Extract maximum absolute wave speeds dynamically for this particle!
        # Automatically dispatches to Euler, Burgers, or LinearAdvection methods.
        Lambda = Space{D}(ntuple(d -> max_eigenvalue(eq, U_i, d), Val(D)))

        # 3. Build the MLS Matrix N_s
        N_s = @SMatrix zeros(Float64, D, D)
        @inbounds for k in nb_slice
            w  = w_vec[k]
            dx = dist_vec[k] # This is already Space{D}
            N_s += w * (dx * dx')
        end

        if abs(det(N_s)) < 1e-14
            continue
        end

        # 4. Compile-time analytic inversion using StaticArrays
        inv_N_s = inv(N_s)

        # 5. Accumulate the stability condition dynamically
        sum_c = 0.0
        @inbounds for k in nb_slice
            w  = w_vec[k]
            dx = dist_vec[k]
            
            # C_k is the effective MLS shape function vector
            C_k = inv_N_s * (w * dx)
            
            # Worst-case upwind contribution using absolute maximum wave speeds
            sum_c += sum(ntuple(d -> Lambda[d] * abs(C_k[d]), Val(D)))
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