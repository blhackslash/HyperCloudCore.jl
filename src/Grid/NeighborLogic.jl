
@inline get_n_offsets(::Val{1}) = 3
@inline get_n_offsets(::Val{2}) = 9
@inline get_n_offsets(::Val{3}) = 27

function GlobalBins(
    ::Type{T}, domain_mins::Space{D, T}, domain_maxs::Space{D, T}, 
    R::Real, max_particles::Int
) where {D, T}
    
    # We no longer need to convert to Space{D,T} since they are already passed as such
    domain_size = domain_maxs .- domain_mins

    # Bins must be at least R wide, otherwise the ±1-bin search window misses neighbours that sit
    # between `coarse_size` and `R` away from a particle at the edge of its bin. `floor` guarantees
    # coarse_size = domain_size / coarse_dims ≥ R (`ceil` gave coarse_size ≤ R).
    coarse_dims = ntuple(d -> max(1, floor(Int, domain_size[d] / R)), Val(D))
    coarse_size = domain_size ./ coarse_dims

    total_coarse_bins = prod(coarse_dims)
    head = zeros(Int, total_coarse_bins)
    next = zeros(Int, ceil(Int, max_particles * 1.25))
    
    N_OFF = get_n_offsets(Val(D))
    bin_neighbors = Vector{SVector{N_OFF, Int}}(undef, 0)

    # Note: Ensure your GlobalBins struct definition also had BC removed!
    # i.e., struct GlobalBins{D, T, N_OFF}
    return GlobalBins{D, T, N_OFF}(
        domain_mins, domain_maxs, coarse_size, coarse_dims, head, next, bin_neighbors
    )
end


# =========================================================================
# FAST SPATIAL HASHING (Coordinates -> 1D Bin Index)
# =========================================================================
"""
    get_flat_bin_index(pos, mins, bin_size, dims)

Converts 1D, 2D, or 3D physical coordinates into a flattened, 1D array index corresponding to a specific spatial coarse bin.
"""
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
# GLOBAL BIN BUILDING
# =========================================================================
"""
    build_global_bins!(pg::ParticleGrid)

Constructs the spatial hashing linked list for the current particle configuration.

# Details
- Resolves valid neighbor bins for each spatial cell, accounting for per-axis periodic wrapping.
- Iterates over all particles, inserting them into their respective spatial bins using a highly efficient `head` and `next` linked-list methodology.
"""
function build_global_bins!(pg::ParticleGrid)
    N = pg.meta.N
    pos = get_positions(pg)
    bins = pg.bins
    
    update_bin_neighbors!(bins, pg.domain) # Passes the domain for per-axis logic
    
    fill!(bins.head, 0)
    
    @inbounds for i in 1:N
        # Link particle into the coarse spatial bin
        coarse_idx = get_flat_bin_index(pos[i], bins.mins, bins.coarse_size, bins.coarse_dims)
        bins.next[i] = bins.head[coarse_idx]
        bins.head[coarse_idx] = i
    end
    return nothing
end

function update_bin_neighbors!(bins::GlobalBins{D, T, N_OFF}, domain::ComputationalDomain{D, T}) where {D, T, N_OFF}
    coarse_dims = bins.coarse_dims
    ci = CartesianIndices(coarse_dims)
    li = LinearIndices(coarse_dims)
    window = CartesianIndices(ntuple(_ -> -1:1, Val(D)))
    
    num_bins = prod(coarse_dims)
    dummy_bin = num_bins + 1 

    # Resize vectors to include the dummy bin
    if length(bins.bin_neighbors) != num_bins
        resize!(bins.bin_neighbors, num_bins)
        if length(bins.head) < dummy_bin
            resize!(bins.head, dummy_bin)
        end
    end

    is_per = domain.is_periodic

    for b in 1:num_bins
        cart_idx = ci[b]

        raw = ntuple(Val(N_OFF)) do idx
            offset = window[idx]
            nb_cart = cart_idx + offset
            
            # Resolve wrapping per axis
            valid_bin = true
            final_cart = ntuple(Val(D)) do d
                nc = nb_cart[d]
                cd = coarse_dims[d]
                
                if nc < 1 || nc > cd
                    if is_per[d]
                        return mod1(nc, cd)
                    else
                        valid_bin = false
                        return 1 # Junk value, will be caught by valid_bin
                    end
                end
                return nc
            end
            
            if valid_bin
                return li[CartesianIndex(final_cart)]
            else
                return dummy_bin
            end
        end
        # A periodic axis with fewer than three bins maps several offsets onto the same bin. Visiting
        # it more than once would list every neighbour in it several times, so repeats become the dummy bin.
        bins.bin_neighbors[b] = SVector{N_OFF, Int}(ntuple(Val(N_OFF)) do idx
            v = raw[idx]
            any(k -> raw[k] == v, 1:(idx - 1)) ? dummy_bin : v
        end)
    end

    return nothing
end

# =========================================================================
# UNIFIED NEIGHBOR SEARCH (Branchless & Type-Stable)
# =========================================================================
"""
    (nd::NeighborData)(pg::ParticleGrid)

The universal, branchless neighbor search functor managing topology updates.

# Details
- Forces physical coordinates back into the primary periodic domain limits if periodicity is active.
- Rebuilds the coarse global bins (`build_global_bins!`).
- **Pass 1 (Counting):** Scans adjacent spatial bins to count the valid neighbors falling within the interaction radius `R_sq` for each particle. Computes a sequential prefix sum to establish writing ranges and resizes the target buffers.
- **Pass 2 (Writing):** Rescans the valid particles, accurately writing their indices, computed weights, and computed distances directly into the pre-allocated neighbor buffers.
"""
function (nd::NeighborData{D, T, WF})(pg::ParticleGrid{D, M, T, WF, GM, N_OFF, Dom}) where {D, M, T, WF, GM, N_OFF, Dom}
    
    backend = get_backend()
    N = pg.meta.N

    # --- ENFORCE PER-AXIS PERIODIC WRAPPING ---
    foreach_particle(_wrap_position!, backend, N, (pg.core.positions, pg.domain.is_periodic, pg.domain.L, pg.domain.canvas_mins))

    build_global_bins!(pg)

    pos  = pg.core.positions
    bins = pg.bins
    R_sq = pg.meta.R^2
    L, L_inv = pg.domain.L_wrap, pg.domain.invL_wrap

    # --- PASS 1: COUNTING ---
    counts = nd.counts
    foreach_particle(_count_neighbours!, backend, N,
        (counts, pos, bins.head, bins.next, bins.bin_neighbors, bins.mins, bins.coarse_size, bins.coarse_dims, L, L_inv, R_sq))

    # --- SEQUENTIAL PREFIX SUM (host) ---
    starts = pg.shared.int_buffer
    max_so_far = 0
    current_ptr = 1
    @inbounds for i in 1:N
        c = counts[i]
        nd.ranges[i] = current_ptr:(current_ptr + c - 1)
        starts[i] = current_ptr
        current_ptr += c
        if c > max_so_far; max_so_far = c; end
    end
    pg.meta.max_nb = max_so_far
    nd.ranges[N + 1] = current_ptr:(current_ptr - 1)

    total_neighbors = current_ptr - 1
    ensure_capacity!(nd, total_neighbors)

    # --- PASS 2: WRITING ---
    offsets = nd.offsets
    fill!(offsets, 0)
    foreach_particle(_write_neighbours!, backend, N,
        (pos, bins.head, bins.next, bins.bin_neighbors, bins.mins, bins.coarse_size, bins.coarse_dims, L, L_inv, R_sq,
         starts, offsets, nd.indices, nd.weights, nd.distances, nd.weight_func))

    pg.meta.geometry_version += 1   # neighbour lists, distances and weights changed
    return nothing
end
# =========================================================================
# KERNEL BODIES OF THE NEIGHBOUR SEARCH (see Backend.jl)
# =========================================================================
@inline function _wrap_position!(i, pos::AbstractVector{Space{D, T}}, is_per, L_vec, mins_vec) where {D, T}
    @inbounds p_i = pos[i]
    @inbounds pos[i] = Space{D, T}(ntuple(Val(D)) do d
        p = p_i[d]
        is_per[d] ? mins_vec[d] + mod(p - mins_vec[d], L_vec[d]) : p
    end)
    return nothing
end

@inline function _count_neighbours!(i, counts, pos, head, next, bin_neighbors, mins, coarse_size, coarse_dims, L, L_inv, R_sq)
    bin_idx = get_flat_bin_index(pos[i], mins, coarse_size, coarse_dims)
    c = 0
    @inbounds for nb_bin_idx in bin_neighbors[bin_idx]
        j = head[nb_bin_idx]
        while j > 0
            if i != j
                dist = get_distance(pos, i, j, L, L_inv)
                sum(abs2, dist) <= R_sq && (c += 1)
            end
            j = next[j]
        end
    end
    counts[i] = c
    return nothing
end

@inline function _write_neighbours!(i, pos, head, next, bin_neighbors, mins, coarse_size, coarse_dims, L, L_inv, R_sq,
                                    starts, offsets, indices, weights, distances, weight_func)
    bin_idx = get_flat_bin_index(pos[i], mins, coarse_size, coarse_dims)
    @inbounds for nb_bin_idx in bin_neighbors[bin_idx]
        j = head[nb_bin_idx]
        while j > 0
            if i != j
                dist = get_distance(pos, i, j, L, L_inv)
                d2 = sum(abs2, dist)
                if d2 <= R_sq
                    write_idx = starts[i] + offsets[i]
                    offsets[i] += 1
                    indices[write_idx]   = j
                    weights[write_idx]   = weight_func(d2)
                    distances[write_idx] = dist
                end
            end
            j = next[j]
        end
    end
    return nothing
end
