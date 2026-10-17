
@inline get_n_offsets(::Val{1}) = 3
@inline get_n_offsets(::Val{2}) = 9
@inline get_n_offsets(::Val{3}) = 27

"""
    GlobalBins(T, domain_mins, domain_maxs, R, max_particles, is_periodic)

The coarse cell list of the neighbour search, stored as a counting sort (every pass of its
construction is a kernel; see `build_global_bins!`):

- `cell_particles`: particle indices sorted by bin, in descending index order within a bin;
- `cell_start` (`num_bins + 2` entries): the particles of bin `b` are
  `cell_particles[cell_start[b]:cell_start[b + 1] - 1]`; bin `num_bins + 1` is the always-empty dummy
  bin that `bin_neighbors` points to for out-of-domain and repeated offsets;
- `cell_count`: particles per bin (also the scatter cursor while building);
- `bin_of`: the bin of every particle;
- `bin_neighbors`: the `3^D` bins around every bin, fixed at construction (periodic axes wrap).

Bins are at least `R` wide, so the ±1-bin window around a particle's bin contains all its neighbours.
"""
function GlobalBins(::Type{T}, domain_mins::Space{D, T}, domain_maxs::Space{D, T}, R::Real, max_particles::Int,
                    is_periodic::SVector{D, Bool}) where {D, T}
    domain_size = domain_maxs .- domain_mins

    # `floor` guarantees coarse_size = domain_size / coarse_dims ≥ R (`ceil` gave coarse_size ≤ R, which
    # misses neighbours between `coarse_size` and `R` away from a particle at the edge of its bin).
    coarse_dims = ntuple(d -> max(1, floor(Int, domain_size[d] / R)), Val(D))
    coarse_size = domain_size ./ coarse_dims
    num_bins    = prod(coarse_dims)
    cap         = ceil(Int, max_particles * 1.25)

    N_OFF = get_n_offsets(Val(D))
    bin_neighbors = Vector{SVector{N_OFF, Int}}(undef, num_bins)
    _fill_bin_neighbors!(bin_neighbors, coarse_dims, is_periodic)

    return GlobalBins{D, T, N_OFF}(domain_mins, domain_maxs, coarse_size, coarse_dims,
                                   fill(num_bins + 1, num_bins + 2), zeros(Int, num_bins + 1),
                                   Growable(zeros(Int, cap)), Growable(zeros(Int, cap)), bin_neighbors)
end

# The 3^D window around every bin, with per-axis periodic wrapping (host, once per grid).
function _fill_bin_neighbors!(bin_neighbors::Vector{SVector{N_OFF, Int}}, coarse_dims::NTuple{D, Int},
                              is_per::SVector{D, Bool}) where {N_OFF, D}
    ci = CartesianIndices(coarse_dims)
    li = LinearIndices(coarse_dims)
    window = CartesianIndices(ntuple(_ -> -1:1, Val(D)))
    num_bins  = prod(coarse_dims)
    dummy_bin = num_bins + 1

    for b in 1:num_bins
        cart_idx = ci[b]
        raw = ntuple(Val(N_OFF)) do idx
            nb_cart = cart_idx + window[idx]
            valid_bin = true
            final_cart = ntuple(Val(D)) do d
                nc = nb_cart[d]
                cd = coarse_dims[d]
                if nc < 1 || nc > cd
                    is_per[d] && return mod1(nc, cd)
                    valid_bin = false
                    return 1                      # junk, caught by valid_bin
                end
                return nc
            end
            valid_bin ? li[CartesianIndex(final_cart)] : dummy_bin
        end
        # A periodic axis with fewer than three bins maps several offsets onto the same bin. Visiting
        # it more than once would list every neighbour in it several times, so repeats become the dummy bin.
        bin_neighbors[b] = SVector{N_OFF, Int}(ntuple(Val(N_OFF)) do idx
            v = raw[idx]
            any(k -> raw[k] == v, 1:(idx - 1)) ? dummy_bin : v
        end)
    end
    return nothing
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
# GLOBAL BIN BUILDING (counting sort)
# =========================================================================
"""
    build_global_bins!(pg::ParticleGrid)

Rebuilds the cell list for the current positions: bin index and per-bin histogram (atomic adds), a
prefix sum over the bins, an atomic scatter into `cell_particles`, and an insertion sort of every
bin's (small) segment by descending particle index. The sort makes the lists independent of the order
in which the atomics landed, so the neighbour lists, and everything computed from them, are the same
on every backend and in every run.
"""
function build_global_bins!(pg::ParticleGrid)
    N       = pg.meta.N
    bins    = pg.bins
    backend = get_backend()
    nb      = prod(bins.coarse_dims)
    cs, cc, cp, bo = bins.cell_start, bins.cell_count, bins.cell_particles, bins.bin_of

    fill!(cc, 0)
    foreach_particle(_bin_count!, backend, N, (bo, cc, pg.core.positions, bins.mins, bins.coarse_size, bins.coarse_dims))

    # Segment starts: exclusive prefix sum of the counts, 1-based; the dummy bin nb + 1 is empty.
    fill!(view(cs, 1:1), 0)
    cumsum!(view(cs, 2:nb + 1), view(cc, 1:nb))
    view(cs, 1:nb + 1) .+= 1
    fill!(view(cs, nb + 2:nb + 2), N + 1)

    fill!(view(cc, 1:nb), 0)                              # reused as the scatter cursor; ends up as the count again
    foreach_particle(_bin_scatter!, backend, N, (cp, bo, cs, cc))
    foreach_particle(_bin_sort!, backend, nb, (cp, cs))
    return nothing
end

# Kernel bodies of the cell list (see Backend.jl).
@inline function _bin_count!(i, bin_of, cell_count, pos, mins, coarse_size, coarse_dims)
    b = get_flat_bin_index(pos[i], mins, coarse_size, coarse_dims)
    bin_of[i] = b
    KA.@atomic cell_count[b] += 1
    return nothing
end

@inline function _bin_scatter!(i, cell_particles, bin_of, cell_start, cursor)
    b = bin_of[i]
    k = KA.@atomic cursor[b] += 1
    cell_particles[cell_start[b] + k - 1] = i
    return nothing
end

# One bin per work item: insertion sort of its segment, descending (the order the former linked list produced).
@inline function _bin_sort!(b, cell_particles, cell_start)
    lo = cell_start[b]
    hi = cell_start[b + 1] - 1
    @inbounds for s in (lo + 1):hi
        v = cell_particles[s]
        t = s - 1
        while t >= lo && cell_particles[t] < v
            cell_particles[t + 1] = cell_particles[t]
            t -= 1
        end
        cell_particles[t + 1] = v
    end
    return nothing
end

# =========================================================================
# SLOT LAYOUTS
#
# Row-major (`UnitRange{Int}` slot ranges): the slots of a particle are contiguous, the layout the
# host likes. Slot-major (`StepRange{Int, Int}` with stride N): slot k of all particles is contiguous,
# so the threads of a wavefront, which handle consecutive particles, read consecutive addresses; the
# layout a device likes (it pads every particle to `max_nb` slots). The element type of `ranges`
# selects the layout; `device_grid` converts between them.
# =========================================================================
const RowMajorSlots  = UnitRange{Int}
const SlotMajorSlots = StepRange{Int, Int}

@inline _slot_range(::Type{RowMajorSlots}, i, c, incl, N)  = (incl - c + 1):incl
@inline _slot_range(::Type{SlotMajorSlots}, i, c, incl, N) = i:N:(i + (c - 1) * N)
_slot_capacity(::Type{RowMajorSlots}, total, max_nb, N)  = total
_slot_capacity(::Type{SlotMajorSlots}, total, max_nb, N) = max_nb * N
_empty_slots(::Type{RowMajorSlots})  = 1:0
_empty_slots(::Type{SlotMajorSlots}) = 1:1:0
_slot_sentinel(::Type{RowMajorSlots}, cap, N)  = (cap + 1):cap              # `first - 1` is the number of slots in use
_slot_sentinel(::Type{SlotMajorSlots}, cap, N) = (cap + 1):max(N, 1):cap

"""
    with_slot_layout(nd::NeighborData, N, R) -> NeighborData

The neighbour data of `N` particles in the layout `R` (`RowMajorSlots` or `SlotMajorSlots`), the same
object if it already has it. Host arrays only.
"""
function with_slot_layout(nd::NeighborData{D, T, WF}, N::Int, ::Type{R}) where {D, T, WF, R}
    eltype(nd.ranges) === R && return nd
    old    = nd.ranges
    counts = [length(old[i]) for i in 1:N]
    max_nb = maximum(counts; init = 0)
    cap    = _slot_capacity(R, sum(counts), max_nb, N)
    incl   = cumsum(counts)
    ranges = fill(_empty_slots(R), length(old))
    for i in 1:N; ranges[i] = _slot_range(R, i, counts[i], incl[i], N); end
    ranges[N + 1] = _slot_sentinel(R, cap, N)
    indices = Vector{Int}(undef, cap); weights = Vector{T}(undef, cap); distances = Vector{Space{D, T}}(undef, cap)
    for i in 1:N
        for (k_new, k_old) in zip(ranges[i], old[i])
            indices[k_new]   = nd.indices[k_old]
            weights[k_new]   = nd.weights[k_old]
            distances[k_new] = nd.distances[k_old]
        end
    end
    return NeighborData{D, T, WF}(nd.weight_func, Growable(ranges), Growable(indices), Growable(weights), Growable(distances),
                                  nd.counts, nd.offsets)
end

# =========================================================================
# UNIFIED NEIGHBOR SEARCH (Branchless & Type-Stable)
# =========================================================================
"""
    (nd::NeighborData)(pg::ParticleGrid)

The universal, branchless neighbor search functor managing topology updates. Every pass is a kernel
or a device reduction, so it runs on a grid built with `device_grid` as well.

# Details
- Forces physical coordinates back into the primary periodic domain limits if periodicity is active.
- Rebuilds the cell list (`build_global_bins!`).
- **Pass 1 (Counting):** Scans the adjacent bins to count the neighbours within the interaction radius
  `R_sq` of each particle. A prefix sum of the counts gives the neighbour slots (`ranges`, in the
  layout of the grid), and the slot arrays are grown to the capacity.
- **Pass 2 (Writing):** Rescans the bins, writing neighbour indices, weights and distances into the slots.
"""
function (nd::NeighborData{D, T})(pg::ParticleGrid{D, M, T}) where {D, M, T}
    backend = get_backend()
    N = pg.meta.N
    R = eltype(nd.ranges)                                 # the slot layout

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
        (counts, pos, bins.bin_of, bins.cell_start, bins.cell_particles, bins.bin_neighbors, L, L_inv, R_sq))

    # --- PREFIX SUM → neighbour slots ---
    incl = pg.shared.int_buffer
    cumsum!(view(incl, 1:N), view(counts, 1:N))
    total_neighbors = N == 0 ? 0 : sum(view(counts, 1:N))
    pg.meta.max_nb  = maximum(view(counts, 1:N); init = 0)
    capacity = _slot_capacity(R, total_neighbors, pg.meta.max_nb, N)
    foreach_particle(_neighbour_ranges!, backend, N, (nd.ranges, incl, counts, N))
    fill!(view(nd.ranges, (N + 1):(N + 1)), _slot_sentinel(R, capacity, N))
    ensure_capacity!(nd, capacity)

    # --- PASS 2: WRITING ---
    offsets = nd.offsets
    fill!(offsets, 0)
    foreach_particle(_write_neighbours!, backend, N,
        (pos, bins.bin_of, bins.cell_start, bins.cell_particles, bins.bin_neighbors, L, L_inv, R_sq,
         nd.ranges, offsets, nd.indices, nd.weights, nd.distances, nd.weight_func))

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

@inline function _count_neighbours!(i, counts, pos, bin_of, cell_start, cell_particles, bin_neighbors, L, L_inv, R_sq)
    c = 0
    @inbounds for b in bin_neighbors[bin_of[i]]
        for s in cell_start[b]:(cell_start[b + 1] - 1)
            j = cell_particles[s]
            if i != j
                dist = get_distance(pos, i, j, L, L_inv)
                sum(abs2, dist) <= R_sq && (c += 1)
            end
        end
    end
    counts[i] = c
    return nothing
end

# Slot range of particle `i` in the layout of `ranges`, from its count and the inclusive prefix sum of the counts.
@inline function _neighbour_ranges!(i, ranges, incl, counts, N)
    ranges[i] = _slot_range(eltype(ranges), i, counts[i], incl[i], N)
    return nothing
end

@inline function _write_neighbours!(i, pos, bin_of, cell_start, cell_particles, bin_neighbors, L, L_inv, R_sq,
                                    ranges, offsets, indices, weights, distances, weight_func)
    r = ranges[i]
    @inbounds for b in bin_neighbors[bin_of[i]]
        for s in cell_start[b]:(cell_start[b + 1] - 1)
            j = cell_particles[s]
            if i != j
                dist = get_distance(pos, i, j, L, L_inv)
                d2 = sum(abs2, dist)
                if d2 <= R_sq
                    write_idx = first(r) + offsets[i] * step(r)
                    offsets[i] += 1
                    indices[write_idx]   = j
                    weights[write_idx]   = weight_func(d2)
                    distances[write_idx] = dist
                end
            end
        end
    end
    return nothing
end
