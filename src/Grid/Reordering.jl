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

"""
    morton_2D(x::UInt32, y::UInt32)
    morton_3D(x::UInt32, y::UInt32, z::UInt32)

Generates Morton Z-order curve indices by interleaving the bits of spatial coordinates.
- The 2D variant expands 16-bit integers by inserting a zero after every bit.
- The 3D variant expands 10-bit integers by inserting two zeros after every bit.
"""
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
"""
    (rd::ReorderData)(pg::ParticleGrid)

Executes a proximity-optimized spatial reordering of the entire particle grid to drastically improve CPU cache locality and memory access patterns.

# Details
- Normalizes the domain coordinates and sorts the permutation buffer using Morton Z-order curves for 2D/3D grids, or simple lexicographical sorting for 1D grids.
- Bypasses the memory mutation entirely if the permutation buffer is already sorted.
- Permutes the active range `1:N` of the positions, boundary flags, tags, orders, state vectors, curvatures and volumes
  (the buffers may be longer than `N` after growth), and bumps `pg.meta.geometry_version`.
- Host only. The neighbour lists are stale afterwards: run `pg.neighbor(pg)` before the next step. Per-particle data
  outside the grid, such as a stepper's `mood_flags`, keep the old order until they are next written.
- 3-D keys have 10 bits per axis: beyond 1024 particles per axis, particles in the same key cell keep their order.
"""
function (rd::ReorderData{D})(pg::ParticleGrid{D}) where {D}
    N = pg.meta.N
    if N <= 1; return nothing; end
    
    pos = pg.core.positions
    ensure_capacity!(rd, N)
    p = rd.permutation # Alias the existing pre-allocated buffer
    
    # 1. Update permutation buffer to current range
    for i in 1:N
        p[i] = i
    end
    
    # 2. Extract domain boundaries from the bins (removed from meta)
    mins = pg.bins.mins
    extents = pg.bins.maxs .- mins
    # Integer key in 0:top of coordinate `x` on axis `d` (scaled in the grid's float type)
    key(x, d, top) = UInt32(clamp(floor((x - mins[d]) / extents[d] * oftype(x, top)), 0, top))

    # 3. Sort using the Morton Curve
    if D == 1
        # 1D is naturally perfectly local
        sort!(view(p, 1:N), by = i -> pos[i][1], alg=QuickSort)

    elseif D == 2
        sort!(view(p, 1:N), by = i -> morton_2D(key(pos[i][1], 1, 65535), key(pos[i][2], 2, 65535)), alg=QuickSort)

    else # D == 3
        sort!(view(p, 1:N), by = i -> morton_3D(key(pos[i][1], 1, 1023), key(pos[i][2], 2, 1023), key(pos[i][3], 3, 1023)),
              alg=QuickSort)
    end
    
    # Check if already mostly sorted to prevent unnecessary memory writes
    if issorted(view(p, 1:N)); return nothing; end
    
    # 4. Permute the active range of every per-particle array. The buffers may be longer than N (grown by the
    #    adaptivity, or the permutation longer than the grid after `reserve!`), so only `1:N` takes part.
    perm = view(p, 1:N)
    for x in (pg.core.positions, pg.core.is_boundary, pg.core.tags, pg.core.particle_orders, pg.rhos, pg.curvatures,
              pg.core.volumes)
        @assert length(x) >= N "per-particle buffer shorter than the particle count"
        v = view(parent(x), 1:N)
        v .= v[perm]
    end
    pg.meta.geometry_version += 1   # the neighbour lists and every cache of the old order are stale

    return nothing
end