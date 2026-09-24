reorder_particles!(pg::ParticleGrid) = pg.reorder(pg)

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
- Performs fast, in-place native Julia permutations (`Base.permute!`) across the positions, boundary flags, state vectors, and volumes.
"""
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