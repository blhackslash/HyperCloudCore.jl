"""
    manage_particles!(pg::ParticleGrid1D)

Main management routine. 
1. Voxel Fill (Splitting) - Fills gaps using Linear Interpolation or delegates to neighbors.
2. Merge - Coarsens dense regions everywhere.
3. Update Boundaries - Enforces domain limits, removes outliers, and sets flags.
4. Finalize - Rebuilds graph and sorts.
"""
function manage_particles!(pg::ParticleGrid)
    # =========================================================================
    # PHASE 1: VOXEL FILL (Splitting)
    # =========================================================================
    return
    empty!(pg.split_buffer_pos)
    empty!(pg.split_buffer_rho)

    if length(pg.merged_buffer) < pg.meta.N
        safe_resize!(pg.merged_buffer, pg.meta.N)
    end
    visited = pg.merged_buffer 
    fill!(view(visited, 1:pg.meta.N), false)
    
    lv = pg.local_voxels

    for i in 1:pg.meta.N
        if !visited[i]
            reset_voxels!(lv)
            check_occupation!(lv, pg, i, visited)
            
            # Pass 'visited' so we can un-mark neighbors if needed
            fill_empty_voxels!(lv, pg, i, visited)
            
            visited[i] = true
        end
    end

    # Append New Particles
    N_new = length(pg.split_buffer_pos)
    if N_new > 0
        N_curr = pg.meta.N
        N_total = N_curr + N_new
        
        # Resize all persistent arrays
        safe_resize!(get_positions(pg), N_total)
        safe_resize!(pg.rhos, N_total)
        safe_resize!(pg.curvatures, N_total)
        safe_resize!(pg.core.is_boundary, N_total)
        safe_resize!(pg.volumes, N_total)
        safe_resize!(pg.mood_events, N_total)
        
        for k in 1:N_new
            idx = N_curr + k
            get_positions(pg)[idx]   = pg.split_buffer_pos[k]
            pg.rhos[idx]        = pg.split_buffer_rho[k]
            # Defaults
            pg.curvatures[idx]  = 0.0
            pg.core.is_boundary[idx] = false 
            pg.volumes[idx]     = 0.0
            pg.mood_events[idx] = false
        end
        
        pg.meta.N = N_total
        
        # Intermediate Rebuild needed for Merge
        safe_resize!(pg.neighbor.ranges, pg.meta.N)
        
        sort_1d_particles!(pg)
        updateNeighbors!(pg)
    end

    _merge_particles_pairwise!(pg)

    # =========================================================================
    # PHASE 3: BOUNDARY UPDATE (Cleanup)
    # =========================================================================
    
    update_boundaries!(pg)

    # =========================================================================
    # PHASE 4: FINALIZE
    # =========================================================================
    
    safe_resize!(pg.neighbor.ranges, pg.meta.N)
    if pg isa ParticleGrid1D; sort_1d_particles!(pg) end
    updateNeighbors!(pg)
    if pg isa ParticleGrid1D; determineVolumes!(pg) end
end
function _merge_particles!(pg::ParticleGrid1D)
    # =========================================================================
    # PHASE 2: MERGE (Coarsen)
    # =========================================================================
    
    safe_resize!(pg.merged_buffer, pg.meta.N)
    fill!(view(pg.merged_buffer, 1:pg.meta.N), false)
    merged = pg.merged_buffer

    write_idx = 0 
    
    pos    = get_positions(pg)
    rhos   = pg.rhos
    
    # Pre-allocate a queue to track the cluster chain
    # (Size hint assumes clusters rarely exceed 16 particles)
    queue = Int[]
    sizehint!(queue, 16)

    for i in 1:pg.meta.N
        if merged[i]; continue; end

        write_idx += 1
        
        # Accumulators
        sum_x = pos[i]
        sum_rho = rhos[i]
        count = 1.0
        
        # Initialize BFS for this cluster
        empty!(queue)
        push!(queue, i)
        
        # BFS: Process every particle added to the cluster to find ITS neighbors
        q_head = 1
        while q_head <= length(queue)
            u = queue[q_head]
            q_head += 1
            
            # Iterate neighbors of 'u' (the current link in the chain)
            for k in pg.neighbor.ranges[i]
                j = pg.neighbor.indices[k]
                
                # 1. Forward check (j > i) ensures we don't merge backwards into finished data
                # 2. !merged[j] ensures we don't double-process
                if j > i && !merged[j]
                    # Distance between 'u' and 'j'
                    dist = abs(get_xdistance(pg)[k]) 
                    
                    if dist < pg.min_dist
                        # --- MERGE ---
                        sum_x += pos[j]
                        sum_rho += rhos[j]
                        count += 1.0
                        merged[j] = true
                        
                        # Add j to queue to check *its* neighbors next
                        push!(queue, j)
                    end
                end
            end
        end
        
        # Write compacted result
        if count > 1.0
            pos[write_idx]   = sum_x / count
            rhos[write_idx]  = sum_rho / count
        else
            if i != write_idx
                pos[write_idx]  = pos[i]
                rhos[write_idx] = rhos[i]
            end
        end
    end
    
    pg.meta.N = write_idx
    return nothing
end

# For Burgers: f(u) = u^2/2
a_burgers(u, v) = (u + v) / 2.0

# For Test Case: f(u) = u^3/3
# Form derived from: [u*f'(u) - f(u)] / [f'(v) - f'(u)]
function a_cubic(u, v)
    if abs(u + v) < 1e-12
        return (u^2 + v^2) / 3.0 # Limit case for symmetry
    end
    return (2.0/3.0) * (u^2 + u*v + v^2) / (u + v)
end

function _merge_particles_conservative!(pg::ParticleGrid1D)
    safe_resize!(pg.merged_buffer, pg.meta.N)
    fill!(view(pg.merged_buffer, 1:pg.meta.N), false)
    merged = pg.merged_buffer

    write_idx = 0 
    
    pos    = get_positions(pg)
    rhos   = pg.rhos
    vols   = pg.volumes
    
    # We iterate 1 to N. Since list is sorted, neighbors are i-1 and i+1.
    for i in 1:pg.meta.N
        if merged[i]; continue; end

        write_idx += 1
        
        # Check if we should merge with the NEXT particle (i+1)
        # We handle periodicity for the 'next' index
        j = (i == pg.meta.N) ? 1 : i + 1
        
        did_merge = false
        
        # Criteria:
        # 1. j is not processed
        # 2. distance is small
        # 3. i and j are actually neighbors in the sorted list (index check)
        
        if !merged[j]
            # Calculate distance respecting periodicity
            dist_ij = getDistance(pg, i, j)
            
            if abs(dist_ij) < pg.min_dist
                
                # --- IDENTIFY 4-POINT STENCIL (1, 2, 3, 4) ---
                # P2 = i, P3 = j
                
                # Find P1 (Left of i)
                idx_1 = (i == 1) ? pg.meta.N : i - 1
                
                # Find P4 (Right of j)
                idx_4 = (j == pg.meta.N) ? 1 : j + 1
                
                # Check Validity of Stencil
                # In non-periodic, we can't do this at the very edges.
                valid_stencil = true
                if pg.bc != :periodic
                    if i == 1 || j == pg.meta.N; valid_stencil = false; end
                end
                
                if valid_stencil
                    # --- AREA PRESERVING MERGE ---
                    
                    # Gather Values
                    u1, u2 = rhos[idx_1], rhos[i]
                    u3, u4 = rhos[j], rhos[idx_4]
                    
                    # Gather Distances (always positive lengths for area calc)
                    # We use getDistance to handle periodicity, then abs()
                    d12 = abs(getDistance(pg, idx_1, i))
                    d23 = abs(dist_ij)
                    d34 = abs(getDistance(pg, j, idx_4))
                    
                    # New Midpoint Geometry
                    d1_new = d12 + 0.5 * d23  # Dist from 1 to New
                    dnew_4 = 0.5 * d23 + d34  # Dist from New to 4
                    d14    = d12 + d23 + d34  # Total span
                    
                    # 1. Calculate Old Area (Trapezoidal Rule)
                    # A = 0.5 * (uL + uR) * dx
                    area_old = 0.5 * ((u1+u2)*d12 + (u2+u3)*d23 + (u3+u4)*d34)
                    
                    # 2. Solve for u_new
                    # The formula simplifies to:
                    # u_new = (2*Area - u1*d1_new - u4*dnew_4) / d14
                    
                    u_new = (2.0 * area_old - u1 * d1_new - u4 * dnew_4) / d14
                    #u_new = clamp(u_new, min(u2, u3), max(u2, u3))
                    # 3. Update Position
                    # Conservative position update (volume weighted) is usually still best
                    # but you specifically asked for the midpoint:
                    
                    # If you want EXACT midpoint relative to neighbors:
                    # pos[write_idx] = pos[i] + 0.5 * (signed distance i->j)
                    pos[write_idx] = pos[i] + 0.5 * dist_ij
                    
                    rhos[write_idx] = u_new
                else
                    # --- FALLBACK: Volume Weighted (Edges) ---
                    v_i, v_j = vols[i], vols[j]
                    sum_v = v_i + v_j
                    rhos[write_idx] = (rhos[i]*v_i + rhos[j]*v_j) / sum_v
                    pos[write_idx]  = pos[i] + (dist_ij * v_j) / sum_v
                end
                
                # Periodic Wrap for Position
                if pg.bc == :periodic
                    L = pg.xmax - pg.xmin
                    if pos[write_idx] > pg.xmax; pos[write_idx] -= L; end
                    if pos[write_idx] < pg.xmin; pos[write_idx] += L; end
                end

                merged[j] = true
                did_merge = true
            end
        end
        
        if !did_merge
            if i != write_idx
                pos[write_idx]  = pos[i]
                rhos[write_idx] = rhos[i]
            end
        end
    end
    
    pg.meta.N = write_idx
    return
end

function _merge_particles_flux_conserving!(pg::ParticleGrid1D)
# =========================================================================
    # PHASE 2: MERGE (Coarsen) - ALE CONSISTENT CONSERVATION
    # =========================================================================
    
    safe_resize!(pg.merged_buffer, pg.meta.N)
    fill!(view(pg.merged_buffer, 1:pg.meta.N), false)
    merged = pg.merged_buffer

    write_idx = 0 
    pos  = get_positions(pg)
    rhos = pg.rhos
    vols = pg.volumes
    
    for i in 1:pg.meta.N
        if merged[i]; continue; end

        write_idx += 1
        j = (i == pg.meta.N) ? 1 : i + 1 # Sorted neighbor
        
        did_merge = false
        if !merged[j]
            dist_ij = getDistance(pg, i, j)
            
            if abs(dist_ij) < pg.min_dist
                # --- Neighbors of the merging pair ---
                idx_L = (i == 1) ? pg.meta.N : i - 1
                idx_R = (j == pg.meta.N) ? 1 : j + 1
                
                # 1. New Position: Arithmetic midpoint (or volume weighted)
                new_pos = pos[i] + 0.5 * dist_ij
                if pg.bc == :periodic
                    L_dom = pg.xmax - pg.xmin
                    if new_pos > pg.xmax; new_pos -= L_dom; end
                    if new_pos < pg.xmin; new_pos += L_dom; end
                end

                # 2. Conservation Logic:
                # We need to preserve Total Mass = sum(u_k * V_k)
                # Before merge: m_old = u_L*V_L + u_i*V_i + u_j*V_j + u_R*V_R
                # After merge:  m_new = u_L*V_L_new + u_new*V_new + u_R*V_R_new
                
                # Calculate old masses of the affected 4-particle stencil
                m_stencil_old = (rhos[idx_L]*vols[idx_L] + 
                                 rhos[i]*vols[i] + 
                                 rhos[j]*vols[j] + 
                                 rhos[idx_R]*vols[idx_R])

                # 3. Calculate New Volumes for neighbors and the merged particle
                # V_k = (x_{k+1} - x_{k-1}) / 2
                
                # New Midpoints
                mid_L_new = getDistance(pg, idx_L, i) * 0.5 + pos[idx_L] # Approximation
                # Better: use the actual formula for V_k in your determineVolumes!
                # V_i = 0.5 * (pos[i+1] - pos[i-1])
                
                v_L_new = abs(getDistance(pg, (idx_L==1 ? pg.meta.N : idx_L-1), write_idx)) * 0.5
                v_R_new = abs(getDistance(pg, write_idx, (idx_R==pg.meta.N ? 1 : idx_R+1))) * 0.5
                v_new   = abs(getDistance(pg, idx_L, idx_R)) * 0.5
                
                # 4. Determine u_new to satisfy conservation
                # u_new = (m_stencil_old - u_L*v_L_new - u_R*v_R_new) / v_new
                u_new = (m_stencil_old - rhos[idx_L]*v_L_new - rhos[idx_R]*v_R_new) / v_new

                pos[write_idx] = new_pos
                rhos[write_idx] = u_new
                
                merged[j] = true
                did_merge = true
            end
        end
        
        if !did_merge
            if i != write_idx
                pos[write_idx]  = pos[i]
                rhos[write_idx] = rhos[i]
            end
        end
    end
    pg.meta.N = write_idx
end

using Random # Ensure Random is available for rand(Bool)

function _merge_particles_pairwise!(pg::ParticleGrid1D)
    # =========================================================================
    # PHASE 2: PAIRWISE MERGE (Simple Average + Random Tie Break)
    # =========================================================================
    
    safe_resize!(pg.merged_buffer, pg.meta.N)
    fill!(view(pg.merged_buffer, 1:pg.meta.N), false)
    merged = pg.merged_buffer

    write_idx = 0 
    
    pos  = get_positions(pg)
    rhos = pg.rhos
    vols = pg.volumes
    # vols = pg.volumes # Not used for simple average
    
    for i in 1:pg.meta.N
        # If 'i' was already consumed by a previous merge, skip it.
        if merged[i]; continue; end

        write_idx += 1
        
        # --- 1. Find the SINGLE closest mergeable neighbor ---
        best_j = -1
        min_dist_found = pg.min_dist # Initialize with threshold
        
        for k in pg.neighbor.ranges[i]
            j = pg.neighbor.indices[k]
            
            # Criteria:
            # 1. Forward neighbor (j > i) to prevent double processing
            # 2. Not already merged
            if j > i && !merged[j]
                dist = abs(get_xdistance(pg)[k]) 
                
                if dist < min_dist_found
                    # Found a strictly closer neighbor
                    min_dist_found = dist
                    best_j = j
                elseif abs(dist - min_dist_found) < 1e-14
                    # TIE DETECTED: Use Random coin flip
                    # If true, switch to this new candidate.
                    if rand(Bool)
                        best_j = j
                    end
                end
            end
        end
        
        if best_j != -1
            # --- MERGE PERFORMED (i + best_j) ---
            # 1. Mass Conservation (CRITICAL for Shock Speed)
            # Mass = Density * Volume
            mass_i = rhos[i] * vols[i]
            mass_j = rhos[best_j] * vols[best_j]
            total_mass = mass_i + mass_j
            total_vol  = vols[i] + vols[best_j]
            
            # 2. New Position: Geometric Average 
            # (Keeps grid smooth; Center of Mass can be used but geometric is std for regridding)
            pos[write_idx] = 0.5 * (pos[i] + pos[best_j])
            
            # 3. New Density: Total Mass / Total Volume
            rhos[write_idx] = total_vol > 1e-15 ? (total_mass / (total_vol)) : 0.0
            
            # 4. Mark 'best_j' as merged so it is skipped by the main loop
            merged[best_j] = true
            # # 1. New Position: Geometric Average 
            # pos[write_idx] = 0.5 * (pos[i] + pos[best_j])
            
            # # 2. New Density: Simple Arithmetic Mean (No Volume Weighting)
            # rhos[write_idx] = 0.5 * (rhos[i] + rhos[best_j])
            
            # # 3. Mark 'best_j' as merged so it is skipped by the main loop
            # merged[best_j] = true
            
            # Note: Other close neighbors (not best_j) are effectively "copied"
            # because they are not marked 'merged' here. They will be processed
            # as the primary particle 'i' in a subsequent iteration of the loop.
            
        else
            # --- NO MERGE (Copy i) ---
            if i != write_idx
                pos[write_idx]  = pos[i]
                rhos[write_idx] = rhos[i]
            end
        end
    end
    
    pg.meta.N = write_idx
    return nothing
end

"""
    fill_empty_voxels!(lv::LocalVoxels, pg::ParticleGrid1D, i::Int, visited::BitVector)

Fills empty voxels. 
- Case A (Interior): Inserts a particle at the exact MIDPOINT of the bounding neighbors
  with the AVERAGE density.
- Case B/C (Boundary): Un-visits the neighbor to delegate the split (no extrapolation).
"""
function fill_empty_voxels!(lv::LocalVoxels, pg::ParticleGrid1D, i::Int, visited::BitVector)
    center_offset = lv.half_bins + 1
    
    for bin_idx in 1:lv.num_bins
        if !lv.occupation[bin_idx]
            rel_idx = bin_idx - center_offset
            rel_pos = rel_idx * lv.voxel_size
            
            # Note: We calculate 'abs_pos' here only for the domain check.
            # The actual insertion position in Case A will be the physical midpoint.
            abs_pos = get_positions(pg)[i] + rel_pos
            
            # Domain Check
            if pg.bc == :periodic
                L = pg.xmax - pg.xmin
                if abs_pos > pg.xmax; abs_pos -= L; end
                if abs_pos < pg.xmin; abs_pos += L; end
            elseif abs_pos < pg.xmin || abs_pos > pg.xmax
                continue
            end
            
            closest_L_dist = -Inf; closest_L_idx = -1
            closest_R_dist = Inf;  closest_R_idx = -1
            
            for flat_idx in pg.neighbor.ranges[i]
                d_from_i = get_xdistance(pg)[flat_idx]
                nb_idx   = pg.neighbor.indices[flat_idx]
                d_new = d_from_i - rel_pos
                if d_new < 0 && d_new > closest_L_dist
                    closest_L_dist = d_new; closest_L_idx = nb_idx
                elseif d_new > 0 && d_new < closest_R_dist
                    closest_R_dist = d_new; closest_R_idx = nb_idx
                end
            end
            
            d_new_i = 0.0 - rel_pos
            if d_new_i < 0 && d_new_i > closest_L_dist
                closest_L_dist = d_new_i; closest_L_idx = i
            elseif d_new_i > 0 && d_new_i < closest_R_dist
                closest_R_dist = d_new_i; closest_R_idx = i
            end

            # --- LOGIC UPDATE ---
            
            if closest_L_idx != -1 && closest_R_idx != -1
                # Case A: Interior Voxel
                # INSERTION STRATEGY: Physical Midpoint + Simple Average
                idx_L = closest_L_idx
                idx_R = closest_R_idx
                
                # 1. Calculate Physical Midpoint
                # We use getDistance to handle periodic wrapping automatically.
                # dist_LR = pos_R - pos_L (shortest path)
                dist_LR = getDistance(pg, idx_L, idx_R)
                
                new_abs_pos = get_positions(pg)[idx_L] + 0.5 * dist_LR
                
                # Wrap the new position if necessary (standard periodic safety)
                if pg.bc == :periodic
                    L_domain = pg.xmax - pg.xmin
                    if new_abs_pos > pg.xmax; new_abs_pos -= L_domain; end
                    if new_abs_pos < pg.xmin; new_abs_pos += L_domain; end
                end
                
                # 2. Calculate Simple Average Rho
                new_rho = 0.5 * (pg.rhos[idx_L] + pg.rhos[idx_R])
                
                push!(pg.split_buffer_pos, new_abs_pos)
                push!(pg.split_buffer_rho, new_rho)
                
            elseif closest_L_idx == -1 && closest_R_idx != -1
                # Case B: Outer Voxel (Left Void) -> Un-visit Right Neighbor
                if pg.core.is_boundary[closest_R_idx]
                    push!(pg.split_buffer_pos, abs_pos)
                    push!(pg.split_buffer_rho, pg.rhos[i])
                else
                    visited[closest_R_idx] = false
                end
            elseif closest_L_idx != -1 && closest_R_idx == -1
                # Case C: Outer Voxel (Right Void) -> Un-visit Left Neighbor
                if pg.core.is_boundary[closest_L_idx]
                    push!(pg.split_buffer_pos, abs_pos)
                    push!(pg.split_buffer_rho, pg.rhos[i])
                else
                    visited[closest_L_idx] = false
                end
                
            else
                # Case D: Isolated. 
                # Fallback: create at voxel center with current rho
                # (This is rare if initial distribution is sane)
                push!(pg.split_buffer_pos, abs_pos)
                push!(pg.split_buffer_rho, pg.rhos[i])
            end
        end
    end
end

"""
    update_boundaries!(pg::ParticleGrid1D)

Iterates through all particles.
1. Removes any particle strictly outside [pg.xmin, pg.xmax].
2. Sets `is_boundary = true` if particle is in the ghost region.
3. Sets `is_boundary = false` if particle is in the interior region.
"""
function update_boundaries!(pg::ParticleGrid1D)
    # Cache bounds
    outer_min = pg.xmin
    outer_max = pg.xmax
    inner_min = pg.inner_xmin
    inner_max = pg.inner_xmax
    
    pos   = get_positions(pg)
    rhos  = pg.rhos
    is_bd = pg.core.is_boundary
    curv  = pg.curvatures
    vols  = pg.volumes
    mood  = pg.mood_events
    
    write_idx = 0
    
    for i in 1:pg.meta.N
        x = pos[i]
        
        # 1. Filter: Strictly keep only those within OUTER limits
        # (Allows particles to exist in the ghost zones, but not infinite space)
        if x < outer_min || x > outer_max
            continue
        end
        
        write_idx += 1
        
        # 2. Compaction: Move data if we skipped any particles
        if i != write_idx
            pos[write_idx]   = x
            rhos[write_idx]  = rhos[i]
            curv[write_idx]  = curv[i]
            vols[write_idx]  = vols[i]
            mood[write_idx]  = mood[i]
        end
        
        # 3. Classify: Interior vs Boundary
        # Interior is [inner_min, inner_max]
        # Boundary is everything else (but still within outer limits)
        if x >= inner_min && x <= inner_max
            is_bd[write_idx] = false
        else
            is_bd[write_idx] = true
        end
    end
    
    pg.meta.N = write_idx
    return nothing
end

# --- Voxel Helper Functions ---

"""
    reset_voxels!(lv::LocalVoxels)
"""
function reset_voxels!(lv::LocalVoxels)
    fill!(lv.occupation, false)
    center_idx = lv.half_bins + 1
    lv.occupation[center_idx] = true
end

"""
    check_occupation!(lv::LocalVoxels, pg::ParticleGrid1D, i::Int, visited::BitVector)
"""
function check_occupation!(lv::LocalVoxels, pg::ParticleGrid1D, i::Int, visited::BitVector)
    R = pg.max_dist
    center_offset = lv.half_bins + 1

    for flat_idx in pg.neighbor.ranges[i]
        dist = get_xdistance(pg)[flat_idx]
        
        if abs(dist) > R; continue; end
        
        # Calculate Bin Index
        rel_idx = floor(Int, dist / lv.voxel_size + 0.5)
        bin_idx = center_offset + rel_idx
        
        if bin_idx >= 1 && bin_idx <= lv.num_bins
            lv.occupation[bin_idx] = true
            
            # Mark Global Visited
            nb_idx = pg.neighbor.indices[flat_idx]
            visited[nb_idx] = true 
        end
    end
end
