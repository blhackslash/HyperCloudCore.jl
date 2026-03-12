# =========================================================================
# HELPER FUNCTIONS (Zero-Allocation State Management)
# =========================================================================

@inline function _extract_state(rhos::AbstractMatrix, i::Int, M::Int)
    return ntuple(c -> rhos[i, c], Val(M))
end

@inline function _write_state!(rhos::AbstractMatrix, i::Int, U::Tuple, M::Int)
    for c in 1:M; rhos[i, c] = U[c]; end
end

@inline function _sum_kinetic(rhos::AbstractMatrix, i::Int, km::Kin2Macro{NM}) where {NM}
    return ntuple(m -> sum(rhos[i, k] for k in km.ranges[m]), Val(NM))
end

# --- Conservative Mathematics ---
@inline function average_states_conservative(U_L::Tuple, U_R::Tuple, V_L::Float64, V_R::Float64, eq::HyperbolicPDE)
    W_L = prim2cons(eq, U_L)
    W_R = prim2cons(eq, U_R)
    V_new = V_L + V_R
    W_new = ntuple(c -> (W_L[c] * V_L + W_R[c] * V_R) / V_new, length(W_L))
    return cons2prim(eq, W_new)
end

@inline function split_states_conservative(U_L::Tuple, U_R::Tuple, eq::HyperbolicPDE)
    W_L = prim2cons(eq, U_L)
    W_R = prim2cons(eq, U_R)
    W_new = ntuple(c -> 0.5 * (W_L[c] + W_R[c]), length(W_L))
    return cons2prim(eq, W_new)
end

# --- Kinetic State Reconstruction ---
@inline function reconstruct_kinetic(U_macro::Tuple, st::RelaxationSourceTerm{D, N, NK, PDE}) where {D, N, NK, PDE}
    flux_vals = flux(st.system_eq, U_macro)
    return ntuple(Val(NK)) do k
        m_idx = st.kin2macro(k)
        dim = st.dimensions[k]
        f_val = get_flux_component(flux_vals, m_idx, dim, Val(D))
        st.coefficients[k] * (U_macro[m_idx] + st.interior_factors[k] * f_val / st.relax_speeds[k])
    end
end

@inline function reconstruct_kinetic(U_macro::Tuple, st::NonLocalRelaxationSourceTerm{D, N, NK, PDE}, T_vals::Tuple) where {D, N, NK, PDE}
    return ntuple(Val(NK)) do k
        m_idx = st.kin2macro(k)
        T_val = T_vals[m_idx]
        st.coefficients[m_idx] * (U_macro[m_idx] + st.interior_factor * T_val / st.relax_speeds[k])
    end
end

# =========================================================================
# MAIN ROUTINE
# =========================================================================
manage_particles!(kwargs...) = return 
function manage_particles!(pg::ParticleGrid1D, eq::HyperbolicPDE, source_term=nothing)
    # PHASE 1: VOXEL FILL (Splitting)
    if hasproperty(pg.manage, :local_voxels)
        _split_particles!(pg, eq, source_term)
    end
    
    # PHASE 2: PAIRWISE MERGE (Coarsening)
    _merge_particles_pairwise!(pg, eq, source_term)

    # PHASE 3: BOUNDARY UPDATE (Cleanup)
    update_boundaries!(pg)

    # PHASE 4: FINALIZE
    safe_resize!(pg.neighbor.ranges, pg.meta.N)
    sort_1d_particles!(pg)
    updateNeighbors!(pg)
    determineVolumes!(pg)
end

# =========================================================================
# PHASE 1: SPLITTING
# =========================================================================

function _split_particles!(pg::ParticleGrid1D, eq::HyperbolicPDE, source_term)
    empty!(pg.manage.split_buffer_pos)
    empty!(pg.manage.split_buffer_rho)
    
    visited = pg.manage.merge_flags
    if length(visited) < pg.meta.N; safe_resize!(visited, pg.meta.N); end
    fill!(view(visited, 1:pg.meta.N), false)
    
    lv = pg.manage.local_voxels
    M = size(pg.rhos, 2)
    
    for i in 1:pg.meta.N
        if !visited[i]
            reset_voxels!(lv)
            check_occupation!(lv, pg, i, visited)
            fill_empty_voxels!(lv, pg, i, visited, eq, source_term)
            visited[i] = true
        end
    end
    
    N_new = length(pg.manage.split_buffer_pos)
    if N_new > 0
        N_curr = pg.meta.N
        N_total = N_curr + N_new
        
        safe_resize!(get_positions(pg), N_total)
        safe_resize!(pg.rhos, N_total)
        safe_resize!(pg.curvatures, N_total)
        safe_resize!(pg.core.is_boundary, N_total)
        safe_resize!(pg.volumes, N_total)
        safe_resize!(pg.mood_events, N_total)
        
        for k in 1:N_new
            idx = N_curr + k
            get_positions(pg)[idx] = pg.manage.split_buffer_pos[k]
            _write_state!(pg.rhos, idx, pg.manage.split_buffer_rho[k], M)
            
            pg.curvatures[idx] = 0.0
            pg.core.is_boundary[idx] = false 
            pg.volumes[idx] = 0.0
            pg.mood_events[idx] = false
        end
        
        pg.meta.N = N_total
        safe_resize!(pg.neighbor.ranges, pg.meta.N)
        sort_1d_particles!(pg)
        updateNeighbors!(pg)
    end
end

function fill_empty_voxels!(lv::LocalVoxels, pg::ParticleGrid1D, i::Int, visited::AbstractVector{Bool}, eq::HyperbolicPDE, source_term)
    center_offset = lv.half_bins + 1
    M = size(pg.rhos, 2)
    
    for bin_idx in 1:lv.num_bins
        if !lv.occupation[bin_idx]
            rel_idx = bin_idx - center_offset
            rel_pos = rel_idx * lv.voxel_size
            abs_pos = get_positions(pg)[i] + rel_pos
            
            # Domain Check
            if pg.meta.bc == :periodic
                L = pg.meta.maxs[1] - pg.meta.mins[1]
                if abs_pos > pg.meta.maxs[1]; abs_pos -= L; end
                if abs_pos < pg.meta.mins[1]; abs_pos += L; end
            elseif abs_pos < pg.meta.mins[1] || abs_pos > pg.meta.maxs[1]
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

            if closest_L_idx != -1 && closest_R_idx != -1
                idx_L, idx_R = closest_L_idx, closest_R_idx
                dist_LR = getDistance(pg, idx_L, idx_R)
                new_abs_pos = get_positions(pg)[idx_L] + 0.5 * dist_LR
                
                if pg.meta.bc == :periodic
                    L_domain = pg.meta.maxs[1] - pg.meta.mins[1]
                    if new_abs_pos > pg.meta.maxs[1]; new_abs_pos -= L_domain; end
                    if new_abs_pos < pg.meta.mins[1]; new_abs_pos += L_domain; end
                end
                
                # --- State Reconstruction for New Particle ---
                local new_state_tuple
                if isnothing(source_term)
                    U_L = _extract_state(pg.rhos, idx_L, M)
                    U_R = _extract_state(pg.rhos, idx_R, M)
                    new_state_tuple = split_states_conservative(U_L, U_R, eq)
                else
                    U_macro_L = _sum_kinetic(pg.rhos, idx_L, source_term.kin2macro)
                    U_macro_R = _sum_kinetic(pg.rhos, idx_R, source_term.kin2macro)
                    U_macro_new = split_states_conservative(U_macro_L, U_macro_R, eq)
                    
                    if source_term isa NonLocalRelaxationSourceTerm
                        T_L = ntuple(c -> source_term.T_potential[idx_L, c], Val(length(U_macro_L)))
                        T_R = ntuple(c -> source_term.T_potential[idx_R, c], Val(length(U_macro_R)))
                        T_new = ntuple(c -> 0.5 * (T_L[c] + T_R[c]), length(T_L))
                        new_state_tuple = reconstruct_kinetic(U_macro_new, source_term, T_new)
                    else
                        new_state_tuple = reconstruct_kinetic(U_macro_new, source_term)
                    end
                end
                
                push!(pg.manage.split_buffer_pos, new_abs_pos)
                push!(pg.manage.split_buffer_rho, new_state_tuple)
                
            elseif closest_L_idx == -1 && closest_R_idx != -1
                if pg.core.is_boundary[closest_R_idx]
                    push!(pg.manage.split_buffer_pos, abs_pos)
                    push!(pg.manage.split_buffer_rho, _extract_state(pg.rhos, i, M))
                else
                    visited[closest_R_idx] = false
                end
            elseif closest_L_idx != -1 && closest_R_idx == -1
                if pg.core.is_boundary[closest_L_idx]
                    push!(pg.manage.split_buffer_pos, abs_pos)
                    push!(pg.manage.split_buffer_rho, _extract_state(pg.rhos, i, M))
                else
                    visited[closest_L_idx] = false
                end
            else
                push!(pg.manage.split_buffer_pos, abs_pos)
                push!(pg.manage.split_buffer_rho, _extract_state(pg.rhos, i, M))
            end
        end
    end
end

function check_occupation!(lv::LocalVoxels, pg::ParticleGrid1D, i::Int, visited::AbstractVector{Bool})
    # Uses dx[1] as proxy for max_dist if not defined
    R = hasproperty(pg, :min_dist) ? pg.min_dist : pg.meta.dx[1] * 0.8 
    center_offset = lv.half_bins + 1

    for flat_idx in pg.neighbor.ranges[i]
        dist = get_xdistance(pg)[flat_idx]
        if abs(dist) > R; continue; end
        
        rel_idx = floor(Int, dist / lv.voxel_size + 0.5)
        bin_idx = center_offset + rel_idx
        
        if bin_idx >= 1 && bin_idx <= lv.num_bins
            lv.occupation[bin_idx] = true
            nb_idx = pg.neighbor.indices[flat_idx]
            visited[nb_idx] = true 
        end
    end
end

function reset_voxels!(lv::LocalVoxels)
    fill!(lv.occupation, false)
    center_idx = lv.half_bins + 1
    lv.occupation[center_idx] = true
end

# =========================================================================
# PHASE 2: MERGING
# =========================================================================

function _merge_particles_pairwise!(pg::ParticleGrid1D, eq::HyperbolicPDE, source_term)
    N = pg.meta.N
    merged = pg.manage.merge_flags
    if length(merged) < N; safe_resize!(merged, N); end
    fill!(view(merged, 1:N), false)

    write_idx = 0 
    pos  = get_positions(pg)
    rhos = pg.rhos
    vols = pg.volumes
    M    = size(rhos, 2)
    min_dist_thresh = hasproperty(pg, :min_dist) ? pg.min_dist : pg.meta.dx[1] * 0.25
    
    for i in 1:N
        if merged[i]; continue; end

        write_idx += 1
        best_j = -1
        min_dist_found = min_dist_thresh 
        
        for k in pg.neighbor.ranges[i]
            j = pg.neighbor.indices[k]
            
            if j > i && !merged[j]
                dist = abs(get_xdistance(pg)[k]) 
                
                if dist < min_dist_found
                    min_dist_found = dist
                    best_j = j
                elseif abs(dist - min_dist_found) < 1e-14
                    if rand(Bool); best_j = j; end
                end
            end
        end
        
        if best_j != -1
            V_i = vols[i]
            V_j = vols[best_j]
            
            # 1. New Position: Geometric Average
            pos[write_idx] = 0.5 * (pos[i] + pos[best_j])
            
            # 2. Conservative State Averaging
            if isnothing(source_term)
                U_i = _extract_state(rhos, i, M)
                U_j = _extract_state(rhos, best_j, M)
                U_new = average_states_conservative(U_i, U_j, V_i, V_j, eq)
                _write_state!(rhos, write_idx, U_new, M)
            else
                U_macro_i = _sum_kinetic(rhos, i, source_term.kin2macro)
                U_macro_j = _sum_kinetic(rhos, best_j, source_term.kin2macro)
                U_macro_new = average_states_conservative(U_macro_i, U_macro_j, V_i, V_j, eq)
                
                local V_kin_new
                if source_term isa NonLocalRelaxationSourceTerm
                    T_i = ntuple(c -> source_term.T_potential[i, c], Val(length(U_macro_i)))
                    T_j = ntuple(c -> source_term.T_potential[best_j, c], Val(length(U_macro_j)))
                    T_new = ntuple(c -> (T_i[c]*V_i + T_j[c]*V_j)/(V_i+V_j), length(T_i))
                    V_kin_new = reconstruct_kinetic(U_macro_new, source_term, T_new)
                else
                    V_kin_new = reconstruct_kinetic(U_macro_new, source_term)
                end
                _write_state!(rhos, write_idx, V_kin_new, M)
            end
            
            # 3. Update Volumes
            vols[write_idx] = V_i + V_j
            merged[best_j] = true
        else
            if i != write_idx
                pos[write_idx]  = pos[i]
                _write_state!(rhos, write_idx, _extract_state(rhos, i, M), M)
                vols[write_idx] = vols[i]
            end
        end
    end
    pg.meta.N = write_idx
    return nothing
end

# =========================================================================
# PHASE 3: BOUNDARIES
# =========================================================================

function update_boundaries!(pg::ParticleGrid1D)
    # Derive boundary limits dynamically from the grid metadata 
    outer_min = pg.meta.mins[1]
    outer_max = pg.meta.maxs[1]
    
    # Inner interior bounds
    inner_min = outer_min + pg.meta.N_ghost * pg.meta.dx[1]
    inner_max = outer_max - pg.meta.N_ghost * pg.meta.dx[1]
    
    pos   = get_positions(pg)
    rhos  = pg.rhos
    is_bd = pg.core.is_boundary
    curv  = pg.curvatures
    vols  = pg.volumes
    mood  = pg.mood_events
    M     = size(rhos, 2)
    
    write_idx = 0
    
    for i in 1:pg.meta.N
        x = pos[i] 
        
        # 1. Filter: Strictly keep only those within OUTER limits
        if x < outer_min || x > outer_max
            continue
        end
        
        write_idx += 1
        
        # 2. Compaction
        if i != write_idx
            pos[write_idx]   = x
            _write_state!(rhos, write_idx, _extract_state(rhos, i, M), M)
            curv[write_idx]  = curv[i]
            vols[write_idx]  = vols[i]
            mood[write_idx]  = mood[i]
        end
        
        # 3. Classify: Interior vs Ghost Boundary
        if x >= inner_min && x <= inner_max
            is_bd[write_idx] = false
        else
            is_bd[write_idx] = true
        end
    end
    
    pg.meta.N = write_idx
    return nothing
end