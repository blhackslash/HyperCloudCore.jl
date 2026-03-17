
# =========================================================================
# SHARED BUFFER ACCESSORS & SAFE RESIZING
# =========================================================================

@inline function ensure_shared_capacity!(pg::ParticleGrid, required_capacity::Int)
    if length(pg.shared.pos_buffer) < required_capacity
        new_cap = ceil(Int, required_capacity * 1.25)
        resize!(pg.shared.pos_buffer, new_cap)
        resize!(pg.shared.bit_buffer, new_cap)
        resize!(pg.shared.int_buffer, new_cap)
        
        # Explicit Matrix resize to avoid MethodErrors on standard resize!
        M = size(pg.rhos, 2)
        if size(pg.shared.rho_buffer, 1) < new_cap
            pg.shared.rho_buffer = Matrix{Float64}(undef, new_cap, M)
        end
    end
end

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

# =========================================================================
# MULTIPLE DISPATCH KINETIC RECONSTRUCTION
# =========================================================================

@inline function reconstruct_kinetic(U_macro::Tuple, st::RelaxationSourceTerm{NM, NK}) where {NM, NK}
    flux_vals = flux(st.system_eq, U_macro)
    return ntuple(Val(NK)) do k
        m_idx = st.kin2macro(k)
        dim = st.dimensions[k]
        f_val = get_flux_component(flux_vals, m_idx, dim, Val(D))
        st.coefficients[k] * (U_macro[m_idx] + st.interior_factors[k] * f_val / st.relax_speeds[k])
    end
end

@inline function reconstruct_kinetic(U_macro::Tuple, st::NonLocalRelaxationSourceTerm{NM, NK}, T_vals::Tuple) where {NM, NK}
    return ntuple(Val(NK)) do k
        m_idx = st.kin2macro(k)
        T_val = T_vals[m_idx]
        st.coefficients[m_idx] * (U_macro[m_idx] + st.interior_factor * T_val / st.relax_speeds[k])
    end
end

# --- Dispatched Splitting Extractors ---
@inline function compute_split_state(rhos::AbstractMatrix, idx_L::Int, idx_R::Int, eq::HyperbolicPDE, ::NoSourceTerm, M::Int)
    U_L = _extract_state(rhos, idx_L, M)
    U_R = _extract_state(rhos, idx_R, M)
    return split_states_conservative(U_L, U_R, eq)
end

@inline function compute_split_state(rhos::AbstractMatrix, idx_L::Int, idx_R::Int, eq::HyperbolicPDE, st::AbstractSourceTerm, M::Int)
    U_macro_L = _sum_kinetic(rhos, idx_L, st.kin2macro)
    U_macro_R = _sum_kinetic(rhos, idx_R, st.kin2macro)
    U_macro_new = split_states_conservative(U_macro_L, U_macro_R, eq)
    return _get_split_kinetic_state(U_macro_new, idx_L, idx_R, st)
end

@inline _get_split_kinetic_state(U_new::Tuple, idx_L::Int, idx_R::Int, st::RelaxationSourceTerm) = reconstruct_kinetic(U_new, st)
@inline function _get_split_kinetic_state(U_new::Tuple, idx_L::Int, idx_R::Int, st::NonLocalRelaxationSourceTerm)
    T_L = ntuple(c -> st.T_potential[idx_L, c], Val(length(U_new)))
    T_R = ntuple(c -> st.T_potential[idx_R, c], Val(length(U_new)))
    T_new = ntuple(c -> 0.5 * (T_L[c] + T_R[c]), length(T_L))
    return reconstruct_kinetic(U_new, st, T_new)
end

# --- Dispatched Merging Extractors ---
@inline function compute_merged_state(rhos::AbstractMatrix, i::Int, j::Int, V_i::Float64, V_j::Float64, eq::HyperbolicPDE, ::NoSourceTerm, M::Int)
    U_i = _extract_state(rhos, i, M)
    U_j = _extract_state(rhos, j, M)
    return average_states_conservative(U_i, U_j, V_i, V_j, eq)
end

@inline function compute_merged_state(rhos::AbstractMatrix, i::Int, j::Int, V_i::Float64, V_j::Float64, eq::HyperbolicPDE, st::AbstractSourceTerm, M::Int)
    U_macro_i = _sum_kinetic(rhos, i, st.kin2macro)
    U_macro_j = _sum_kinetic(rhos, j, st.kin2macro)
    U_macro_new = average_states_conservative(U_macro_i, U_macro_j, V_i, V_j, eq)
    return _get_merged_kinetic_state(U_macro_new, i, j, V_i, V_j, st)
end

@inline _get_merged_kinetic_state(U_new::Tuple, i::Int, j::Int, V_i::Float64, V_j::Float64, st::RelaxationSourceTerm) = reconstruct_kinetic(U_new, st)
@inline function _get_merged_kinetic_state(U_new::Tuple, i::Int, j::Int, V_i::Float64, V_j::Float64, st::NonLocalRelaxationSourceTerm)
    T_i = ntuple(c -> st.T_potential[i, c], Val(length(U_new)))
    T_j = ntuple(c -> st.T_potential[j, c], Val(length(U_new)))
    T_new = ntuple(c -> (T_i[c]*V_i + T_j[c]*V_j)/(V_i+V_j), length(T_i))
    return reconstruct_kinetic(U_new, st, T_new)
end

# =========================================================================
# MAIN ROUTINE
# =========================================================================
# User-facing wrapper: automatically extracts the mover!
function manage_particles!(pg::ParticleGrid1D, source_term::AbstractSourceTerm=NoSourceTerm())
    _manage_particles!(pg.mover, pg, source_term)
end

function manage_particles!(::NoGridMover, kwargs...)
    return
end

function manage_particles!(gm::PhysicalGridMover, pg::ParticleGrid1D, source_term::AbstractSourceTerm=NoSourceTerm())

    eq = gm.pde
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
    pg.reorder(pg)
    pg.neighbor(pg)
    determineVolumes!(pg)
end

# =========================================================================
# PHASE 1: SPLITTING
# =========================================================================

function _split_particles!(pg::ParticleGrid1D, eq::HyperbolicPDE, source_term::AbstractSourceTerm)
    N = pg.meta.N
    ensure_capacity!(pg.shared, N)
    is_boundary = pg.core.is_boundary
    visited = pg.shared.bit_buffer
    fill!(view(visited, 1:N), false)
    
    lv = pg.manage.local_voxels
    M = size(pg.rhos, 2)
    
    num_new = Ref(0) # Track how many new particles we generate
    
    for i in 1:N
        if !visited[i]
            reset_voxels!(lv)
            check_occupation!(lv, pg, i, visited)
            fill_empty_voxels!(lv, pg, i, visited, eq, source_term, num_new)
            visited[i] = true
        end
    end
    
    N_new = num_new[]
    if N_new > 0
        N_curr = pg.meta.N
        N_total = N_curr + N_new
        
        ensure_capacity!(pg, N_total)
        
        grid_pos = get_positions(pg)
        shared_pos = get_positions(pg.shared)
        shared_bd = pg.shared.int_buffer
        
        for k in 1:N_new
            idx = N_curr + k
            grid_pos[idx] = shared_pos[k]
            is_boundary[idx] = shared_bd[k]
            
            for c in 1:M
                pg.rhos[idx, c] = pg.shared.rho_buffer[k, c]
            end
            
            pg.curvatures[idx] = 0.0
            pg.core.volumes[idx] = 0.0
            pg.mood_events[idx] = false
        end
        
        pg.meta.N = N_total
        safe_resize!(pg.neighbor.ranges, pg.meta.N)
        pg.reorder(pg)
        pg.neighbor(pg)
    end
end

function fill_empty_voxels!(lv::LocalVoxels, pg::ParticleGrid1D, i::Int, visited::AbstractVector{Bool}, eq::HyperbolicPDE, source_term::AbstractSourceTerm, num_new::Ref{Int})
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
                
                # We are officially creating a particle!
                new_state_tuple = compute_split_state(pg.rhos, idx_L, idx_R, eq, source_term, M)
                
                idx_new = num_new[] + 1
                ensure_shared_capacity!(pg, idx_new)
                num_new[] = idx_new
                
                pg.shared.int_buffer[idx_new] = false
                get_positions(pg.shared)[idx_new] = new_abs_pos
                _write_state!(pg.shared.rho_buffer, idx_new, new_state_tuple, M)
                
            elseif closest_L_idx == -1 && closest_R_idx != -1
                if pg.core.is_boundary[i]
                    idx_new = num_new[] + 1
                    ensure_shared_capacity!(pg, idx_new)
                    num_new[] = idx_new
                    
                    pg.shared.int_buffer[idx_new] = true
                    get_positions(pg.shared)[idx_new] = abs_pos
                    _write_state!(pg.shared.rho_buffer, idx_new, _extract_state(pg.rhos, i, M), M)
                else
                    visited[closest_R_idx] = false
                end
            elseif closest_L_idx != -1 && closest_R_idx == -1
                if pg.core.is_boundary[i]
                    idx_new = num_new[] + 1
                    ensure_shared_capacity!(pg, idx_new)
                    num_new[] = idx_new

                    pg.shared.int_buffer[idx_new] = true
                    get_positions(pg.shared)[idx_new] = abs_pos
                    _write_state!(pg.shared.rho_buffer, idx_new, _extract_state(pg.rhos, i, M), M)
                else
                    visited[closest_L_idx] = false
                end
            else
                idx_new = num_new[] + 1
                ensure_shared_capacity!(pg, idx_new)
                num_new[] = idx_new
                
                pg.shared.int_buffer[idx_new] = true
                get_positions(pg.shared)[idx_new] = abs_pos
                _write_state!(pg.shared.rho_buffer, idx_new, _extract_state(pg.rhos, i, M), M)
            end
        end
    end
end

function check_occupation!(lv::LocalVoxels, pg::ParticleGrid1D, i::Int, visited::AbstractVector{Bool})
    R = pg.meta.R
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

function _merge_particles_pairwise!(pg::ParticleGrid1D, eq::HyperbolicPDE, source_term::AbstractSourceTerm)
    N = pg.meta.N
    ensure_shared_capacity!(pg, N)
    
    merged = pg.shared.bit_buffer
    fill!(view(merged, 1:N), false)

    write_idx = 0 
    pos  = get_positions(pg)
    rhos = pg.rhos
    vols = pg.core.volumes
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
            
            # 2. Fully Dispatched State Averaging
            new_state_tuple = compute_merged_state(rhos, i, best_j, V_i, V_j, eq, source_term, M)
            _write_state!(rhos, write_idx, new_state_tuple, M)
            
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
    outer_min = pg.meta.mins[1]
    outer_max = pg.meta.maxs[1]
    
    inner_min = pg.meta.inner_mins[1]
    inner_max = pg.meta.inner_maxs[1]
    
    pos   = get_positions(pg)
    rhos  = pg.rhos
    is_bd = pg.core.is_boundary
    curv  = pg.curvatures
    vols  = pg.core.volumes
    mood  = pg.mood_events
    M     = size(rhos, 2)
    
    write_idx = 0

    for i in 1:pg.meta.N
        x = pos[i] 
        
        if x < outer_min || x > outer_max
            continue
        end
        
        write_idx += 1
        
        if i != write_idx
            pos[write_idx]   = x
            _write_state!(rhos, write_idx, _extract_state(rhos, i, M), M)
            curv[write_idx]  = curv[i]
            vols[write_idx]  = vols[i]
            mood[write_idx]  = mood[i]
        end
        
        if x >= inner_min && x <= inner_max
            is_bd[write_idx] = false
        else
            is_bd[write_idx] = true
        end
    end
    
    pg.meta.N = write_idx
    return nothing
end