
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

@inline function _sum_kinetic(rhos::AbstractVector{State{NK}}, i::Int, km::Kin2Macro{M}) where {NK, M}
    # ntuple is still used here because we are summing specific arbitrary indices
    return State{M}(ntuple(m -> sum(rhos[i][k] for k in km.ranges[m]), Val(M)))
end

# --- Conservative Mathematics ---
@inline function average_states_conservative(U_L::State{M}, U_R::State{M}, V_L::Float64, V_R::Float64, eq::HyperbolicPDE) where {M}
    W_L = prim2cons(eq, U_L) # Assuming prim2cons returns a State{M}
    W_R = prim2cons(eq, U_R)
    
    V_new = V_L + V_R
    
    # Native SVector math!
    W_new = (W_L * V_L + W_R * V_R) / V_new 
    return cons2prim(eq, W_new)
end

@inline function split_states_conservative(U_L::State{M}, U_R::State{M}, eq::HyperbolicPDE) where {M}
    W_L = prim2cons(eq, U_L)
    W_R = prim2cons(eq, U_R)
    
    # Native SVector math!
    W_new = 0.5 * (W_L + W_R)
    return cons2prim(eq, W_new)
end

# =========================================================================
# MULTIPLE DISPATCH KINETIC RECONSTRUCTION
# =========================================================================

@inline function reconstruct_kinetic(U_macro::State{M}, st::RelaxationSourceTerm{M, K}) where {M, K}
    flux_vals = flux(st.system_eq, U_macro)
    
    return State{K}(ntuple(Val(K)) do k
        m_idx = st.kin2macro(k)
        dim = st.dimensions[k]
        f_val = get_flux_component(flux_vals, m_idx, dim, Val(D))
        st.coefficients[k] * (U_macro[m_idx] + st.interior_factors[k] * f_val / st.relax_speeds[k])
    end)
end

@inline function reconstruct_kinetic(U_macro::State{M}, st::NonLocalRelaxationSourceTerm{M, K}, T_vals::State{M}) where {M, K}
    return State{K}(ntuple(Val(K)) do k
        m_idx = st.kin2macro(k)
        T_val = T_vals[m_idx]
        st.coefficients[m_idx] * (U_macro[m_idx] + st.interior_factor * T_val / st.relax_speeds[k])
    end)
end

# --- Dispatched Splitting Extractors ---
@inline function compute_split_state(rhos::AbstractVector{State{NK}}, idx_L::Int, idx_R::Int, eq::HyperbolicPDE, ::NoSourceTerm) where {NK}
    # Direct extraction
    U_L = rhos[idx_L] 
    U_R = rhos[idx_R]
    return split_states_conservative(U_L, U_R, eq)
end

@inline function compute_split_state(rhos::AbstractVector{State{K}}, idx_L::Int, idx_R::Int, eq::HyperbolicPDE, st::AbstractSourceTerm) where {K}
    U_macro_L = _sum_kinetic(rhos, idx_L, st.kin2macro)
    U_macro_R = _sum_kinetic(rhos, idx_R, st.kin2macro)
    U_macro_new = split_states_conservative(U_macro_L, U_macro_R, eq)
    return _get_split_kinetic_state(U_macro_new, idx_L, idx_R, st)
end

@inline _get_split_kinetic_state(U_new::State, idx_L::Int, idx_R::Int, st::RelaxationSourceTerm) = reconstruct_kinetic(U_new, st)

@inline function _get_split_kinetic_state(U_new::State{M}, idx_L::Int, idx_R::Int, st::NonLocalRelaxationSourceTerm) where {M}
    # Extract directly into SVectors
    T_L = State{M}(ntuple(c -> st.T_potential[idx_L, c], Val(M)))
    T_R = State{M}(ntuple(c -> st.T_potential[idx_R, c], Val(M)))
    
    # Native SVector math!
    T_new = 0.5 * (T_L + T_R) 
    return reconstruct_kinetic(U_new, st, T_new)
end

# --- Dispatched Merging Extractors ---
@inline function compute_merged_state(rhos::AbstractVector{State{NK}}, i::Int, j::Int, V_i::Float64, V_j::Float64, eq::HyperbolicPDE, ::NoSourceTerm) where {NK}
    U_i = rhos[i]
    U_j = rhos[j]
    return average_states_conservative(U_i, U_j, V_i, V_j, eq)
end

@inline function compute_merged_state(rhos::AbstractVector{State{NK}}, i::Int, j::Int, V_i::Float64, V_j::Float64, eq::HyperbolicPDE, st::AbstractSourceTerm) where {NK}
    U_macro_i = _sum_kinetic(rhos, i, st.kin2macro)
    U_macro_j = _sum_kinetic(rhos, j, st.kin2macro)
    U_macro_new = average_states_conservative(U_macro_i, U_macro_j, V_i, V_j, eq)
    return _get_merged_kinetic_state(U_macro_new, i, j, V_i, V_j, st)
end

@inline _get_merged_kinetic_state(U_new::State, i::Int, j::Int, V_i::Float64, V_j::Float64, st::RelaxationSourceTerm) = reconstruct_kinetic(U_new, st)

@inline function _get_merged_kinetic_state(U_new::State{M}, i::Int, j::Int, V_i::Float64, V_j::Float64, st::NonLocalRelaxationSourceTerm) where {M}
    T_i = State{M}(ntuple(c -> st.T_potential[i, c], Val(M)))
    T_j = State{M}(ntuple(c -> st.T_potential[j, c], Val(M)))
    
    # Native SVector math!
    T_new = (T_i * V_i + T_j * V_j) / (V_i + V_j)
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

    # PHASE 4: FINALIZE
    safe_resize!(pg.neighbor.ranges, pg.meta.N)
    pg.reorder(pg)
    pg.neighbor(pg)
    determineVolumes!(pg)
end

# =========================================================================
# PHASE 1: SPLITTING (Dual-Grid Global Search)
# =========================================================================

function _split_particles!(pg::ParticleGrid{D, M}, eq::HyperbolicPDE, source_term::AbstractSourceTerm) where {D, M}
    bins = pg.bins
    fine_occ = bins.fine_occupation
    fine_type = bins.fine_type
    fine_dims = bins.fine_dims
    
    N_curr = pg.meta.N
    num_new = 0
    
    # 1. First pass: Count how many new particles we need to allocate
    @inbounds for i in 1:length(fine_occ)
        if !fine_occ[i] && fine_type[i] == 1 # Empty AND Interior
            num_new += 1
        end
    end
    
    if num_new == 0
        return nothing
    end
    
    # 2. Allocate space safely
    N_total = N_curr + num_new
    ensure_capacity!(pg, N_total)
    
    pos = get_positions(pg)
    rhos = pg.rhos
    is_boundary = pg.core.is_boundary
    
    # Alias buffers to save states directly into the final array
    curv = pg.curvatures
    vols = pg.core.volumes
    mood = pg.mood_events
    
    write_idx = N_curr
    cart_indices = CartesianIndices(fine_dims)
    
    # 3. Second pass: Actually spawn the particles
    for flat_idx in 1:length(fine_occ)
        if !fine_occ[flat_idx] && fine_type[flat_idx] == 1
            write_idx += 1
            
            # A. Calculate the exact physical center of this empty fine bin
            I = cart_indices[flat_idx]
            new_pos = ntuple(Val(D)) do d
                bins.mins[d] + (I[d] - 0.5) * bins.fine_size
            end
            pos[write_idx] = Space{D}(new_pos)
            
            # B. Interpolate State (Find nearest left/right neighbors)
            # We use the coarse grid linked-list to find the neighbors efficiently!
            coarse_idx = get_flat_bin_index(pos[write_idx], bins.mins, bins.coarse_size, bins.coarse_dims)
            
            # Simple fallback for 1D nearest-neighbor search within the coarse bin
            closest_L_idx = -1
            closest_R_idx = -1
            min_dist_L = Inf
            min_dist_R = Inf
            
            j = bins.head[coarse_idx]
            while j > 0
                dist = pos[write_idx][1] - pos[j][1] # 1D specific
                if dist > 0 && dist < min_dist_L
                    min_dist_L = dist
                    closest_L_idx = j
                elseif dist < 0 && abs(dist) < min_dist_R
                    min_dist_R = abs(dist)
                    closest_R_idx = j
                end
                j = bins.next[j]
            end
            
            # C. Calculate Macroscopic & Kinetic State
            if closest_L_idx != -1 && closest_R_idx != -1
                # Standard conservative split
                new_state = compute_split_state(rhos, closest_L_idx, closest_R_idx, eq, source_term)
            else
                # Fallback: Just copy the nearest available particle if at an edge
                fallback_idx = closest_L_idx != -1 ? closest_L_idx : (closest_R_idx != -1 ? closest_R_idx : 1)
                new_state = rhos[fallback_idx]
            end
            
            # D. Write to arrays directly (No more matrix extract/write loops!)
            rhos[write_idx] = new_state
            is_boundary[write_idx] = false
            curv[write_idx] = State{M}(ntuple(_->0.0, M))
            vols[write_idx] = 0.0
            mood[write_idx] = SVector{M, Bool}(ntuple(_->false, M))
        end
    end
    
    # 4. Finalize
    pg.meta.N = N_total
    
    # Update linked lists immediately so Phase 2 (Merging) knows these new particles exist
    build_global_bins!(pg) 
    
    return nothing
end
# =========================================================================
# PHASE 2: MERGING (Dual-Grid Global Search)
# =========================================================================

function _merge_particles_pairwise!(pg::ParticleGrid{D, M, WF, GM, BC}, eq::HyperbolicPDE, source_term::AbstractSourceTerm) where {D, M, WF, GM, BC}
    N = pg.meta.N
    ensure_shared_capacity!(pg, N)
    
    merged = pg.shared.bit_buffer
    fill!(view(merged, 1:N), false)

    write_idx = 0 
    pos  = get_positions(pg)
    rhos = pg.rhos
    vols = pg.core.volumes
    
    min_dist_thresh = hasproperty(pg, :min_dist) ? pg.min_dist : pg.meta.dx[1] * 0.25
    
    bins = pg.bins
    coarse_dims = bins.coarse_dims
    ci = CartesianIndices(coarse_dims)
    li = LinearIndices(coarse_dims)
    window = CartesianIndices(ntuple(_ -> -1:1, Val(D)))
    L = pg.meta.maxs .- pg.meta.mins
    
    # The compiler evaluates this at compile-time and deletes the unused branch below!
    is_periodic = (BC == :periodic)

    for i in 1:N
        if merged[i]
            continue
        end
        # 1. Map particle 'i' to its fine bin to check boundary status
        fine_idx = get_flat_bin_index(pos[i], bins.mins, bins.fine_size, bins.fine_dims)
        
        # If it's 0 (OutOfBounds), we just skip it entirely!
        # It won't merge, and write_idx won't increment, effectively deleting it.
        if !is_periodic && bins.fine_type[fine_idx] == 0 
            continue
        end
        
        write_idx += 1
        best_j = -1
        min_dist_found = min_dist_thresh 
        
        # 1. Map particle 'i' to its coarse bin
        bin_idx = get_flat_bin_index(pos[i], bins.mins, bins.coarse_size, coarse_dims)
        cart_idx = ci[bin_idx]
        
        # 2. Search local neighborhood using the Coarse Grid
        for offset in window
            nb_cart = cart_idx + offset
            
            if is_periodic
                wrapped_cart = map((nc, cd) -> mod1(nc, cd), Tuple(nb_cart), coarse_dims)
                nb_bin_idx = li[CartesianIndex(wrapped_cart)]
            else
                if !checkbounds(Bool, li, nb_cart)
                    continue
                end
                nb_bin_idx = li[nb_cart]
            end
            
            # Walk the linked list
            j = bins.head[nb_bin_idx]
            while j > 0
                # Enforce j > i to avoid merging backwards or with itself
                if j > i && !merged[j]
                    dist_vec = is_periodic ? getPeriodicDistance(pos, i, j, L) : getEuclideanDistance(pos, i, j)
                    dist = sqrt(sum(abs2, dist_vec)) # D-dimensional distance
                    
                    if dist < min_dist_found
                        min_dist_found = dist
                        best_j = j
                    elseif abs(dist - min_dist_found) < 1e-14
                        if rand(Bool)
                            best_j = j
                        end
                    end
                end
                j = bins.next[j]
            end
        end
        
        # 3. Apply the merge if a partner was found
        if best_j != -1
            V_i = vols[i]
            V_j = vols[best_j]
            
            # Position: Geometric Average natively handled by Space{D} / SVector
            pos[write_idx] = 0.5 * (pos[i] + pos[best_j])
            
            # State Averaging natively handled by State{M} / SVector
            rhos[write_idx] = compute_merged_state(rhos, i, best_j, V_i, V_j, eq, source_term)
            
            # Update Volumes
            vols[write_idx] = V_i + V_j
            merged[best_j] = true
        else
            # Compact the arrays if no merge occurred
            if i != write_idx
                pos[write_idx]  = pos[i]
                rhos[write_idx] = rhos[i]
                vols[write_idx] = vols[i]
            end
        end
    end
    
    pg.meta.N = write_idx
    return nothing
end