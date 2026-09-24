# =========================================================================
# SHARED BUFFER ACCESSORS & SAFE RESIZING
# =========================================================================

@inline function ensure_shared_capacity!(pg::ParticleGrid{D, M, T}, required_capacity::Int) where {D, M, T}
    if length(pg.shared.pos_buffer) < required_capacity
        new_cap = ceil(Int, required_capacity * 1.25)
        resize!(pg.shared.pos_buffer, new_cap)
        resize!(pg.shared.bit_buffer, new_cap)
        resize!(pg.shared.int_buffer, new_cap)
        
        # Explicit Matrix resize to avoid MethodErrors
        M_dim = size(pg.rhos, 2)
        if size(pg.shared.rho_buffer, 1) < new_cap
            pg.shared.rho_buffer = Matrix{T}(undef, new_cap, M_dim)
        end
    end
end

# --- Conservative Mathematics (Parameterized) ---
@inline function average_states_conservative(U_L::State{M, T}, U_R::State{M, T}, V_L::T, V_R::T, eq::HyperbolicPDE) where {M, T}
    W_L = prim2cons(eq, U_L) 
    W_R = prim2cons(eq, U_R)
    
    V_new = V_L + V_R
    W_new = (W_L * V_L + W_R * V_R) / V_new 
    
    return cons2prim(eq, W_new)
end

@inline function split_states_conservative(U_L::State{M, T}, U_R::State{M, T}, eq::HyperbolicPDE) where {M, T}
    W_L = prim2cons(eq, U_L)
    W_R = prim2cons(eq, U_R)
    
    W_new = T(0.5) * (W_L + W_R)
    return cons2prim(eq, W_new)
end

function _merge_particles_pairwise!(pg::ParticleGrid{D, M, T, WF, GM, BC}, eq::HyperbolicPDE, source_term::AbstractSourceTerm) where {D, M, T, WF, GM, BC}
    N = pg.meta.N
    ensure_shared_capacity!(pg, N)
    
    merged = pg.shared.bit_buffer
    fill!(view(merged, 1:N), false)

    write_idx = 0 
    pos  = get_positions(pg)
    rhos = pg.rhos
    vols = pg.core.volumes
    tags = pg.core.tags
    is_boundary = pg.core.is_boundary
    
    min_dist_thresh = hasproperty(pg, :min_dist) ? pg.min_dist : pg.meta.dx[1] * T(0.25)
    
    bins = pg.bins
    coarse_dims = bins.coarse_dims
    ci = CartesianIndices(coarse_dims)
    li = LinearIndices(coarse_dims)
    window = CartesianIndices(ntuple(_ -> -1:1, Val(D)))
    
    L_vec = pg.meta.maxs .- pg.meta.mins
    mins = pg.meta.mins
    maxs = pg.meta.maxs
    
    is_periodic = (BC == :periodic)

    for i in 1:N
        if merged[i]; continue; end
        
        # 1. Meshfree Out-of-Bounds Check (Replaces fine_type)
        is_inside = true
        for d in 1:D
            if pos[i][d] < mins[d] || pos[i][d] > maxs[d]
                is_inside = false; break;
            end
        end
        if !is_periodic && !is_inside
            continue # Skip and effectively delete out-of-bounds particles
        end
        
        write_idx += 1
        best_j = -1
        min_dist_found = min_dist_thresh 
        
        bin_idx = get_flat_bin_index(pos[i], bins.mins, bins.coarse_size, coarse_dims)
        cart_idx = ci[bin_idx]
        
        for offset in window
            nb_cart = cart_idx + offset
            
            if is_periodic
                wrapped_cart = map((nc, cd) -> mod1(nc, cd), Tuple(nb_cart), coarse_dims)
                nb_bin_idx = li[CartesianIndex(wrapped_cart)]
            else
                if !checkbounds(Bool, li, nb_cart); continue; end
                nb_bin_idx = li[nb_cart]
            end
            
            j = bins.head[nb_bin_idx]
            while j > 0
                # STRICT TAG CHECK: Only merge identical tags!
                if j > i && !merged[j] && tags[i] == tags[j]
                    dist_vec = is_periodic ? getPeriodicDistance(pos, i, j, L_vec, pg.meta.L_inv) : getEuclideanDistance(pos, i, j)
                    dist = sqrt(sum(abs2, dist_vec))
                    
                    if dist < min_dist_found
                        min_dist_found = dist
                        best_j = j
                    end
                end
                j = bins.next[j]
            end
        end
        
        if best_j != -1
            V_i, V_j = vols[i], vols[best_j]
            
            pos[write_idx] = T(0.5) * (pos[i] + pos[best_j])
            rhos[write_idx] = compute_merged_state(rhos, i, best_j, V_i, V_j, eq, source_term)
            vols[write_idx] = V_i + V_j
            
            # Inherit topology from the parent
            tags[write_idx] = tags[i]
            is_boundary[write_idx] = is_boundary[i]
            
            merged[best_j] = true
        else
            if i != write_idx
                pos[write_idx]  = pos[i]
                rhos[write_idx] = rhos[i]
                vols[write_idx] = vols[i]
                tags[write_idx] = tags[i]
                is_boundary[write_idx] = is_boundary[i]
            end
        end
    end
    
    pg.meta.N = write_idx
    return nothing
end

function _split_particles!(pg::ParticleGrid{D, M, T}, eq::HyperbolicPDE, source_term::AbstractSourceTerm) where {D, M, T}
    # Future Logic: Count how many particles need to split based on large neighbor distances
    num_new = 0 
    
    # ... (Your future gap detection logic here) ...
    
    if num_new == 0
        return nothing
    end
    
    N_curr = pg.meta.N
    N_total = N_curr + num_new
    ensure_capacity!(pg, N_total)
    
    pos = get_positions(pg)
    rhos = pg.rhos
    is_bnd = pg.core.is_boundary
    tags = pg.core.tags
    vols = pg.core.volumes
    
    write_idx = N_curr
    
    # When a gap is found and you spawn a new particle:
    # 1. Find the parent particle (i) and the distant neighbor (j)
    # 2. Place the new particle exactly in the middle:
    #    pos[write_idx] = T(0.5) * (pos[i] + pos[j])
    # 3. Interpolate state:
    #    rhos[write_idx] = compute_split_state(rhos, i, j, eq, source_term)
    
    # 4. TOPOLOGY INHERITANCE:
    #    tags[write_idx] = tags[i]
    #    is_bnd[write_idx] = is_bnd[i]
    #    vols[write_idx] = T(0.5) * vols[i]
    
    pg.meta.N = N_total
    build_global_bins!(pg) 
    
    return nothing
end