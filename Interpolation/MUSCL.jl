

# =========================================================================
# PRE-GATHER PASS: CALCULATE AND STORE GRADIENTS
# =========================================================================

function update_size!(muscl::MUSCL, N_particles::Int)
    ensure_capacity!(muscl.gradients, N_particles)
    # Ensure initialized to zero to prevent NaNs on boundary particles
    fill!(muscl.gradients, zero(eltype(muscl.gradients))) 
    return nothing
end

function update_content!(
    muscl::MUSCL{D, M, B_LEN, ORDER}, i::Int, f_i::State{M}, nb_slice::UnitRange{Int},
    pg::ParticleGrid{D}, ib::InteractionBuffer{D, M}
) where {D, M, B_LEN, ORDER}
    
    num_nb = length(nb_slice)
    if i < 0 || num_nb < B_LEN
        muscl.gradients[abs(i)] = zero(SVector{B_LEN, State{M}})
        return nothing
    end

    dist_all = get_distances(pg)
    w_all = get_weights(pg)

    # Zero-copy raw gradient! (Notice we pass nb_slice instead of 1:num_nb)
    raw_grad = muscl.interpolator(
        nb_slice, dist_all, w_all, ib.f; scale = minimum(pg.meta.dx)
    )
    
    # Limit and store globally
    limited_grad = _limit_slopes(muscl.limiter, raw_grad, nb_slice, f_i, ib.f, pg, dist_all)
    muscl.gradients[i] = limited_grad
    return nothing
end

# =========================================================================
# FLUX PASS: RECONSTRUCT INTERFACES AND COMPUTE DIVERGENCE
# =========================================================================

function (muscl::MUSCL{D, M, B_LEN, ORDER})(
    eq::HyperbolicPDE, i::Int, f_i::State{M}, nb_slice::UnitRange{Int},       
    pg::ParticleGrid{D}, ib::InteractionBuffer{D, M}    
) where {D, M, B_LEN, ORDER}

    if length(nb_slice) < B_LEN
        return zero(State{M})
    end

    grad_i = muscl.gradients[i]
    F_i    = flux(eq, f_i)
    dist_all = get_distances(pg)
    
    @inbounds for global_idx in nb_slice
        dist_k = dist_all[global_idx]
        f_j    = ib.f[global_idx]
        grad_j = muscl.gradients[pg.neighbor.indices[global_idx]]
        
        p_interface_i = build_basis(Val(ORDER),  0.5 * dist_k)
        p_interface_j = build_basis(Val(ORDER), -0.5 * dist_k)

        fij = f_i + sum(grad_i .* p_interface_i)
        fji = f_j + sum(grad_j .* p_interface_j)
        
        F_ij = flux(eq, fij)
        F_ji = flux(eq, fji)
        
        f_L, f_R, F_L, F_R = sort_flux(fij, fji, F_ij, F_ji, dist_k)
        F_num = muscl.numericalFlux(f_L, f_R, F_L, F_R, eq)
        nc_jump = evaluate_nc_jump(eq, f_L, f_R, dist_k)
        
        # Write directly to the global slot
        ib.dfFlux[global_idx] = F_num - F_i + nc_jump
    end
    
    # Zero-copy interpolation
    div = muscl.interpolator(
        nb_slice, dist_all, get_weights(pg), ib.dfFlux, ib.df_scratch; scale = pg.meta.dx
    )
    
    return 2.0 * div
end