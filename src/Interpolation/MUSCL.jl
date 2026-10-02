export MUSCL

# =========================================================================
# DIVERGENCE ORDER DISPATCH
# =========================================================================
@inline _resolve_div_idx(::Val{0}, p_order) = max(2, p_order)
@inline _resolve_div_idx(::Val{DO}, p_order) where {DO} = max(2, DO)

@inline function (::ConstantReconstruction)(
    nb_slice, dist_all, w_all, df_flux, df_scratch; scale::T
) where {T}
    return zero(T)
end

# =========================================================================
# MUSCL STRUCTURE
# =========================================================================

"""
    MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, INTERPS, L, NF}

A divergence interpolator executing MUSCL-type interface reconstruction and flux evaluation.
Dynamically tracks particle polynomial limits strictly via `ParticleGridCore`.
"""
struct MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, INTERPS, L, NF} <: DivergenceInterpolator
    interpolators::INTERPS
    limiter::L
    flux::NF
    gradients::Vector{SVector{B_LEN, State{M, T}}} 
end

@inline _extract_order(::MUSCL{D, M, T, B_LEN, MAX_ORDER}) where {D, M, T, B_LEN, MAX_ORDER} = MAX_ORDER

function MUSCL(
    ::Type{T}, dimension::Int, M::Int, max_order::Int;
    div_order::Int=0, limiter=NoLimiter(), flux=RusanovFlux()
) where {T}
    @assert max_order >= 1 "MUSCL must have a maximum order of at least 1."
    @assert div_order >= 0 "Divergence order only supports 0 (adaptive) or positive values!"
    if div_order > max_order; div_order = max_order; end
    
    max_degree = max(1, max_order - 1) 
    B_LEN_VAL = basis_length(Val(dimension), Val(max_degree))
    B_LEN = typeof(B_LEN_VAL).parameters[1] 
    
    interps = ntuple(Val(max_order)) do k
        k == 1 ? ConstantReconstruction() : Interpolator{dimension, k - 1, 1}()
    end
    
    return MUSCL{dimension, M, T, B_LEN, max_order, div_order, typeof(interps), typeof(limiter), typeof(flux)}(
        interps, limiter, flux, SVector{B_LEN,State{M, T}}[]
    )
end

function update_size!(muscl::MUSCL{D, M, T, B_LEN, MAX_ORDER}, N_particles::Int) where {D, M, T, B_LEN, MAX_ORDER}
    ensure_capacity!(muscl.gradients, N_particles)
    fill!(muscl.gradients, zero(eltype(muscl.gradients)))
    return nothing
end

# =========================================================================
# PRE-GATHER PASS: CALCULATE AND STORE GRADIENTS
# =========================================================================

function update_content!(
    muscl::MUSCL{D, M, T, B_LEN, MAX_ORDER}, i::Int, f_i::State{M, T}, nb_slice::UnitRange{Int},
    pg::ParticleGrid{D, M, T}, ib::InteractionBuffer{D, M, T}
) where {D, M, T, B_LEN, MAX_ORDER}
    
    num_nb = length(nb_slice)
    p_order = pg.core.particle_orders[i]

    req_deg = p_order - 1
    req_nb = req_deg == 0 ? 0 : typeof(basis_length(Val(D), Val(req_deg))).parameters[1]
    
    # Dynamically clamp spatial order in core arrays if neighborhood is starved
    if i < 0 || num_nb < req_nb
        pg.core.particle_orders[i] = 1 
        p_order = 1
    end

    dist_all = get_distances(pg)
    w_all = get_weights(pg)

    scale_val = minimum(pg.meta.dx)
    raw_grad = dispatch_interpolator(muscl.interpolators, p_order, nb_slice, dist_all, w_all, ib.df, scale_val, Val(B_LEN), State{M, T})
    
    if p_order == 1
        muscl.gradients[i] = raw_grad 
    else
        muscl.gradients[i] = _limit_slopes(muscl.limiter, raw_grad, nb_slice, f_i, ib.f, pg, dist_all, Val(MAX_ORDER - 1))
    end
    return nothing
end

# =========================================================================
# FLUX PASS: RECONSTRUCT INTERFACES AND COMPUTE DIVERGENCE
# =========================================================================

function (muscl::MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER})(
    eq::HyperbolicPDE, i::Int, f_i::State{M, T}, nb_slice::UnitRange{Int},       
    pg::ParticleGrid{D, M, T}, ib::InteractionBuffer{D, M, T}    
) where {D, M, T, B_LEN, MAX_ORDER, DIV_ORDER}

    if isempty(nb_slice)
        return zero(State{M, T})
    end

    grad_i = muscl.gradients[i]
    F_i    = flux(eq, f_i)
    dist_all = get_distances(pg)
    nb_indices = pg.neighbor.indices
    
    eff_order_i = pg.core.particle_orders[i]
    
    @inbounds for global_idx in nb_slice
        dist_k = dist_all[global_idx]
        f_j    = ib.f[global_idx]
        j_idx  = nb_indices[global_idx]
        
        grad_j = muscl.gradients[j_idx]
        eff_order_j = pg.core.particle_orders[j_idx]
        
        p_interface_i_raw = build_basis(Val(MAX_ORDER-1),  T(0.5) * dist_k)
        p_interface_j_raw = build_basis(Val(MAX_ORDER-1), -T(0.5) * dist_k)

        # Standard MUSCL Interface blending (safe min intersection of allowed orders)
        interface_order_i = min(eff_order_i, eff_order_j)
        interface_order_j = interface_order_i
        
        p_interface_i = mask_basis(p_interface_i_raw, interface_order_i, Val(D))
        p_interface_j = mask_basis(p_interface_j_raw, interface_order_j, Val(D))

        fij = f_i + sum(grad_i .* p_interface_i)
        fji = f_j + sum(grad_j .* p_interface_j)
        
        F_ij = flux(eq, fij)
        F_ji = flux(eq, fji)
        
        f_L, f_R, F_L, F_R = sort_flux(fij, fji, F_ij, F_ji, dist_k)
        F_num = muscl.flux(f_L, f_R, F_L, F_R, eq)
        nc_jump = evaluate_nc_jump(eq, f_L, f_R, dist_k)
        
        ib.df_flux[global_idx] = F_num - F_i + nc_jump
    end

    div_idx = _resolve_div_idx(Val(DIV_ORDER), eff_order_i)
    
    div = compute_dynamic_divergence(
        muscl.interpolators, div_idx, 
        nb_slice, dist_all, get_weights(pg), 
        ib.df_flux, ib.df_scratch, pg.meta.dx
    )
    
    return T(2.0) * div
end