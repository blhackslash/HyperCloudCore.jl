export MUSCL

struct ConstantReconstruction end

"""
    MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, MOOD, INTERPS, L, NF}

A divergence interpolator executing MUSCL-type interface reconstruction and flux evaluation.

# Fields
- `interpolators`: A tuple of MLS interpolators instantiated for orders up to `MAX_ORDER`.
- `limiter::L`: The selected slope limiter configuration.
- `flux::NF`: The numerical interface flux function.
- `mood::MOOD`: The multidimensional optimal order detection strategy.
- `gradients`: A pre-allocated vector storing the computed and limited gradients for each particle.
- `particle_orders`: Tracks the dynamically adjusted spatial order of each particle.
"""
struct MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, MOOD, INTERPS, L, NF} <: DivergenceInterpolator
    interpolators::INTERPS
    limiter::L
    flux::NF
    mood::MOOD
    gradients::Vector{SVector{B_LEN, State{M, T}}} 
    particle_orders::Vector{Int} 
    mood_triggered::Vector{Bool} 
end

@inline _extract_order(::MUSCL{D, M, T, B_LEN, MAX_ORDER}) where {D, M, T, B_LEN, MAX_ORDER} = MAX_ORDER
# =========================================================================
# DYNAMIC DISPATCH ROUTER (Zero-Allocation)
# =========================================================================

@inline _pad_grad(g::SVector{L, T}, ::Val{MAX_L}) where {L, MAX_L, T} = SVector{MAX_L, T}(ntuple(i -> i <= L ? g[i] : zero(T), Val(MAX_L)))

@inline _compute_raw_grad(::ConstantReconstruction, nb_slice, dist_all, w_all, df, scale, ::Val{MAX_B_LEN}, ::Type{State{M, T}}) where {MAX_B_LEN, M, T} = zero(SVector{MAX_B_LEN, State{M, T}})

@inline function _compute_raw_grad(interp::Interpolator, nb_slice, dist_all, w_all, df, scale, ::Val{MAX_B_LEN}, ::Type{State{M, T}}) where {MAX_B_LEN, M, T}
    raw = interp(nb_slice, dist_all, w_all, df; scale=scale)
    return _pad_grad(raw, Val(MAX_B_LEN))
end

@generated function dispatch_interpolator(interps::Tuple, order::Int, args...)
    N = length(interps.parameters)
    expr = :(error("Order out of bounds"))
    for i in N:-1:1
        expr = :(order == $i ? _compute_raw_grad(interps[$i], args...) : $expr)
    end
    return expr
end

@generated function compute_dynamic_divergence(interps::Tuple, div_idx::Int, nb_slice, dist_all, w_all, df_flux, df_scratch, scale)
    N = length(interps.parameters)
    expr = :(interps[$N](nb_slice, dist_all, w_all, df_flux, df_scratch; scale=scale))
    for i in (N-1):-1:2
        expr = :(div_idx == $i ? interps[$i](nb_slice, dist_all, w_all, df_flux, df_scratch; scale=scale) : $expr)
    end
    return expr
end

# =========================================================================
# DIVERGENCE ORDER DISPATCH
# =========================================================================
@inline _resolve_div_idx(::Val{0}, p_order) = max(2, p_order)
@inline _resolve_div_idx(::Val{DO}, p_order) where {DO} = max(2, DO)

# =========================================================================
# CONSTRUCTOR & SIZING
# =========================================================================

@inline function (::ConstantReconstruction)(
    nb_slice, dist_all, w_all, df_flux, df_scratch; scale::T
) where {T}
    return zero(T)
end

"""
    MUSCL(::Type{T}, dimension, M, max_order; div_order=0, limiter=NoLimiter(), flux=RusanovFlux(), mood=NoMOOD())

Constructs a `MUSCL` divergence evaluator system.

# Details
- Asserts that `max_order` is at least 1 and that `div_order` is 0 (adaptive mode) or a positive integer.
- Automatically instantiates `ConstantReconstruction` for 1st-order schemes and dynamically builds generalized `Interpolator` instances for all higher orders up to `max_order`.
"""
function MUSCL(
    ::Type{T}, dimension::Int, M::Int, max_order::Int;
    div_order::Int=0, limiter=NoLimiter(), flux=RusanovFlux(), mood=NoMOOD()
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
    
    return MUSCL{dimension, M, T, B_LEN, max_order, div_order, typeof(mood), typeof(interps), typeof(limiter), typeof(flux)}(
        interps, limiter, flux, mood, SVector{B_LEN,State{M, T}}[], Int[], Bool[]
    )
end

function update_size!(muscl::MUSCL{D, M, T, B_LEN, MAX_ORDER}, N_particles::Int) where {D, M, T, B_LEN, MAX_ORDER}
    ensure_capacity!(muscl.gradients, N_particles)
    ensure_capacity!(muscl.particle_orders, N_particles)
    ensure_capacity!(muscl.mood_triggered, N_particles)
    fill!(muscl.gradients, zero(eltype(muscl.gradients)))
    fill!(muscl.particle_orders, MAX_ORDER)
    return nothing
end

# =========================================================================
# PRE-GATHER PASS: CALCULATE AND STORE GRADIENTS
# =========================================================================

"""
    update_content!(muscl::MUSCL, i, f_i, nb_slice, pg, ib)

Executes the pre-gather pass to calculate, limit, and store the MUSCL gradients for a single particle.

# Details
- Validates the active neighborhood size against the required polynomial basis length. If the stencil is insufficient, the local particle order drops to 1.
- Dynamically dispatches to the correct internal interpolator based on the resolved `particle_order`.
- Replaces the raw gradient with the slope-limited gradient (unless the order is 1) and stores it in the internal `gradients` buffer.
"""
function update_content!(
    muscl::MUSCL{D, M, T, B_LEN, MAX_ORDER}, i::Int, f_i::State{M, T}, nb_slice::UnitRange{Int},
    pg::ParticleGrid{D, M, T}, ib::InteractionBuffer{D, M, T}
) where {D, M, T, B_LEN, MAX_ORDER}
    
    num_nb = length(nb_slice)
    p_order = muscl.particle_orders[i]

    req_deg = p_order - 1
    req_nb = req_deg == 0 ? 0 : typeof(basis_length(Val(D), Val(req_deg))).parameters[1]
    
    if i < 0 || num_nb < req_nb
        muscl.particle_orders[i] = 1 
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
"""
    (muscl::MUSCL)(eq, i, f_i, nb_slice, pg, ib)

The primary functor execution for computing the `MUSCL` divergence update.

# Details
- Iterates over the neighborhood slice to evaluate MUSCL-reconstructed left and right interface states `fij` and `fji`.
- Masks the spatial basis vectors down to the dynamically calculated `interface_order` derived from the target MOOD strategy.
- Solves the numerical interface flux using the configured `muscl.flux` evaluator and adds any non-conservative jump corrections.
- Calculates and returns the final bounded divergence scaled by 2.0 utilizing the dynamically resolved divergence order.
"""
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
    
    strategy = muscl.mood.strategy
    is_nomood = muscl.mood.criterion isa NoMOOD
    effective_orders = pg.shared.int_buffer
    
    eff_order_i = is_nomood ? MAX_ORDER : effective_orders[i]
    
    @inbounds for global_idx in nb_slice
        dist_k = dist_all[global_idx]
        f_j    = ib.f[global_idx]
        j_idx  = nb_indices[global_idx]
        
        grad_j = muscl.gradients[j_idx]
        eff_order_j = is_nomood ? MAX_ORDER : effective_orders[j_idx]
        
        p_interface_i_raw = build_basis(Val(MAX_ORDER-1),  T(0.5) * dist_k)
        p_interface_j_raw = build_basis(Val(MAX_ORDER-1), -T(0.5) * dist_k)

        interface_order_i, interface_order_j = _get_interface_orders(strategy, eff_order_i, eff_order_j)
        
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

    p_order = muscl.particle_orders[i]
    div_idx = _resolve_div_idx(Val(DIV_ORDER), p_order)
    
    div = compute_dynamic_divergence(
        muscl.interpolators, div_idx, 
        nb_slice, dist_all, get_weights(pg), 
        ib.df_flux, ib.df_scratch, pg.meta.dx
    )
    
    return T(2.0) * div
end