# =========================================================================
# DYNAMIC DISPATCH ROUTER (Zero-Allocation)
# =========================================================================

# Helper to pad smaller gradients with zeros up to the MAX basis length
@inline _pad_grad(g::SVector{L, T}, ::Val{MAX_L}) where {L, MAX_L, T} = SVector{MAX_L, T}(ntuple(i -> i <= L ? g[i] : zero(T), Val(MAX_L)))

# Dispatch for Order 1 (Constant / Euler) -> Returns all zeros
@inline _compute_raw_grad(::ConstantReconstruction, nb_slice, dist_all, w_all, df, scale, ::Val{MAX_B_LEN}, ::Type{State{M}}) where {MAX_B_LEN, M} = zero(SVector{MAX_B_LEN, State{M}})

# Dispatch for Order > 1 (MLS Interpolator)
@inline function _compute_raw_grad(interp::Interpolator, nb_slice, dist_all, w_all, df, scale, ::Val{MAX_B_LEN}, ::Type{State{M}}) where {MAX_B_LEN, M}
    raw = interp(nb_slice, dist_all, w_all, df; scale=scale)
    return _pad_grad(raw, Val(MAX_B_LEN))
end
@inline _get_interface_orders(::MOODStrategy, oi, oj) = (min(oi, oj), min(oi, oj)) # Default EPD1/EPD2 behavior
@inline _get_interface_orders(::EPD0, oi, oj) = (oi, oj) # True EPD0 Asymmetric behavior
# Generated function creates a highly optimized if-elseif chain at compile time 
# so we can index a Tuple using a runtime variable (`order`) without type instability!
@generated function dispatch_interpolator(interps::Tuple, order::Int, args...)
    N = length(interps.parameters)
    expr = :(error("Order out of bounds"))
    for i in N:-1:1
        expr = :(order == $i ? _compute_raw_grad(interps[$i], args...) : $expr)
    end
    return expr
end

# Dynamically drops the MLS degree of the divergence operator to prevent Runge's Phenomenon
@generated function compute_dynamic_divergence(interps::Tuple, div_idx::Int, nb_slice, dist_all, w_all, dfFlux, df_scratch, scale)
    N = length(interps.parameters)
    expr = :(interps[$N](nb_slice, dist_all, w_all, dfFlux, df_scratch; scale=scale))
    # Build if/else chain from N-1 down to 2 (Since Index 1 is Constant/Zero, we stop at 2 for Linear/1st-Deriv)
    for i in (N-1):-1:2
        expr = :(div_idx == $i ? interps[$i](nb_slice, dist_all, w_all, dfFlux, df_scratch; scale=scale) : $expr)
    end
    return expr
end

# =========================================================================
# DIVERGENCE ORDER DISPATCH (Zero-Overhead Compile-Time Logic)
# =========================================================================
@inline _resolve_div_idx(::Val{0}, p_order) = max(2, p_order)
@inline _resolve_div_idx(::Val{DO}, p_order) where {DO} = max(2, DO)

# =========================================================================
# CONSTRUCTOR & SIZING
# =========================================================================

# 1. Add the missing functor for ConstantReconstruction
@inline function (::ConstantReconstruction)(
    nb_slice, dist_all, w_all, dfFlux, df_scratch; scale
)
    return 0.0
end

# 2. Update the MUSCL Constructor
function MUSCL(
    dimension::Int, M::Int, max_order::Int;
    div_order::Int=0, limiter=NoLimiter(), numericalFlux=RusanovFlux(), mood=NoMOOD()
)
    @assert max_order >= 1 "MUSCL must have a maximum order of at least 1."
    @assert div_order >= 0 "Divergence order only supports 0 (adaptive) or positive values!"
    if div_order > max_order; div_order = max_order; end
    
    # Fake the degree to at least 1 for the zero-padding type stability 
    # (Order 1 gradients will just remain safely zeroed out)
    max_degree = max(1, max_order - 1) 
    B_LEN_VAL = basis_length(Val(dimension), Val(max_degree))
    B_LEN = typeof(B_LEN_VAL).parameters[1] 
    
    interps = ntuple(Val(max_order)) do k
        k == 1 ? ConstantReconstruction() : Interpolator{dimension, k - 1, 1}()
    end
    
    return MUSCL{dimension, M, B_LEN, max_order, div_order, typeof(mood), typeof(interps), typeof(limiter), typeof(numericalFlux)}(
        interps, limiter, numericalFlux, mood, SVector{B_LEN,State{M}}[], Int[], Bool[]
    )
end

function update_size!(muscl::MUSCL{D, M, B_LEN, MAX_ORDER}, N_particles::Int) where {D, M, B_LEN, MAX_ORDER}
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

function update_content!(
    muscl::MUSCL{D, M, B_LEN, MAX_ORDER}, i::Int, f_i::State{M}, nb_slice::UnitRange{Int},
    pg::ParticleGrid{D}, ib::InteractionBuffer{D, M}
) where {D, M, B_LEN, MAX_ORDER}
    
    num_nb = length(nb_slice)
    p_order = muscl.particle_orders[i]

    # Dynamically determine minimum neighbors based on current order
    req_deg = p_order - 1
    req_nb = req_deg == 0 ? 0 : typeof(basis_length(Val(D), Val(req_deg))).parameters[1]
    
    if i < 0 || num_nb < req_nb
        muscl.particle_orders[i] = 1 # Force drop to Order 1 if physically starved
        p_order = 1
    end

    dist_all = get_distances(pg)
    w_all = get_weights(pg)

    # Clean, zero-allocation dispatch to the correct interpolator!
    scale_val = minimum(pg.meta.dx)
    raw_grad = dispatch_interpolator(muscl.interpolators, p_order, nb_slice, dist_all, w_all, ib.df, scale_val, Val(B_LEN), State{M})
    
    if p_order == 1
        muscl.gradients[i] = raw_grad # Order 1 is strictly 0.0, no limiters needed
    else
        muscl.gradients[i] = _limit_slopes(muscl.limiter, raw_grad, nb_slice, f_i, ib.f, pg, dist_all, Val(MAX_ORDER - 1))
    end
    return nothing
end

# =========================================================================
# FLUX PASS: RECONSTRUCT INTERFACES AND COMPUTE DIVERGENCE
# =========================================================================

function (muscl::MUSCL{D, M, B_LEN, MAX_ORDER, DIV_ORDER})(
    eq::HyperbolicPDE, i::Int, f_i::State{M}, nb_slice::UnitRange{Int},       
    pg::ParticleGrid{D}, ib::InteractionBuffer{D, M}    
) where {D, M, B_LEN, MAX_ORDER, DIV_ORDER}

    if isempty(nb_slice)
        return zero(State{M})
    end

    grad_i = muscl.gradients[i]
    F_i    = flux(eq, f_i)
    dist_all = get_distances(pg)
    nb_indices = pg.neighbor.indices
    
    nb_indices = pg.neighbor.indices
    strategy = muscl.mood.strategy
    
    # Pre-evaluate the effective order for cell i ONCE
    eff_order_i = get_effective_order(strategy, muscl.particle_orders, i, nb_slice, nb_indices)
    
    @inbounds for global_idx in nb_slice
        dist_k = dist_all[global_idx]
        f_j    = ib.f[global_idx]
        j_idx  = nb_indices[global_idx]
        
        grad_j = muscl.gradients[j_idx]
        
        # Evaluate effective order for cell j
        nb_slice_j = pg.neighbor.ranges[j_idx]
        eff_order_j = get_effective_order(strategy, muscl.particle_orders, j_idx, nb_slice_j, nb_indices)
        
        
        p_interface_i_raw = build_basis(Val(MAX_ORDER-1),  0.5 * dist_k)
        p_interface_j_raw = build_basis(Val(MAX_ORDER-1), -0.5 * dist_k)

        interface_order_i, interface_order_j = _get_interface_orders(strategy, eff_order_i, eff_order_j)
        
        p_interface_i = mask_basis(p_interface_i_raw, interface_order_i, Val(D))
        p_interface_j = mask_basis(p_interface_j_raw, interface_order_j, Val(D))

        fij = f_i + sum(grad_i .* p_interface_i)
        fji = f_j + sum(grad_j .* p_interface_j)
        
        F_ij = flux(eq, fij)
        F_ji = flux(eq, fji)
        
        f_L, f_R, F_L, F_R = sort_flux(fij, fji, F_ij, F_ji, dist_k)
        F_num = muscl.numericalFlux(f_L, f_R, F_L, F_R, eq)
        nc_jump = evaluate_nc_jump(eq, f_L, f_R, dist_k)
        
        ib.dfFlux[global_idx] = F_num - F_i + nc_jump
    end

    p_order = muscl.particle_orders[i]
    
    div_idx = _resolve_div_idx(Val(DIV_ORDER), p_order)
    
    div = compute_dynamic_divergence(
        muscl.interpolators, div_idx, 
        nb_slice, dist_all, get_weights(pg), 
        ib.dfFlux, ib.df_scratch, pg.meta.dx
    )
    
    return 2.0 * div
end