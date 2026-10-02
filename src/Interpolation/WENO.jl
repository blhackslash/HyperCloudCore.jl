export WENO

"""
    WENO{D, M, T, MAX_ORDER, DIV_ORDER, INTERPS} <: DivergenceInterpolator
    WENO(::Type{T}, dimension::Int, M::Int, order::Int; div_order::Int=0)
    (weno::WENO)(eq, i, f_i, nb_slice, pg, ib)

Weighted Essentially Non-Oscillatory (WENO) divergence interpolator for evaluating flux derivatives in hyperbolic PDEs. It dynamically blends central and directional stencils based on local solution smoothness to prevent oscillations near discontinuities.

# Constructors

    WENO(::Type{T}, dimension::Int, M::Int, order::Int; div_order::Int=0)

- `T`: The numeric type (e.g., `Float64`).
- `dimension`: Spatial dimension.
- `M`: Number of state components.
- `order`: Maximum interpolation order, which must be `≥ 2` to support the second derivatives required for evaluating smoothness indicators.
- `div_order`: Target order (defaults to `0`, dynamically bound to the available particle order).

# Callable / Functor

    (weno::WENO)(eq, i, f_i, nb_slice, pg, ib) -> State{M, T}

Computes the divergence of the flux at particle `i`. The execution follows these steps:
- Evaluates a full central stencil using all available neighbors within the interaction buffer.
- Evaluates directional one-sided stencils for each spatial dimension, selecting neighbors based on the sign of the local advection velocity.
- Computes smoothness indicators for both the central and directional stencils using locally scaled spatial derivatives.
- Calculates non-linear weights to heavily penalize stencils crossing discontinuities.
- Blends the central and directional interpolations using these weights to form the final stable divergence computation.

# Fields
- `interpolators`: Pre-allocated tuple of `Interpolator` instances ranging up to `MAX_ORDER` to support dynamic order degradation.
"""
struct WENO{D, M, T, MAX_ORDER, DIV_ORDER, INTERPS} <: DivergenceInterpolator
    interpolators::INTERPS
end

@inline update_size!(::WENO, ::Int) = nothing
@inline update_content!(::WENO, args...) = nothing
@inline _extract_order(::WENO{D, M, T, MAX_ORDER}) where {D, M, T, MAX_ORDER} = MAX_ORDER

function WENO(::Type{T}, dimension::Int, M::Int, order::Int; div_order::Int=0) where {T}
    @assert order >= 2 "WENO requires order >= 2 for second derivatives."
    if div_order > order; div_order = order; end
    
    interps = ntuple(Val(order)) do k
        Interpolator{dimension, k, 1}()
    end
    
    return WENO{dimension, M, T, order, div_order, typeof(interps)}(interps)
end

function (weno::WENO{D, M, T, MAX_ORDER, DIV_ORDER, INTERPS})(
    eq::HyperbolicPDE, i::Int, f_i::State{M, T}, nb_slice::UnitRange{Int},       
    pg::ParticleGrid{D, M, T}, ib::InteractionBuffer{D, M, T}    
) where {D, M, T, MAX_ORDER, DIV_ORDER, INTERPS}

    vel = velocity(eq, f_i, D)
    dist_all = get_distances(pg)
    w_all = get_weights(pg)

    num_nb = length(nb_slice)
    p_order = pg.core.particle_orders[i]

    while p_order > 1
        req_nb = typeof(basis_length(Val(D), Val(p_order))).parameters[1]
        if num_nb >= req_nb; break; end
        p_order -= 1
    end

    if num_nb < typeof(basis_length(Val(D), Val(p_order))).parameters[1]
        return zero(State{M, T})
    end

    scale_val = minimum(pg.meta.dx)
    dx2 = scale_val^2
    dx4 = dx2^2
    e_tol = T(1e-12)

    # --- 1. Central Stencil Calculation ---
    @inbounds for global_idx in nb_slice; ib.mask[global_idx] = true; end
    
    div_idx_C = DIV_ORDER == 0 ? p_order : min(DIV_ORDER, p_order)
    B_LEN_C_VAL = typeof(basis_length(Val(D), Val(div_idx_C))).parameters[1]
    
    resC_raw = dispatch_interpolator(
        weno.interpolators, div_idx_C, 
        nb_slice, dist_all, w_all, ib.df, ib.mask, scale_val, Val(B_LEN_C_VAL), State{M, T}
    )
    resC = SVector{D, State{M, T}}(ntuple(d -> resC_raw[d], Val(D)))

    smoothC = zero(State{M, T})
    @inbounds for k in 1:length(resC)
        weight = k <= D ? dx2 : dx4 
        smoothC += (resC[k] .* resC[k]) * weight
    end
    betaC = T(0.5) ./ ((smoothC .+ e_tol) .^ 2)

    div_total = zero(State{M, T})

    # --- 2. Directional Stencils (One per spatial dimension) ---
    for d in 1:D
        stencil_size = 0
        use_left = vel[d] > zero(T)
        
        @inbounds for global_idx in nb_slice
            dist_k = dist_all[global_idx][d]
            if (use_left && dist_k < zero(T)) || (!use_left && dist_k >= zero(T))
                ib.mask[global_idx] = true
                stencil_size += 1
            else
                ib.mask[global_idx] = false
            end
        end

        p_order_d = p_order
        while p_order_d > 1
            req_nb = typeof(basis_length(Val(D), Val(p_order_d))).parameters[1]
            if stencil_size >= req_nb; break; end
            p_order_d -= 1
        end

        if stencil_size < typeof(basis_length(Val(D), Val(p_order_d))).parameters[1]
            div_total += resC[d] * vel[d]
            continue
        end

        div_idx_S = DIV_ORDER == 0 ? p_order_d : min(DIV_ORDER, p_order_d)
        B_LEN_D_VAL = typeof(basis_length(Val(D), Val(div_idx_S))).parameters[1]
        
        resS_raw = dispatch_interpolator(
            weno.interpolators, div_idx_S, 
            nb_slice, dist_all, w_all, ib.df, ib.mask, scale_val, Val(B_LEN_D_VAL), State{M, T}
        )
        resS = SVector{D, State{M, T}}(ntuple(k -> resS_raw[k], Val(D)))

        smoothS = zero(State{M, T})
        @inbounds for k in 1:length(resS)
            weight = k <= D ? dx2 : dx4
            smoothS += (resS[k] .* resS[k]) * weight
        end
        betaS = T(0.5) ./ ((smoothS .+ e_tol) .^ 2)

        # --- 3. Apply Non-Linear Weights ---
        sum_beta = betaS .+ betaC
        
        wS = betaS ./ sum_beta
        wC = betaC ./ sum_beta

        div_d = wS .* resS[d] .+ wC .* resC[d]
        div_total += div_d * vel[d]
    end

    return div_total
end