export WENO

"""
    WENO{D, M, T, MAX_ORDER, INTERPS}

A divergence interpolator executing a Weighted Essentially Non-Oscillatory (WENO) reconstruction scheme.
Dynamically tracks polynomial limits via `ParticleGridCore` and supports universal order degradation.
"""
struct WENO{D, M, T, MAX_ORDER, INTERPS} <: DivergenceInterpolator
    interpolators::INTERPS
end

# =========================================================================
# STATELESS WENO API
# =========================================================================

@inline update_size!(::WENO, ::Int) = nothing
@inline update_content!(::WENO, args...) = nothing
@inline _extract_order(::WENO{D, M, T, MAX_ORDER}) where {D, M, T, MAX_ORDER} = MAX_ORDER

# =========================================================================
# UNIVERSAL N-DIMENSIONAL WENO FUNCTOR
# =========================================================================

function (weno::WENO{D, M, T, MAX_ORDER, INTERPS})(
    eq::HyperbolicPDE, i::Int, f_i::State{M, T}, nb_slice::UnitRange{Int},       
    pg::ParticleGrid{D, M, T}, ib::InteractionBuffer{D, M, T}    
) where {D, M, T, MAX_ORDER, INTERPS}

    vel = velocity(eq, f_i, D)
    dist_all = get_distances(pg)
    w_all = get_weights(pg)

    num_nb = length(nb_slice)
    p_order = pg.core.particle_orders[i]

    # Dynamically degrade order if the central neighborhood is starved
    while p_order > 1
        req_nb = typeof(basis_length(Val(D), Val(p_order))).parameters[1]
        if num_nb >= req_nb
            break
        end
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
    
    # FIX: Degree maps directly to p_order for WENO
    B_LEN_VAL = typeof(basis_length(Val(D), Val(p_order))).parameters[1]
    
    resC_raw = dispatch_interpolator(
        weno.interpolators, p_order, 
        nb_slice, dist_all, w_all, ib.df, ib.mask, scale_val, Val(B_LEN_VAL), State{M, T}
    )
    
    # Truncate to just the required spatial components for the indicator
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
            if stencil_size >= req_nb
                break
            end
            p_order_d -= 1
        end

        if stencil_size < typeof(basis_length(Val(D), Val(p_order_d))).parameters[1]
            div_total += resC[d] * vel[d]
            continue
        end

        # FIX: Degree maps directly to p_order_d for the directional stencil
        B_LEN_D_VAL = typeof(basis_length(Val(D), Val(p_order_d))).parameters[1]
        resS_raw = dispatch_interpolator(
            weno.interpolators, p_order_d, 
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