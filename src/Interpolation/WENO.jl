export WENO

"""
    WENO{D, M, T, I <: Interpolator}

A divergence interpolator executing a Weighted Essentially Non-Oscillatory (WENO) reconstruction scheme.

# Fields
- `order::Int`: The numerical order of the interpolation.
- `interpolator::I`: The underlying `Interpolator` instance.
"""
struct WENO{D, M, T, I <: Interpolator} <: DivergenceInterpolator
    order::Int
    interpolator::I
end
# =========================================================================
# STATELESS WENO API
# =========================================================================

@inline update_size!(::WENO, ::Int) = nothing
@inline update_content!(::WENO, args...) = nothing
@inline _extract_order(g::WENO) = g.order
"""
    WENO(::Type{T}, dimension::Int, M::Int, order::Int; flux=RusanovFlux())

Constructs a `WENO` divergence evaluator.

# Details
- Asserts that the `order` is at least 2, which is strictly required to capture the second derivatives used for the smoothness indicators.
- Initializes the corresponding moving least squares interpolator for the spatial evaluations.
"""
function WENO(::Type{T}, dimension::Int, M::Int, order::Int; flux::NumericalFluxFunction = RusanovFlux()) where {T}
    @assert order >= 2 "WENO requires order >= 2 for second derivatives."
    
    interpolator = Interpolator{dimension, order, 1}()
    return WENO{dimension, M, T, typeof(interpolator)}(order, interpolator)
end

# =========================================================================
# UNIVERSAL N-DIMENSIONAL WENO FUNCTOR
# =========================================================================
"""
    (weno::WENO)(eq::HyperbolicPDE, i::Int, f_i, nb_slice, pg, ib)

The primary functor execution for computing the N-dimensional WENO divergence update.

# Details
- Evaluates a fully centered neighborhood stencil to calculate the baseline central derivative (`resC`) and its associated smoothness indicator (`smoothC`).
- The central smoothness indicator uses a scale weighting of `dx^2` for the first spatial dimensions and `dx^4` for higher-order basis terms.
- Iterates over each spatial dimension to construct a strictly directional, upwind-biased stencil determined by the sign of the local velocity.
- Computes the directional smoothness indicator (`smoothS`). If the constructed directional stencil lacks sufficient neighbors, it automatically falls back to utilizing the central derivative for that specific dimension.
- Dynamically blends the central and directional stencils using non-linear weights (`wC` and `wS`) that are inversely proportional to their computed smoothness indicators.
"""
function (weno::WENO{D, M, T, I})(
    eq::HyperbolicPDE, i::Int, f_i::State{M, T}, nb_slice::UnitRange{Int},       
    pg::ParticleGrid{D, M, T}, ib::InteractionBuffer{D, M, T}    
) where {D, M, T, I}

    vel = velocity(eq, f_i, D)
    interp = weno.interpolator
    dist_all = get_distances(pg)
    w_all = get_weights(pg)

    num_nb = length(nb_slice)
    if num_nb < weno.order; return zero(State{M, T}); end

    scale_val = minimum(pg.meta.dx)
    dx2 = scale_val^2
    dx4 = dx2^2
    e_tol = T(1e-12)

    # --- 1. Central Stencil Calculation ---
    @inbounds for global_idx in nb_slice; ib.mask[global_idx] = true; end
    
    resC = interp(nb_slice, dist_all, w_all, ib.df, ib.mask; scale=scale_val)

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

        if stencil_size < weno.order
            div_total += resC[d] * vel[d]
            continue
        end

        resS = interp(nb_slice, dist_all, w_all, ib.df, ib.mask; scale=scale_val)

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