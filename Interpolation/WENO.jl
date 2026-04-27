# =========================================================================
# STATELESS WENO API
# =========================================================================

@inline update_size!(::WENO, ::Int) = nothing
@inline update_content!(::WENO, args...) = nothing

function WENO(order::Int, dimension::Int; numericalFlux::NumericalFluxFunction = RusanovFlux())
    @assert order >= 2 "WENO requires order >= 2 for second derivatives."
    
    interpolator = Interpolator{dimension, order, 1}()
    return WENO{dimension, typeof(interpolator)}(order, interpolator)
end

# =========================================================================
# UNIVERSAL N-DIMENSIONAL WENO FUNCTOR
# =========================================================================

"""
Functor for Universal N-Dimensional WENO.
Calculates smoothness and weights natively component-by-component for State{M}.
"""
function (weno::WENO{D, I})(
    eq::HyperbolicPDE, i::Int, f_i::State{M}, nb_slice::UnitRange{Int},       
    pg::ParticleGrid{D}, ib::InteractionBuffer{D, M}    
) where {D, M, I}

    vel = velocity(eq, f_i)
    interp = weno.interpolator
    dist_all = get_distances(pg)
    w_all = get_weights(pg)

    num_nb = length(nb_slice)
    if num_nb < weno.order; return zero(State{M}); end

    # Use minimum dx for dimensional scaling
    scale_val = minimum(pg.meta.dx)
    dx2 = scale_val^2
    dx4 = dx2^2
    e_tol = 1e-12

    # --- 1. Central Stencil Calculation ---
    @inbounds for global_idx in nb_slice; ib.mask[global_idx] = true; end
    
    # Bufferless masked MLS call
    resC = interp(nb_slice, dist_all, w_all, ib.df, ib.mask; scale=scale_val)

    # Smoothness Indicator (Native SVector Component-wise!)
    smoothC = zero(State{M})
    @inbounds for k in 1:length(resC)
        weight = k <= D ? dx2 : dx4 # First D terms are linear, rest are quadratic+
        smoothC += (resC[k] .* resC[k]) * weight
    end
    betaC = 0.5 ./ ((smoothC .+ e_tol) .^ 2)

    div_total = zero(State{M})

    # --- 2. Directional Stencils (One per spatial dimension) ---
    for d in 1:D
        stencil_size = 0
        use_left = vel[d] > 0.0
        
        # Build Upwind-biased stencil for dimension d
        @inbounds for global_idx in nb_slice
            dist_k = dist_all[global_idx][d]
            if (use_left && dist_k < 0.0) || (!use_left && dist_k >= 0.0)
                ib.mask[global_idx] = true
                stencil_size += 1
            else
                ib.mask[global_idx] = false
            end
        end

        # Fallback to pure central if the directional stencil is starved
        if stencil_size < weno.order
            div_total += resC[d] * vel[d]
            continue
        end

        # Interpolate directional stencil
        resS = interp(nb_slice, dist_all, w_all, ib.df, ib.mask; scale=scale_val)

        # Smoothness Indicator for Directional Stencil
        smoothS = zero(State{M})
        @inbounds for k in 1:length(resS)
            weight = k <= D ? dx2 : dx4
            smoothS += (resS[k] .* resS[k]) * weight
        end
        betaS = 0.5 ./ ((smoothS .+ e_tol) .^ 2)

        # --- 3. Apply Non-Linear Weights ---
        sum_beta = betaS .+ betaC
        
        wS = betaS ./ sum_beta
        wC = betaC ./ sum_beta

        # The derivative for dimension d is exactly the d-th basis component
        div_d = wS .* resS[d] .+ wC .* resC[d]
        
        # Accumulate into total divergence
        div_total += div_d * vel[d]
    end

    return div_total
end