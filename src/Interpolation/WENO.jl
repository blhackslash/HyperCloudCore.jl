export WENO

"""
    WENO{D, M, T, MAX_ORDER, DIV_ORDER, INTERPS} <: DivergenceInterpolator
    WENO(::Type{T}, dimension::Int, M::Int, order::Int; div_order::Int=0)
    (weno::WENO)(eq, i, f_i, nb_slice, pg, ib, strategy)

Weighted Essentially Non-Oscillatory (WENO) divergence interpolator for evaluating flux derivatives in hyperbolic PDEs. It dynamically blends central and directional stencils based on local solution smoothness to prevent oscillations near discontinuities.

# Constructors

    WENO(::Type{T}, dimension::Int, M::Int, order::Int; div_order::Int=0)

- `T`: The numeric type (e.g., `Float64`).
- `dimension`: Spatial dimension.
- `M`: Number of state components.
- `order`: Maximum interpolation order (degree of the fits), which must be `≥ 2`.
- `div_order`: Target order (defaults to `0`, dynamically bound to the available particle order).

# Callable / Functor

    (weno::WENO)(eq, i, f_i, nb_slice, pg, ib, strategy) -> State{M, T}

Computes the divergence of the flux at particle `i`. The execution follows these steps:
- Evaluates a full central stencil using all available neighbors within the interaction buffer.
- Evaluates directional one-sided stencils for each spatial dimension, selecting neighbors based on the sign of the local advection velocity.
- Computes smoothness indicators for both the central and directional stencils from the first derivatives of their fits, `IS = Σ_d (∂_d u)² Δx²` (the higher-degree coefficients do not enter).
- Calculates non-linear weights to heavily penalize stencils crossing discontinuities.
- Blends the central and directional interpolations using these weights to form the final stable divergence computation.
- Scalar equations only (`M == 1`); the advection speed is taken per direction from `velocity(eq, U, d)`.

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
    @assert M == 1 "WENO is implemented for scalar equations only (directional stencils need a single wave speed)."
    if div_order > order; div_order = order; end
    
    interps = ntuple(Val(order)) do k
        Interpolator{dimension, k, 1}()
    end
    
    return WENO{dimension, M, T, order, div_order, typeof(interps)}(interps)
end

function (weno::WENO{D, M, T, MAX_ORDER, DIV_ORDER, INTERPS})(
    eq::HyperbolicPDE, i::Int, f_i::State{M, T}, nb_slice::OrdinalRange{Int, Int},
    pg::ParticleGrid{D, M, T}, U::AbstractVector{State{M, T}}, ib::InteractionBuffer{D, M, T}, ::MOODStrategy
) where {D, M, T, MAX_ORDER, DIV_ORDER, INTERPS}

    # Scalar advection speed per spatial direction (WENO is a scalar scheme, see constructor)
    vel = SVector{D, T}(ntuple(d -> velocity(eq, f_i, d)[1, 1], Val(D)))
    dist_all = get_distances(pg)
    w_all = get_weights(pg)
    df = neighbour_differences(U, pg, f_i)

    num_nb = length(nb_slice)
    p_order = pg.core.particle_orders[i]

    while p_order > 1
        req_nb = basis_count(D, p_order)
        if num_nb >= req_nb; break; end
        p_order -= 1
    end

    if num_nb < basis_count(D, p_order)
        return zero(State{M, T})
    end

    scale_val = minimum(pg.meta.dx)
    dx2 = scale_val^2
    dx4 = dx2^2

    # --- 1. Central Stencil Calculation ---
    @inbounds for global_idx in nb_slice; ib.mask[global_idx] = true; end
    
    div_idx_C = DIV_ORDER == 0 ? p_order : min(DIV_ORDER, p_order)
    # The dispatcher pads every result to the basis length of MAX_ORDER (static); a `Val` of the runtime
    # order would be a dynamic dispatch, which kernels cannot do.
    b_len = basis_length(Val(D), Val(MAX_ORDER))

    resC_raw = dispatch_interpolator(
        weno.interpolators, div_idx_C,
        nb_slice, dist_all, w_all, df, ib.mask, scale_val, b_len, State{M, T}
    )
    resC = SVector{D, State{M, T}}(ntuple(d -> resC_raw[d], Val(D)))

    # Smoothness indicator from the D first derivatives only: `resC` holds just those, so `k <= D` always and the
    # Δx⁴ weight of higher-order coefficients is never used (see docs/src/spatial/weno.md)
    smoothC = zero(State{M, T})
    @inbounds for k in 1:length(resC)
        weight = k <= D ? dx2 : dx4 
        smoothC += (resC[k] .* resC[k]) * weight
    end
    # Offset of the smoothness indicators relative to the local data (`u_i²` and the central indicator), not an
    # absolute 1e-12: the weights then do not depend on the scale of `u` (a profile of height 1e-13 is blended like
    # one of height 1). `floatmin` only keeps a locally zero field finite.
    e_tol = T(1e-12) .* max.(f_i .* f_i, smoothC) .+ floatmin(T)
    aC    = smoothC .+ e_tol

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
            req_nb = basis_count(D, p_order_d)
            if stencil_size >= req_nb; break; end
            p_order_d -= 1
        end

        if stencil_size < basis_count(D, p_order_d)
            div_total += resC[d] * vel[d]
            continue
        end

        div_idx_S = DIV_ORDER == 0 ? p_order_d : min(DIV_ORDER, p_order_d)

        resS_raw = dispatch_interpolator(
            weno.interpolators, div_idx_S,
            nb_slice, dist_all, w_all, df, ib.mask, scale_val, b_len, State{M, T}
        )
        resS = SVector{D, State{M, T}}(ntuple(k -> resS_raw[k], Val(D)))

        smoothS = zero(State{M, T})
        @inbounds for k in 1:length(resS)
            weight = k <= D ? dx2 : dx4
            smoothS += (resS[k] .* resS[k]) * weight
        end
        # --- 3. Apply Non-Linear Weights ---
        # β = ½ / (IS + ε)², normalised: w_S = β_S / (β_S + β_C) = 1 / (1 + (a_S / a_C)²). The ratio form cannot
        # overflow or underflow to Inf/Inf, which the squared offsets of a tiny field would.
        q  = (smoothS .+ e_tol) ./ aC
        wS = one(T) ./ (one(T) .+ q .* q)
        wC = one(T) .- wS

        div_d = wS .* resS[d] .+ wC .* resC[d]
        div_total += div_d * vel[d]
    end

    return div_total
end