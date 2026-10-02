export CentralDivergence

"""
    CentralDivergence{D, M, T, MAX_ORDER, INTERPS}

A divergence interpolator executing a stateless central difference numerical scheme.
Dynamically tracks polynomial limits via `ParticleGridCore` and supports universal order degradation.
"""
struct CentralDivergence{D, M, T, MAX_ORDER, INTERPS} <: DivergenceInterpolator
    interpolators::INTERPS
end

# =========================================================================
# STATELESS CENTRAL DIVERGENCE API
# =========================================================================

@inline update_size!(::CentralDivergence, ::Int) = nothing
@inline update_content!(::CentralDivergence, args...) = nothing
@inline _extract_order(::CentralDivergence{D, M, T, MAX_ORDER}) where {D, M, T, MAX_ORDER} = MAX_ORDER

# =========================================================================
# UNIVERSAL N-DIMENSIONAL CENTRAL FUNCTOR
# =========================================================================

function (central::CentralDivergence{D, M, T, MAX_ORDER, INTERPS})(
    eq::HyperbolicPDE, i::Int, f_i::State{M, T}, nb_slice::UnitRange{Int},       
    pg::ParticleGrid{D, M, T}, ib::InteractionBuffer{D, M, T}    
) where {D, M, T, MAX_ORDER, INTERPS}

    num_nb = length(nb_slice)
    p_order = pg.core.particle_orders[i]

    # Dynamically degrade order if the neighborhood is starved
    while p_order > 1
        req_nb = typeof(basis_length(Val(D), Val(p_order))).parameters[1]
        if num_nb >= req_nb
            break
        end
        p_order -= 1
    end
    
    # Final starvation check (if even order 1 is unsupported)
    if num_nb < typeof(basis_length(Val(D), Val(p_order))).parameters[1]
        return zero(State{M, T})
    end

    F_i = flux(eq, f_i)
    dist_all = get_distances(pg)

    # --- 1. Compute Raw Central Flux Differences ---
    @inbounds for global_idx in nb_slice
        f_j    = ib.f[global_idx]
        F_j    = flux(eq, f_j)
        ib.df_flux[global_idx] = F_j - F_i
    end

    # --- 2. Call the Stateless Matrix-Vectorized Interpolator ---
    div = compute_dynamic_divergence(
        central.interpolators, p_order, 
        nb_slice, dist_all, get_weights(pg), 
        ib.df_flux, ib.df_scratch, pg.meta.dx
    )

    return div
end