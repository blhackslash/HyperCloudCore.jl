# =========================================================================
# STATELESS CENTRAL DIVERGENCE API
# =========================================================================

@inline update_size!(::CentralDivergence, ::Int) = nothing
@inline update_content!(::CentralDivergence, args...) = nothing

function CentralDivergence(::Type{T}, dimension::Int, M::Int, order::Int) where {T}
    @assert order >= 1 "Order must be 1 or greater."       

    interpolator = Interpolator{dimension, order, 1}()
    I = typeof(interpolator)

    return CentralDivergence{dimension, M, T, I}(order, interpolator)
end

# =========================================================================
# UNIVERSAL N-DIMENSIONAL CENTRAL FUNCTOR
# =========================================================================

"""
Functor for CentralDivergence.
Works natively for 1D, 2D, 3D, and fully supports Systems via `State{M, T}`.
"""
function (central::CentralDivergence{D, M, T, I})(
    eq::HyperbolicPDE, i::Int, f_i::State{M, T}, nb_slice::UnitRange{Int},       
    pg::ParticleGrid{D, M, T}, ib::InteractionBuffer{D, M, T}    
) where {D, M, T, I}

    num_nb = length(nb_slice)
    
    # Check if we have enough neighbors for the interpolation order
    if num_nb < central.order
        return zero(State{M, T})
    end

    F_i = flux(eq, f_i)
    dist_all = get_distances(pg)

    # --- 1. Compute Raw Central Flux Differences ---
    @inbounds for global_idx in nb_slice
        dist_k = dist_all[global_idx]
        f_j    = ib.f[global_idx]
        F_j    = flux(eq, f_j)
        
        # Central difference directly applies the physical jump
        nc_jump = evaluate_nc_jump(eq, f_i, f_j, dist_k)
        
        # Write directly to the global, mutually-exclusive slot
        ib.df_flux[global_idx] = F_j - F_i + nc_jump
    end

    # --- 2. Call the Stateless Matrix-Vectorized Interpolator ---
    # By passing df_flux directly, the interpolator perfectly computes the full divergence natively
    div = central.interpolator(
        nb_slice, dist_all, get_weights(pg), ib.df_flux, ib.df_scratch; scale = pg.meta.dx
    )

    return div
end