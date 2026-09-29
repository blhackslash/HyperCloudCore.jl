export CentralDivergence
"""
    CentralDivergence{D, M, T, I <: Interpolator}

A divergence interpolator executing a stateless central difference numerical scheme.

# Fields
- `order::Int`: The numerical order of the interpolation.
- `interpolator::I`: The underlying `Interpolator` instance used for the divergence calculation.
"""
struct CentralDivergence{D, M, T, I <: Interpolator} <: DivergenceInterpolator
    order::Int
    interpolator::I
end

# =========================================================================
# STATELESS CENTRAL DIVERGENCE API
# =========================================================================

@inline update_size!(::CentralDivergence, ::Int) = nothing
@inline update_content!(::CentralDivergence, args...) = nothing
@inline _extract_order(g::CentralDivergence) = g.order

"""
    CentralDivergence(::Type{T}, dimension::Int, M::Int, order::Int)

Constructs a `CentralDivergence` evaluator system.

# Arguments
- `::Type{T}`: The numeric type used for evaluations.
- `dimension::Int`: The spatial dimension.
- `M::Int`: The number of equations in the system.
- `order::Int`: The numerical order.

# Details
- Asserts that the requested numerical `order` is at least 1.
- Initializes an internal moving least squares interpolator configured specifically for the provided spatial dimension and order.
"""
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
    (central::CentralDivergence)(eq::HyperbolicPDE, i::Int, f_i, nb_slice, pg, ib)

The primary functor execution for computing the stateless central divergence. This algorithm natively supports 1D, 2D, and 3D geometries alongside systems of equations via the `State{M, T}` type.

# Details
- Returns a zero state if the available neighborhood slice is smaller than the requested interpolation order.
- Iterates through the active neighborhood to calculate raw central flux differences (`F_j - F_i`).
- Feeds the raw flux differences directly into the internal matrix-vectorized interpolator to extract the final unified divergence.
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
        f_j    = ib.f[global_idx]
        F_j    = flux(eq, f_j)
        
        # Write directly to the global, mutually-exclusive slot
        ib.df_flux[global_idx] = F_j - F_i
    end

    # --- 2. Call the Stateless Matrix-Vectorized Interpolator ---
    # By passing df_flux directly, the interpolator perfectly computes the full divergence natively
    div = central.interpolator(
        nb_slice, dist_all, get_weights(pg), ib.df_flux, ib.df_scratch; scale = pg.meta.dx
    )

    return div
end