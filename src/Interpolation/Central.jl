export CentralDivergence

"""
    CentralDivergence{D, M, T, MAX_ORDER, DIV_ORDER, INTERPS}
"""
struct CentralDivergence{D, M, T, MAX_ORDER, DIV_ORDER, INTERPS} <: DivergenceInterpolator
    interpolators::INTERPS
end

@inline update_size!(::CentralDivergence, ::Int) = nothing
@inline update_content!(::CentralDivergence, args...) = nothing
@inline _extract_order(::CentralDivergence{D, M, T, MAX_ORDER}) where {D, M, T, MAX_ORDER} = MAX_ORDER

function CentralDivergence(::Type{T}, dimension::Int, M::Int, order::Int; div_order::Int=0) where {T}
    @assert order >= 1 "Order must be 1 or greater."       
    if div_order > order; div_order = order; end

    interps = ntuple(Val(order)) do k
        Interpolator{dimension, k, 1}()
    end

    return CentralDivergence{dimension, M, T, order, div_order, typeof(interps)}(interps)
end

function (central::CentralDivergence{D, M, T, MAX_ORDER, DIV_ORDER, INTERPS})(
    eq::HyperbolicPDE, i::Int, f_i::State{M, T}, nb_slice::UnitRange{Int},       
    pg::ParticleGrid{D, M, T}, ib::InteractionBuffer{D, M, T}    
) where {D, M, T, MAX_ORDER, DIV_ORDER, INTERPS}

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

    F_i = flux(eq, f_i)
    dist_all = get_distances(pg)

    @inbounds for global_idx in nb_slice
        f_j    = ib.f[global_idx]
        F_j    = flux(eq, f_j)
        ib.df_flux[global_idx] = F_j - F_i
    end

    div_idx = DIV_ORDER == 0 ? p_order : min(DIV_ORDER, p_order)

    div = compute_dynamic_divergence(
        central.interpolators, div_idx, 
        nb_slice, dist_all, get_weights(pg), 
        ib.df_flux, ib.df_scratch, pg.meta.dx
    )

    return div
end