export UpwindDivergence
export TiwariAlgorithm, PraveenAlgorithm, ClassicAlgorithm, UpwindAlgorithm

abstract type UpwindAlgorithm end 
abstract type TiwariAlgorithm <: UpwindAlgorithm end 
abstract type PraveenAlgorithm <: UpwindAlgorithm end  
abstract type NonLinearPraveenAlgorithm <: UpwindAlgorithm end  
abstract type ClassicAlgorithm <: UpwindAlgorithm end 

"""
    UpwindDivergence{D, M, T, MAX_ORDER, B_LEN, INTERPS, Algorithm <: UpwindAlgorithm, NF}

A divergence interpolator executing upwind numerical schemes.
Dynamically tracks polynomial limits via `ParticleGridCore` and supports universal order degradation.
"""
struct UpwindDivergence{D, M, T, MAX_ORDER, B_LEN, INTERPS, Algorithm <: UpwindAlgorithm, NF} <: DivergenceInterpolator
    interpolators::INTERPS
    flux::NF
end

@inline update_size!(::UpwindDivergence, ::Int) = nothing
@inline update_content!(::UpwindDivergence, args...) = nothing
@inline _extract_order(::UpwindDivergence{D, M, T, MAX_ORDER}) where {D, M, T, MAX_ORDER} = MAX_ORDER

#==============================================================================
  UPWIND GRADIENT FUNCTORS
==============================================================================#

# --- Classic Algorithm ---
function (upwind::UpwindDivergence{D, M, T, MAX_ORDER, B_LEN, INTERPS, ClassicAlgorithm, NF})(
    eq::HyperbolicPDE, i::Int, f_i::State{M, T}, nb_slice::UnitRange{Int},       
    pg::ParticleGrid{D, M, T}, ib::InteractionBuffer{D, M, T}    
) where {D, M, T, MAX_ORDER, B_LEN, INTERPS, NF}
    
    num_nb = length(nb_slice)
    p_order = pg.core.particle_orders[i]

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

    F_i = flux(eq, f_i) 
    dist_all = get_distances(pg)
    
    @inbounds for global_idx in nb_slice
        dist_k = dist_all[global_idx]
        f_j    = ib.f[global_idx]
        F_j    = flux(eq, f_j)
        
        f_L, f_R, F_L, F_R = sort_flux(f_i, f_j, F_i, F_j, dist_k)
        F_num = upwind.flux(f_L, f_R, F_L, F_R, eq)
        nc_jump = evaluate_nc_jump(eq, f_L, f_R, dist_k)
        
        ib.df_flux[global_idx] = F_num - F_i + nc_jump
    end
    
    div = compute_dynamic_divergence(
        upwind.interpolators, p_order, 
        nb_slice, dist_all, get_weights(pg), 
        ib.df_flux, ib.df_scratch, pg.meta.dx
    )
    
    return T(2.0) * div
end

# --- Tiwari Algorithm ---
function (upwind::UpwindDivergence{D, 1, T, MAX_ORDER, B_LEN, INTERPS, TiwariAlgorithm, NF})(
    eq::HyperbolicPDE, i::Int, f_i::State{1, T}, nb_slice::UnitRange{Int}, 
    pg::ParticleGrid{D, 1, T}, ib::InteractionBuffer{D, 1, T} 
) where {D, T, MAX_ORDER, B_LEN, INTERPS, NF}

    dist_all = get_distances(pg)
    w_all = get_weights(pg) 

    num_nb = length(nb_slice)
    p_order_base = pg.core.particle_orders[i]
    scale = pg.meta.dx
    
    div_tuple = ntuple(Val(D)) do d
        vel_d = velocity(eq, f_i, d)[1, 1] 
        
        stencil_size = 0 
        @inbounds for global_idx in nb_slice
            dist_k = dist_all[global_idx]
            if (vel_d * dist_k[d] <= zero(T)) 
                ib.mask[global_idx] = true
                stencil_size += 1
            else
                ib.mask[global_idx] = false
            end
        end

        p_order_d = p_order_base
        while p_order_d > 1
            req_nb = typeof(basis_length(Val(D), Val(p_order_d))).parameters[1]
            if stencil_size >= req_nb
                break
            end
            p_order_d -= 1
        end

        if stencil_size < typeof(basis_length(Val(D), Val(p_order_d))).parameters[1]
            return zero(State{1, T})
        end
        
        scale_d = scale[d]
        
        # FIX: Degree maps directly to p_order_d
        B_LEN_D_VAL = typeof(basis_length(Val(D), Val(p_order_d))).parameters[1]
        
        raw_grad = dispatch_interpolator(
            upwind.interpolators, p_order_d, 
            nb_slice, dist_all, w_all, ib.df, ib.mask, scale_d, Val(B_LEN_D_VAL), State{1, T}
        )
        
        # The first derivatives perfectly align with the leading D components
        dF_dx = raw_grad[d]
        return dF_dx * vel_d
    end

    return sum(div_tuple) 
end

# --- Praveen Algorithm (Strictly 1st Order, Stateless) ---
function (upwind::UpwindDivergence{2, 1, T, MAX_ORDER, B_LEN, INTERPS, PraveenAlgorithm, NF})(
    eq::HyperbolicPDE, i::Int, f_i::State{1, T}, nb_slice::UnitRange{Int}, 
    pg::ParticleGrid{2, 1, T}, ib::InteractionBuffer{2, 1, T} 
) where {T, MAX_ORDER, B_LEN, INTERPS, NF}

    vel_x = velocity(eq, f_i, 1)[1, 1]
    vel_y = velocity(eq, f_i, 2)[1, 1]
    vel = Space{2, T}(vel_x, vel_y)

    dist_all = get_distances(pg)
    w_all = get_weights(pg)

    num_nb = length(nb_slice)
    if num_nb < 3; return zero(State{1, T}); end 
    
    scale = min(pg.meta.dx[1], pg.meta.dx[2])
    if scale < T(1e-14); return zero(State{1, T}); end
    
    invL = one(T) / scale

    N_s = zero(SMatrix{2, 2, T, 4})
    @inbounds for global_idx in nb_slice
        w_k = w_all[global_idx]
        dist_s = dist_all[global_idx] * invL
        N_s += w_k * (dist_s * dist_s')
    end
    
    if abs(det(N_s)) < T(1e-14); return zero(State{1, T}); end

    div = zero(State{1, T})

    @inbounds for global_idx in nb_slice
        w_k = w_all[global_idx] 
        dist_k = dist_all[global_idx] 
        b_s = w_k * dist_k * invL
        c_s = N_s \ b_s
        coeff = c_s * invL

        hyp = norm(dist_k)
        if hyp < T(1e-14)
            nx, ny = one(T), zero(T)
        else
            nx, ny = dist_k[1]/hyp, dist_k[2]/hyp
        end
        sx = -ny 
        sy = nx 
        alfaBar = dot(SVector{2, T}(nx, ny), coeff)
        betaBar = dot(SVector{2, T}(sx, sy), coeff)

        vel_n = dot(vel, SVector{2, T}(nx, ny))
        vel_s = dot(vel, SVector{2, T}(sx, sy))

        bracketMinus1 = min(vel_n, zero(T))
        bracketMinus2 = min(betaBar * vel_s, zero(T))
        cij = alfaBar * bracketMinus1 + bracketMinus2
        
        div += cij * ib.df[global_idx] 
    end
    return T(2.0) * div
end