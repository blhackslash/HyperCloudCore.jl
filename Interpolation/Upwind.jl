
@inline update_size!(::UpwindGradient, ::Int) = nothing
@inline update_content!(::UpwindGradient, args...) = nothing

# =========================================================================
# UPWIND GRADIENT SETUP
# =========================================================================

function UpwindGradient(dimension, M, order; numericalFlux::NumericalFluxFunction=UpwindFlux(), algType::String="Classic")
    @assert order >= 1 "Order must be larger or equal to one."
    
    local alg_type
    local WS_eltype::Type 
    
    if algType == "Classic"
        alg_type = ClassicAlgorithm
    elseif algType == "Tiwari"
        alg_type = TiwariAlgorithm
        @assert M == 1 "Tiwari Algorithm only supports Scalar Equations."
    elseif algType == "Praveen"
        alg_type = PraveenAlgorithm 
        @assert order == 1 "Praveen only supports 1st order."
        @assert M == 1 "Praveen Algorithm only supports Scalar Equations."
    else
        error("Algorithm type $algType not fully configured for workspace selection.")
    end

    interpolator = Interpolator{dimension, order, 1}()
    I = typeof(interpolator)

    UpwindGradient{dimension, I, alg_type}(order, numericalFlux, interpolator)
end

#==============================================================================
  UPWIND GRADIENT FUNCTORS
==============================================================================#

"""
Functor for UpwindGradient (ClassicAlgorithm).
Works for 1D, 2D, 3D, and natively supports both Scalars and Systems via `State{M}`.
"""
function (upwind::UpwindGradient{D, <:Any, ClassicAlgorithm})(
    eq::HyperbolicPDE, i::Int, f_i::State{M}, nb_slice::UnitRange{Int},       
    pg::ParticleGrid{D}, ib::InteractionBuffer{D, M}    
) where {D, M}
    
    if length(nb_slice) < upwind.order
        return zero(State{M})
    end

    F_i = flux(eq, f_i) 
    dist_all = get_distances(pg)
    
    @inbounds for global_idx in nb_slice
        dist_k = dist_all[global_idx]
        f_j    = ib.f[global_idx]
        F_j    = flux(eq, f_j)
        
        f_L, f_R, F_L, F_R = sort_flux(f_i, f_j, F_i, F_j, dist_k)
        F_num = upwind.numericalFlux(f_L, f_R, F_L, F_R, eq)
        nc_jump = evaluate_nc_jump(eq, f_L, f_R, dist_k)
        
        # Write directly to the global, mutually-exclusive slot
        ib.dfFlux[global_idx] = F_num - F_i + nc_jump
    end
    
    # Zero-copy interpolation
    div = upwind.interpolator(
        nb_slice, dist_all, get_weights(pg), ib.dfFlux, ib.df_scratch; scale = pg.meta.dx
    )
    
    return 2.0 * div
end
"""
Functor for TiwariAlgorithm. (Restricted to Scalar PDEs)
"""
function (upwind::UpwindGradient{D, <:Any, TiwariAlgorithm})(
    eq::HyperbolicPDE, i::Int, f_i::State{1}, nb_slice::UnitRange{Int},       
    pg::ParticleGrid{D}, ib::InteractionBuffer{D, 1}    
) where {D}
    
    vel = velocity(eq, f_i)
    interp = upwind.interpolator

    dist_all = get_distances(pg)
    w_all = get_weights(pg) 

    num_nb = length(nb_slice)
    if num_nb < upwind.order; return zero(State{1}); end
    
    scale = pg.meta.dx
    
    div_tuple = ntuple(Val(D)) do d
        stencil_size = 0 
        
        # 1. Flag the valid neighbors directly into the InteractionBuffer mask
        @inbounds for global_idx in nb_slice
            dist_k = dist_all[global_idx]
            if (vel[d] * dist_k[d] <= 0.0) 
                ib.mask[global_idx] = true
                stencil_size += 1
            else
                ib.mask[global_idx] = false
            end
        end

        # 2. Call the masked interpolator using the raw global arrays
        if stencil_size >= upwind.order
            scale_d = scale[d]
            if upwind.order == 1
                res = interp(nb_slice, dist_all, w_all, ib.df, ib.mask; scale = scale_d)
                dF_dx = State{1}(res[d, 1])
            else
                res_tuple = interp(nb_slice, dist_all, w_all, ib.df, ib.mask; scale = scale_d)
                dF_dx = State{1}(res_tuple[1][d, 1])
            end
            return dF_dx * vel[d]
        else
            return zero(State{1})
        end
    end

    return sum(div_tuple) 
end
"""
Functor for PraveenAlgorithm. (Restricted to Scalar PDEs in 2D)
"""
function (upwind::UpwindGradient{2, <:Any, PraveenAlgorithm})(
    eq::HyperbolicPDE, i::Int, f_i::State{1}, nb_slice::UnitRange{Int},       
    pg::ParticleGrid{2}, ib::InteractionBuffer{2, 1}    
)
    vel = Space{2}(velocity(eq, f_i))

    dist_all = get_distances(pg)
    w_all = get_weights(pg)

    num_nb = length(nb_slice)
    if num_nb < 3; return zero(State{1}); end 
    
    scale = min(pg.meta.dx[1], pg.meta.dx[2])
    if scale < 1e-14; return zero(State{1}); end
    invL = 1.0 / scale

    N_s = @SMatrix zeros(Float64, 2, 2)
    
    @inbounds for global_idx in nb_slice
        w_k = w_all[global_idx]
        dist_s = dist_all[global_idx] * invL
        N_s += w_k * (dist_s * dist_s')
    end
    
    if abs(det(N_s)) < 1e-14; return zero(State{1}); end

    div = zero(State{1})

    @inbounds for global_idx in nb_slice
        w_k    = w_all[global_idx] 
        dist_k = dist_all[global_idx] 
       
        b_s = w_k * dist_k * invL
        c_s = N_s \ b_s
        coeff = c_s * invL

        hyp = norm(dist_k)
    
        if hyp < 1e-14
            nx, ny = 1.0, 0.0
        else
            nx, ny = dist_k[1]/hyp, dist_k[2]/hyp
        end
    
        sx = -ny 
        sy = nx  
        
        alfaBar = dot(SVector(nx, ny), coeff)
        betaBar = dot(SVector(sx, sy), coeff)

        vel_n = dot(vel, SVector(nx, ny))
        vel_s = dot(vel, SVector(sx, sy))

        bracketMinus1 = min(vel_n, 0.0)
        bracketMinus2 = min(betaBar * vel_s, 0.0)
    
        cij = alfaBar * bracketMinus1 + bracketMinus2
        
        # Read natively from the Interaction Buffer's difference array
        div += cij * ib.df[global_idx] 
    end
    
    return 2.0 * div
end