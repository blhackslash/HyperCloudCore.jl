export UpwindDivergence
export TiwariAlgorithm, PraveenAlgorithm, ClassicAlgorithm
"""
    TiwariAlgorithm
    PraveenAlgorithm
    ClassicAlgorithm

Abstract types representing various upwind divergence algorithms. 
- `TiwariAlgorithm`: Restricted to scalar PDEs.
- `PraveenAlgorithm`: Restricted to 1st-order scalar PDEs.
- `ClassicAlgorithm`: Natively supports both scalars and systems via state structures.
"""
abstract type TiwariAlgorithm <: UpwindAlgorithm end 
abstract type PraveenAlgorithm <: UpwindAlgorithm end  
abstract type NonLinearPraveenAlgorithm <: UpwindAlgorithm end  
abstract type ClassicAlgorithm <: UpwindAlgorithm end 
## ------------------------------- Upwind -------------------------------
struct UpwindDivergence{D, M, T, I <: Interpolator, Algorithm <: UpwindAlgorithm} <: DivergenceInterpolator
    order::Int
    flux::NumericalFluxFunction
    interpolator::I
end

@inline update_size!(::UpwindDivergence, ::Int) = nothing
@inline update_content!(::UpwindDivergence, args...) = nothing
@inline _extract_order(g::UpwindDivergence) = g.order

# =========================================================================
# UPWIND GRADIENT SETUP
# =========================================================================
"""
    UpwindDivergence(::Type{T}, dimension::Int, M::Int, order::Int; flux=UpwindFlux(), algType="Classic")

Constructs an `UpwindDivergence` evaluator, instantiating the appropriate algorithm and interpolator types.

# Arguments
- `::Type{T}`: The numeric type used for evaluations.
- `dimension::Int`: The spatial dimension.
- `M::Int`: The number of equations (M=1 for scalars).
- `order::Int`: The numerical order, which must be greater than or equal to 1.

# Keyword Arguments
- `flux::NumericalFluxFunction`: Defaults to `UpwindFlux()`.
- `algType::String`: Specifies the algorithm type. Valid options are "Classic", "Tiwari", or "Praveen[span_37](start_span)"[span_37](end_span). 

# Details
- If "Tiwari" is selected, it asserts that the system is scalar (`M == 1`).
- If "Praveen" is selected, it asserts that the system is scalar (`M == 1`) and strictly 1st order (`order == 1`).
"""
function UpwindDivergence(
    ::Type{T}, dimension::Int, M::Int, order::Int; 
    flux::NumericalFluxFunction=UpwindFlux(), algType::String="Classic"
) where {T}
    @assert order >= 1 "Order must be larger or equal to one."
    
    local alg_type
    
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

    return UpwindDivergence{dimension, M, T, I, alg_type}(order, flux, interpolator)
end

#==============================================================================
  UPWIND GRADIENT FUNCTORS
==============================================================================#

"""
    (upwind::UpwindDivergence{D, M, T, <:Any, ClassicAlgorithm})(...)

Functor execution for `ClassicAlgorithm`. This algorithm works natively across 1D, 2D, and 3D geometries, and naturally handles both scalar problems and systems of equations.

# Returns
- A `State{M, T}` representing the divergence multiplied by 2.0. It returns a zero state if the neighborhood slice is smaller than the requested interpolation order.
"""
function (upwind::UpwindDivergence{D, M, T, <:Any, ClassicAlgorithm})(
    eq::HyperbolicPDE, i::Int, f_i::State{M, T}, nb_slice::UnitRange{Int},       
    pg::ParticleGrid{D, M, T}, ib::InteractionBuffer{D, M, T}    
) where {D, M, T}
    
    if length(nb_slice) < upwind.order
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
    
    div = upwind.interpolator(
        nb_slice, dist_all, get_weights(pg), ib.df_flux, ib.df_scratch; scale = pg.meta.dx
    )
    
    return T(2.0) * div
end
"""
    (upwind::UpwindDivergence{D, 1, T, <:Any, TiwariAlgorithm})(...)

Functor execution for `TiwariAlgorithm`. This evaluation is exclusively restricted to scalar PDEs.

# Details
- Constructs a stencil mask by evaluating whether the velocity multiplied by the particle distance is less than or equal to zero.
- Returns a zeroed state if the valid stencil size falls below the interpolator's order.
"""
function (upwind::UpwindDivergence{D, 1, T, <:Any, TiwariAlgorithm})(
    eq::HyperbolicPDE, i::Int, f_i::State{1, T}, nb_slice::UnitRange{Int}, 
    pg::ParticleGrid{D, 1, T}, ib::InteractionBuffer{D, 1, T} 
) where {D, T}

    interp = upwind.interpolator
    dist_all = get_distances(pg)
    w_all = get_weights(pg) 

    num_nb = length(nb_slice)
    if num_nb < upwind.order; return zero(State{1, T}); end
    
    scale = pg.meta.dx
    
    div_tuple = ntuple(Val(D)) do d
        # FIX: Fetch the velocity matrix for this specific direction and extract the scalar
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

        if stencil_size >= upwind.order
            scale_d = scale[d]
            if upwind.order == 1
                res = interp(nb_slice, dist_all, w_all, ib.df, ib.mask; scale = scale_d)
                dF_dx = State{1, T}(res[d, 1])
            else
                res_tuple = interp(nb_slice, dist_all, w_all, ib.df, ib.mask; scale = scale_d)
                dF_dx = State{1, T}(res_tuple[1][d, 1])
            end
            return dF_dx * vel_d
        else
            return zero(State{1, T})
        end
    end

    return sum(div_tuple) 
end
"""
    (upwind::UpwindDivergence{2, 1, T, <:Any, PraveenAlgorithm})(...)

Functor execution for `PraveenAlgorithm`. This execution path is restricted to scalar PDEs exclusively in 2D space.

# Details
- Assembles a 2D velocity vector by explicitly evaluating the velocity in both principal directions.
- Evaluates moving least squares normal and shear components using specialized thresholding (e.g., `min(vel_n, zero(T))`).
"""
function (upwind::UpwindDivergence{2, 1, T, <:Any, PraveenAlgorithm})(
    eq::HyperbolicPDE, i::Int, f_i::State{1, T}, nb_slice::UnitRange{Int}, 
    pg::ParticleGrid{2, 1, T}, ib::InteractionBuffer{2, 1, T} 
) where {T}

    # FIX: Assemble the 2D velocity vector by evaluating both directions
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
