export UpwindDivergence
export TiwariAlgorithm, PraveenAlgorithm, ClassicAlgorithm, UpwindAlgorithm

abstract type UpwindAlgorithm end 
abstract type TiwariAlgorithm <: UpwindAlgorithm end 
abstract type PraveenAlgorithm <: UpwindAlgorithm end  
abstract type NonLinearPraveenAlgorithm <: UpwindAlgorithm end  
abstract type ClassicAlgorithm <: UpwindAlgorithm end 

"""
    UpwindDivergence{D, M, T, MAX_ORDER, DIV_ORDER, B_LEN, INTERPS, Algorithm <: UpwindAlgorithm, NF} <: DivergenceInterpolator
    UpwindDivergence(::Type{T}, dimension::Int, M::Int, order::Int, algType::Symbol, flux::NumericalFluxFunction; div_order::Int=0)
    (upwind::UpwindDivergence)(eq, i, f_i, nb_slice, pg, ib)

Upwind divergence interpolator for evaluating flux derivatives in hyperbolic PDEs using directional physics-based algorithms.

# Constructors

    UpwindDivergence(::Type{T}, dimension::Int, M::Int, order::Int, algType::Symbol, flux::NumericalFluxFunction; div_order::Int=0)

- `T`: The numeric type (e.g., `Float64`).
- `dimension`: Spatial dimension.
- `M`: Number of state components.
- `order`: Maximum interpolation order. Must be `1` (upwind evaluates piecewise-constant states).
- `algType`: Symbol specifying the upwind strategy. Accepts `:Classic`, `:Tiwari` (scalar only), or `:Praveen` (scalar 2D only).
- `flux`: The numerical flux function instance to resolve interface states.
- `div_order`: Target order (defaults to `0`, dynamically bound to the available particle order).

# Callable / Functor

    (upwind::UpwindDivergence)(eq, i, f_i, nb_slice, pg, ib) -> State{M, T}

Evaluates the divergence at particle `i`. The behavior dispatched depends on the underlying `Algorithm`:
- **Classic Algorithm**: Computes interface fluxes via standard Riemann solver logic and non-conservative jump evaluation. Dynamically degrades interpolation order based on neighbor count.
- **Tiwari Algorithm**: Evaluates one-sided directional stencils masked by the advection velocity sign, calculating the divergence dimension-by-dimension.
- **Praveen Algorithm**: Uses a strictly first-order, stateless approach computing upwind fluxes via local moment matrices and edge normal/tangential velocities.

# Fields
- `interpolators`: Pre-allocated tuple of `Interpolator` instances up to `MAX_ORDER`.
- `flux`: The associated numerical flux function (`NF`).
"""
struct UpwindDivergence{D, M, T, MAX_ORDER, DIV_ORDER, B_LEN, INTERPS, Algorithm <: UpwindAlgorithm, NF} <: DivergenceInterpolator
    interpolators::INTERPS
    flux::NF
end

@inline update_size!(::UpwindDivergence, ::Int) = nothing
@inline update_content!(::UpwindDivergence, args...) = nothing
@inline _extract_order(::UpwindDivergence{D, M, T, MAX_ORDER}) where {D, M, T, MAX_ORDER} = MAX_ORDER

function UpwindDivergence(
    ::Type{T}, dimension::Int, M::Int, order::Int, 
    algType::Symbol, flux::NumericalFluxFunction; 
    div_order::Int=0
) where {T}
    @assert order == 1 "Upwind evaluates piecewise-constant states and is strictly 1st order."
    if div_order > order; div_order = order; end
    
    local alg_type
    if algType === :Classic
        alg_type = ClassicAlgorithm
    elseif algType === :Tiwari
        alg_type = TiwariAlgorithm
        @assert M == 1 "Tiwari Algorithm only supports Scalar Equations."
    elseif algType === :Praveen
        alg_type = PraveenAlgorithm 
        @assert M == 1 "Praveen Algorithm only supports Scalar Equations."
    else
        error("Algorithm type $algType not fully configured for workspace selection.")
    end

    B_LEN_VAL = basis_length(Val(dimension), Val(order))
    B_LEN = typeof(B_LEN_VAL).parameters[1] 

    interps = ntuple(Val(order)) do k
        Interpolator{dimension, k, 1}()
    end

    return UpwindDivergence{dimension, M, T, order, div_order, B_LEN, typeof(interps), alg_type, typeof(flux)}(
        interps, flux
    )
end

# --- Classic Algorithm ---
function (upwind::UpwindDivergence{D, M, T, MAX_ORDER, DIV_ORDER, B_LEN, INTERPS, ClassicAlgorithm, NF})(
    eq::HyperbolicPDE, i::Int, f_i::State{M, T}, nb_slice::UnitRange{Int},       
    pg::ParticleGrid{D, M, T}, ib::InteractionBuffer{D, M, T}    
) where {D, M, T, MAX_ORDER, DIV_ORDER, B_LEN, INTERPS, NF}
    
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
        dist_k = dist_all[global_idx]
        f_j    = ib.f[global_idx]
        F_j    = flux(eq, f_j)
        
        f_L, f_R, F_L, F_R = sort_flux(f_i, f_j, F_i, F_j, dist_k)
        F_num = upwind.flux(f_L, f_R, F_L, F_R, eq)
        nc_jump = evaluate_nc_jump(eq, f_L, f_R, dist_k)
        
        ib.df_flux[global_idx] = F_num - F_i + nc_jump
    end
    
    div_idx = DIV_ORDER == 0 ? p_order : min(DIV_ORDER, p_order)
    
    div = compute_dynamic_divergence(
        upwind.interpolators, div_idx, 
        nb_slice, dist_all, get_weights(pg), 
        ib.df_flux, ib.df_scratch, pg.meta.dx
    )
    
    return T(2.0) * div
end

# --- Tiwari Algorithm ---
function (upwind::UpwindDivergence{D, 1, T, MAX_ORDER, DIV_ORDER, B_LEN, INTERPS, TiwariAlgorithm, NF})(
    eq::HyperbolicPDE, i::Int, f_i::State{1, T}, nb_slice::UnitRange{Int}, 
    pg::ParticleGrid{D, 1, T}, ib::InteractionBuffer{D, 1, T} 
) where {D, T, MAX_ORDER, DIV_ORDER, B_LEN, INTERPS, NF}

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
            if stencil_size >= req_nb; break; end
            p_order_d -= 1
        end

        if stencil_size < typeof(basis_length(Val(D), Val(p_order_d))).parameters[1]
            return zero(State{1, T})
        end
        
        div_idx_d = DIV_ORDER == 0 ? p_order_d : min(DIV_ORDER, p_order_d)
        B_LEN_D_VAL = typeof(basis_length(Val(D), Val(div_idx_d))).parameters[1]
        
        raw_grad = dispatch_interpolator(
            upwind.interpolators, div_idx_d, 
            nb_slice, dist_all, w_all, ib.df, ib.mask, scale[d], Val(B_LEN_D_VAL), State{1, T}
        )
        
        dF_dx = raw_grad[d]
        return dF_dx * vel_d
    end

    return sum(div_tuple) 
end

# --- Praveen Algorithm (Strictly 1st Order, Stateless) ---
function (upwind::UpwindDivergence{2, 1, T, MAX_ORDER, DIV_ORDER, B_LEN, INTERPS, PraveenAlgorithm, NF})(
    eq::HyperbolicPDE, i::Int, f_i::State{1, T}, nb_slice::UnitRange{Int}, 
    pg::ParticleGrid{2, 1, T}, ib::InteractionBuffer{2, 1, T} 
) where {T, MAX_ORDER, DIV_ORDER, B_LEN, INTERPS, NF}

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