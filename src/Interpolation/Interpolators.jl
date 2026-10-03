
"""
    Interpolator{D, IO, DO}()

A generalized moving least squares (MLS) interpolator struct parameterized for arbitrary dimensions and orders. 

# Type Parameters
- `D`: The spatial dimension of the problem.
- `IO`: The interpolation order.
- `DO`: The derivative order.
"""
struct Interpolator{D, IO, DO}
    function Interpolator{D, IO, DO}() where {D, IO, DO}
        new{D, IO, DO}()
    end
end

struct ConstantReconstruction end

# =========================================================================
# UNIVERSAL DYNAMIC DISPATCH ROUTERS (Zero-Allocation)
# =========================================================================

@inline _pad_grad(g::SVector{L, T}, ::Val{MAX_L}) where {L, MAX_L, T} = SVector{MAX_L, T}(ntuple(i -> i <= L ? g[i] : zero(T), Val(MAX_L)))

# --- 1. Standard (Unmasked) Evaluators used by MUSCL & Central ---
@inline _compute_raw_grad(::ConstantReconstruction, nb_slice, dist_all, w_all, df, scale, ::Val{MAX_B_LEN}, ::Type{State{M, T}}) where {MAX_B_LEN, M, T} = zero(SVector{MAX_B_LEN, State{M, T}})

@inline function _compute_raw_grad(interp::Interpolator, nb_slice, dist_all, w_all, df, scale, ::Val{MAX_B_LEN}, ::Type{State{M, T}}) where {MAX_B_LEN, M, T}
    raw = interp(nb_slice, dist_all, w_all, df; scale=scale)
    return _pad_grad(raw, Val(MAX_B_LEN))
end

# --- 2. Masked Evaluators used by WENO & Upwind(Tiwari) ---
@inline _compute_raw_grad(::ConstantReconstruction, nb_slice, dist_all, w_all, df, mask::AbstractVector{Bool}, scale, ::Val{MAX_B_LEN}, ::Type{State{M, T}}) where {MAX_B_LEN, M, T} = zero(SVector{MAX_B_LEN, State{M, T}})

@inline function _compute_raw_grad(interp::Interpolator, nb_slice, dist_all, w_all, df, mask::AbstractVector{Bool}, scale, ::Val{MAX_B_LEN}, ::Type{State{M, T}}) where {MAX_B_LEN, M, T}
    raw = interp(nb_slice, dist_all, w_all, df, mask; scale=scale)
    return _pad_grad(raw, Val(MAX_B_LEN))
end

# --- 3. The Generated Dispatchers ---
@generated function dispatch_interpolator(interps::Tuple, order::Int, args...)
    N = length(interps.parameters)
    expr = :(error("Order out of bounds"))
    for i in N:-1:1
        expr = :(order == $i ? _compute_raw_grad(interps[$i], args...) : $expr)
    end
    return expr
end

@generated function compute_dynamic_divergence(interps::Tuple, div_idx::Int, nb_slice, dist_all, w_all, df_flux, df_scratch, scale)
    N = length(interps.parameters)
    expr = :(interps[$N](nb_slice, dist_all, w_all, df_flux, df_scratch; scale=scale))
    
    for i in (N-1):-1:1
        expr = :(div_idx == $i ? interps[$i](nb_slice, dist_all, w_all, df_flux, df_scratch; scale=scale) : $expr)
    end
    return expr
end

# --- 4. Constant Reconstruction Functor ---
# CRITICAL FIX: Extract M and T securely from the strongly-typed flux array 
# to guarantee a valid 0-state return, regardless of the dimension of `scale`.
@inline function (::ConstantReconstruction)(
    nb_slice, dist_all, w_all, df_flux::AbstractVector{Flux{D, M, T}}, df_scratch; scale
) where {D, M, T}
    return zero(State{M, T})
end

# =========================================================================
# COMPILE-TIME METADATA HELPERS
# =========================================================================

# 1. Compile-time factorial calculators
_ct_factorial(n::Int) = n <= 1 ? 1 : n * _ct_factorial(n - 1)
_ct_multi_factorial(t::Tuple) = prod(_ct_factorial.(t))

# 2. Generates exponents (a_1, a_2, ..., a_D) summing to the required orders
"""
    _generate_exponents(D::Int, max_order::Int)

Generates combinations of exponents for polynomial basis terms up to a specified maximum order.

# Arguments
- `D::Int`: The spatial dimension.
- `max_order::Int`: The maximum polynomial order.

# Returns
- Returns a sorted array of `NTuple{D, Int}` where the sum of each tuple equals the respective polynomial order. The sorting guarantees that index 1 corresponds to X, index 2 to Y, and index 3 to Z.
"""
function _generate_exponents(D::Int, max_order::Int)
    res = NTuple{D, Int}[]
    for order in 1:max_order
        current_order_tuples = NTuple{D, Int}[]
        
        for t in Iterators.product(ntuple(_ -> 0:order, D)...)
            if sum(t) == order
                push!(current_order_tuples, t)
            end
        end
        
        # FIX: Using `reverse(x)` ensures (1, 0) comes before (0, 1).
        # This guarantees that index 1 is X, index 2 is Y, index 3 is Z
        sort!(current_order_tuples, by = x -> (-maximum(x), reverse(x)))
        append!(res, current_order_tuples)
    end
    return res
end

# 3. Dynamic B_LEN Resolver
@generated function basis_length(::Val{D}, ::Val{IO}) where {D, IO}
    len = length(_generate_exponents(D, IO))
    return :(Val($len))
end

# =========================================================================
# BASIS VECTOR EVALUATORS (Fully Generalized & Type-Stable)
# =========================================================================
"""
    build_basis(::Val{IO}, d::SVector{D, T})

Generates a fully generalized, type-stable polynomial basis vector for moving least squares interpolation.

# Arguments
- `::Val{IO}`: A value type specifying the interpolation order.
- `d::SVector{D, T}`: The scaled distance vector.

# Returns
- An `SVector` containing the computed basis terms, including precomputed Taylor prefactors.
"""
@generated function build_basis(::Val{IO}, d::SVector{D, T}) where {IO, D, T}
    exps = _generate_exponents(D, IO)
    B_LEN = length(exps)
    
    exprs = Any[] # Changed from Expr[] to Any[]
    for t in exps
        denom = _ct_multi_factorial(t)
        
        # Precompute the Taylor prefactor if it is not 1
        term = denom == 1 ? nothing : :(T($(1.0 / denom)))
        
        # Append the polynomial terms
        for i in 1:D
            if t[i] == 1
                term = isnothing(term) ? :(d[$i]) : :($term * d[$i])
            elseif t[i] > 1
                term = isnothing(term) ? :(d[$i]^$(t[i])) : :($term * d[$i]^$(t[i]))
            end
        end
        
        push!(exprs, term)
    end
    
    return :(SVector{$B_LEN, T}($(exprs...)))
end

# =========================================================================
# PHYSICAL SCALE FACTORS
# =========================================================================
"""
    build_scale_factors(::Val{D}, ::Val{IO}, invL::T)

Constructs the physical scale factors for the interpolator based on the inverse length scale.

# Arguments
- `::Val{D}`: The spatial dimension.
- `::Val{IO}`: The interpolation order.
- `invL::T`: The inverse of the characteristic length scale.

# Returns
- An `SVector` of scale factors corresponding to each polynomial degree in the basis.
"""
@generated function build_scale_factors(::Val{D}, ::Val{IO}, invL::T) where {D, IO, T}
    exps = _generate_exponents(D, IO)
    B_LEN = length(exps)
    
    exprs = Any[] # Changed from Expr[] to Any[]
    for t in exps
        deg = sum(t)
        push!(exprs, deg == 1 ? :(invL) : :(invL^$deg))
    end
    
    return :(SVector{$B_LEN, T}($(exprs...)))
end

# =========================================================================
# EPD_1 BASIS TRUNCATION
# =========================================================================
"""
    mask_basis(basis::SVector{B_LEN, T}, order::Int, ::Val{D})

Truncates the basis vector down to a specific spatial order at compile-time.

# Arguments
- `basis::SVector{B_LEN, T}`: The full evaluated basis vector.
- `order::Int`: The spatial order to which the basis should be truncated.
- `::Val{D}`: The spatial dimension.

# Returns
- A masked `SVector` where basis terms exceeding the requested spatial order are strictly zeroed out.
"""
@generated function mask_basis(basis::SVector{B_LEN, T}, order::Int, ::Val{D}) where {B_LEN, D, T}
    max_o = 1
    while length(_generate_exponents(D, max_o)) < B_LEN
        max_o += 1
    end
    
    expr = :(zero(SVector{B_LEN, T}))
    
    # FIX: Shift max_o up by 1 to match the spatial `order` numbering
    max_spatial_order = max_o + 1
    
    for o in max_spatial_order:-1:2 
        len = length(_generate_exponents(D, o - 1))
        
        mask_tuple = ntuple(k -> k <= len ? :(basis[$k]) : :(zero(T)), B_LEN)
        expr = :(order == $o ? SVector{B_LEN, T}($(mask_tuple...)) : $expr)
    end
    
    return expr
end

# =========================================================================
# UPWIND MATRIX-VECTORIZED DISPATCH (Stateless)
# =========================================================================

function (interp::Interpolator{D, IO, DO})(
    nb_slice::UnitRange{Int}, dists::AbstractVector{Space{D, T}}, weights::AbstractVector{T},
    dfFluxVec::AbstractVector{Flux{D, M, T}}, dfVec_workspace::AbstractVector{State{M, T}};
    scale::Space{D, T}
) where {D, IO, DO, M, T}
    
    div_tuple = ntuple(Val(D)) do d
        @inbounds for global_idx in nb_slice
            dfVec_workspace[global_idx] = dfFluxVec[global_idx][d] 
        end
        
        res = interp(nb_slice, dists, weights, dfVec_workspace; scale = scale[d])
        return res[d] 
    end
    
    return sum(div_tuple)
end

# =========================================================================
# THE UNIVERSAL MLS INTERPOLATOR
# =========================================================================

function (interp::Interpolator{D, IO, 1})(
    nb_slice::UnitRange{Int},
    distVec::AbstractVector{Space{D, T}}, 
    wVec::AbstractVector{T},
    dfVec::AbstractVector{State{M, T}};
    scale::T=one(T)
) where {D, IO, M, T}
    
    B_LEN_VAL = basis_length(Val(D), Val(IO))
    return _mls_solve(nb_slice, distVec, wVec, dfVec, scale, B_LEN_VAL, Val(IO), Val(D))
end

@inline function _mls_solve(
    nb_slice::UnitRange{Int}, distVec::AbstractVector, wVec::AbstractVector,
    dfVec::AbstractVector{State{M, T}}, scale::T,
    ::Val{B_LEN}, ::Val{IO}, ::Val{D}
) where {B_LEN, IO, D, M, T}
    
    invL = one(T) / scale
    N_s = zero(SMatrix{B_LEN, B_LEN, T, B_LEN * B_LEN})
    b_s = zero(SMatrix{B_LEN, M, T, B_LEN * M})

    @inbounds for i in nb_slice
        w = wVec[i]
        p_s = build_basis(Val(IO), distVec[i] * invL) 
        
        N_s += w * (p_s * p_s') 
        b_s += w * (p_s * dfVec[i]') 
    end
    
    # REGULARIZATION: Add small eps to the diagonal to stabilize Cholesky
    eps_reg = T(1e-10) * tr(N_s) / B_LEN
    N_s_reg = N_s + eps_reg * I
    
    C = cholesky(Symmetric(N_s_reg), check=false)
    
    if !LinearAlgebra.issuccess(C)
        return zero(SVector{B_LEN, State{M, T}})
    end
    
    c_s = C \ b_s 
    scales = build_scale_factors(Val(D), Val(IO), invL)
    
    return SVector{B_LEN, State{M, T}}(ntuple(Val(B_LEN)) do k
        row_val = State{M, T}(ntuple(m -> c_s[k, m], Val(M)))
        row_val * scales[k]
    end)
end

function (interp::Interpolator{D, IO, 1})(
    nb_slice::UnitRange{Int}, distVec::AbstractVector{Space{D, T}}, 
    wVec::AbstractVector{T}, dfVec::AbstractVector{State{M, T}},
    mask::AbstractVector{Bool}; scale::T=one(T)
) where {D, IO, M, T}
    
    B_LEN_VAL = basis_length(Val(D), Val(IO))
    return _mls_solve_masked(nb_slice, distVec, wVec, dfVec, mask, scale, B_LEN_VAL, Val(IO), Val(D))
end

@inline function _mls_solve_masked(
    nb_slice::UnitRange{Int}, distVec::AbstractVector{Space{D, T}}, 
    wVec::AbstractVector{T}, dfVec::AbstractVector{State{M, T}},
    mask::AbstractVector{Bool}, scale::T,
    ::Val{B_LEN}, ::Val{IO}, ::Val{D}
) where {B_LEN, IO, D, M, T}
    
    invL = one(T) / scale
    N_s = zero(SMatrix{B_LEN, B_LEN, T, B_LEN * B_LEN})
    b_s = zero(SMatrix{B_LEN, M, T, B_LEN * M})

    @inbounds for i in nb_slice
        if !mask[i]; continue; end
        
        w = wVec[i]
        p_s = build_basis(Val(IO), distVec[i] * invL) 
        N_s += w * (p_s * p_s') 
        b_s += w * (p_s * dfVec[i]') 
    end
    
    C = cholesky(Symmetric(N_s), check=false)
    
    if !LinearAlgebra.issuccess(C)
        return zero(SVector{B_LEN, State{M, T}})
    end
    
    c_s = C \ b_s 
    scales = build_scale_factors(Val(D), Val(IO), invL)
    
    return SVector{B_LEN, State{M, T}}(ntuple(Val(B_LEN)) do k
        row_val = State{M, T}(ntuple(m -> c_s[k, m], Val(M)))
        row_val * scales[k]
    end)
end

include("FluxFunctions.jl")

include("Central.jl")

include("MUSCL.jl")
include("Limiter.jl")

include("Upwind.jl")

include("WENO.jl")
