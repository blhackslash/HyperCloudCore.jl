# =========================================================================
# BASIS BUILDERS & MLS METADATA
# =========================================================================

# B_LEN Resolver (Dimension D, Order IO) -> Length of basis
@inline basis_length(::Val{1}, ::Val{1}) = Val(1)
@inline basis_length(::Val{2}, ::Val{1}) = Val(2)
@inline basis_length(::Val{3}, ::Val{1}) = Val(3)

@inline basis_length(::Val{1}, ::Val{2}) = Val(2)
@inline basis_length(::Val{2}, ::Val{2}) = Val(5)
@inline basis_length(::Val{3}, ::Val{2}) = Val(9)

@inline basis_length(::Val{1}, ::Val{3}) = Val(3)
@inline basis_length(::Val{2}, ::Val{3}) = Val(9)
@inline basis_length(::Val{3}, ::Val{3}) = Val(19)

@inline basis_length(::Val{1}, ::Val{4}) = Val(4)
@inline basis_length(::Val{2}, ::Val{4}) = Val(14)

@inline basis_length(::Val{1}, ::Val{5}) = Val(5)
@inline basis_length(::Val{2}, ::Val{5}) = Val(20)

# =========================================================================
# EPD_1 BASIS TRUNCATION
# =========================================================================
@generated function mask_basis(basis::SVector{B_LEN, Float64}, order::Int, ::Val{D}) where {B_LEN, D}
    expr = :(zero(SVector{B_LEN, Float64}))
    # Assuming MAX_ORDER generally won't exceed 5 based on your Interpolators.jl
    for o in 5:-1:2 
        len = typeof(basis_length(Val(D), Val(o-1))).parameters[1]
        mask_tuple = ntuple(k -> k <= len ? :(basis[$k]) : :(0.0), B_LEN)
        expr = :(order == $o ? SVector{B_LEN, Float64}($(mask_tuple...)) : $expr)
    end
    return expr
end

# =========================================================================
# BASIS VECTOR EVALUATORS
# =========================================================================
# Evaluates the polynomial basis natively.

# --- Order 1 ---
@inline build_basis(::Val{1}, d::SVector{1, Float64}) = SVector(d[1])
@inline build_basis(::Val{1}, d::SVector{2, Float64}) = SVector(d[1], d[2])
@inline build_basis(::Val{1}, d::SVector{3, Float64}) = SVector(d[1], d[2], d[3])

# --- Order 2 ---
@inline build_basis(::Val{2}, d::SVector{1, Float64}) = SVector(d[1], 0.5*d[1]^2)
@inline build_basis(::Val{2}, d::SVector{2, Float64}) = SVector(d[1], d[2], 0.5*d[1]^2, 0.5*d[2]^2, d[1]*d[2])
@inline build_basis(::Val{2}, d::SVector{3, Float64}) = SVector(
    d[1], d[2], d[3], 
    0.5*d[1]^2, 0.5*d[2]^2, 0.5*d[3]^2, d[1]*d[2], d[1]*d[3], d[2]*d[3]
)

# --- Order 3 ---
@inline build_basis(::Val{3}, d::SVector{1, Float64}) = SVector(d[1], 0.5*d[1]^2, (1.0/6.0)*d[1]^3)
@inline build_basis(::Val{3}, d::SVector{2, Float64}) = SVector(
    d[1], d[2], 
    0.5*d[1]^2, 0.5*d[2]^2, d[1]*d[2], 
    (1.0/6.0)*d[1]^3, (1.0/6.0)*d[2]^3, 0.5*d[1]^2*d[2], 0.5*d[1]*d[2]^2
)
@inline build_basis(::Val{3}, d::SVector{3, Float64}) = SVector(
    d[1], d[2], d[3], 
    0.5*d[1]^2, 0.5*d[2]^2, 0.5*d[3]^2, d[1]*d[2], d[1]*d[3], d[2]*d[3],
    (1.0/6.0)*d[1]^3, (1.0/6.0)*d[2]^3, (1.0/6.0)*d[3]^3, 
    0.5*d[1]^2*d[2], 0.5*d[1]^2*d[3], 0.5*d[1]*d[2]^2, 0.5*d[2]^2*d[3], 0.5*d[1]*d[3]^2, 0.5*d[2]*d[3]^2, 
    d[1]*d[2]*d[3]
)

# --- Order 4 ---
@inline build_basis(::Val{4}, d::SVector{1, Float64}) = SVector(d[1], 0.5*d[1]^2, (1.0/6.0)*d[1]^3, (1.0/24.0)*d[1]^4)
@inline build_basis(::Val{4}, d::SVector{2, Float64}) = SVector(
    d[1], d[2], 
    0.5*d[1]^2, 0.5*d[2]^2, d[1]*d[2], 
    (1.0/6.0)*d[1]^3, (1.0/6.0)*d[2]^3, 0.5*d[1]^2*d[2], 0.5*d[1]*d[2]^2,
    (1.0/24.0)*d[1]^4, (1.0/24.0)*d[2]^4, (1.0/6.0)*d[1]^3*d[2], (1.0/6.0)*d[1]*d[2]^3, 0.25*d[1]^2*d[2]^2
)

# --- Order 5 ---
@inline build_basis(::Val{5}, d::SVector{1, Float64}) = SVector(d[1], 0.5*d[1]^2, (1.0/6.0)*d[1]^3, (1.0/24.0)*d[1]^4, (1.0/120.0)*d[1]^5)
@inline build_basis(::Val{5}, d::SVector{2, Float64}) = SVector(
    d[1], d[2], 
    0.5*d[1]^2, 0.5*d[2]^2, d[1]*d[2], 
    (1.0/6.0)*d[1]^3, (1.0/6.0)*d[2]^3, 0.5*d[1]^2*d[2], 0.5*d[1]*d[2]^2,
    (1.0/24.0)*d[1]^4, (1.0/24.0)*d[2]^4, (1.0/6.0)*d[1]^3*d[2], (1.0/6.0)*d[1]*d[2]^3, 0.25*d[1]^2*d[2]^2,
    (1.0/120.0)*d[1]^5, (1.0/120.0)*d[2]^5, (1.0/24.0)*d[1]^4*d[2], (1.0/24.0)*d[1]*d[2]^4, (1.0/12.0)*d[1]^3*d[2]^2, (1.0/12.0)*d[1]^2*d[2]^3
)


# =========================================================================
# PHYSICAL SCALE FACTORS
# =========================================================================
# Matches the basis terms to convert the scaled c_s back into true physical derivatives

# --- Order 1 ---
@inline build_scale_factors(::Val{1}, ::Val{1}, invL) = SVector(invL)
@inline build_scale_factors(::Val{2}, ::Val{1}, invL) = SVector(invL, invL)
@inline build_scale_factors(::Val{3}, ::Val{1}, invL) = SVector(invL, invL, invL)

# --- Order 2 ---
@inline build_scale_factors(::Val{1}, ::Val{2}, invL) = SVector(invL, invL^2)
@inline build_scale_factors(::Val{2}, ::Val{2}, invL) = SVector(invL, invL, invL^2, invL^2, invL^2)
@inline build_scale_factors(::Val{3}, ::Val{2}, invL) = SVector(invL, invL, invL, invL^2, invL^2, invL^2, invL^2, invL^2, invL^2)

# --- Order 3 ---
@inline build_scale_factors(::Val{1}, ::Val{3}, invL) = SVector(invL, invL^2, invL^3)
@inline build_scale_factors(::Val{2}, ::Val{3}, invL) = SVector(invL, invL, invL^2, invL^2, invL^2, invL^3, invL^3, invL^3, invL^3)
@inline build_scale_factors(::Val{3}, ::Val{3}, invL) = SVector(
    invL, invL, invL, 
    invL^2, invL^2, invL^2, invL^2, invL^2, invL^2, 
    invL^3, invL^3, invL^3, invL^3, invL^3, invL^3, invL^3, invL^3, invL^3, invL^3
)

# --- Order 4 ---
@inline build_scale_factors(::Val{1}, ::Val{4}, invL) = SVector(invL, invL^2, invL^3, invL^4)
@inline build_scale_factors(::Val{2}, ::Val{4}, invL) = SVector(
    invL, invL, 
    invL^2, invL^2, invL^2, 
    invL^3, invL^3, invL^3, invL^3, 
    invL^4, invL^4, invL^4, invL^4, invL^4
)

# --- Order 5 ---
@inline build_scale_factors(::Val{1}, ::Val{5}, invL) = SVector(invL, invL^2, invL^3, invL^4, invL^5)
@inline build_scale_factors(::Val{2}, ::Val{5}, invL) = SVector(
    invL, invL, 
    invL^2, invL^2, invL^2, 
    invL^3, invL^3, invL^3, invL^3, 
    invL^4, invL^4, invL^4, invL^4, invL^4,
    invL^5, invL^5, invL^5, invL^5, invL^5, invL^5
)
# =========================================================================
# UPWIND MATRIX-VECTORIZED DISPATCH (Stateless)
# =========================================================================

function (interp::Interpolator{D, IO, DO})(
    nb_slice::UnitRange{Int}, dists::AbstractVector{Space{D}}, weights::AbstractVector{Float64},
    dfFluxVec::AbstractVector{Flux{D, M}}, dfVec_workspace::AbstractVector{State{M}};
    scale::Space{D}
) where {D, IO, DO, M}
    
    div_tuple = ntuple(Val(D)) do d
        
        # Pull the exact directional flux column natively from the global array
        @inbounds for global_idx in nb_slice
            dfVec_workspace[global_idx] = dfFluxVec[global_idx][d] 
        end
        
        # Call the Universal Interpolator with the global slice
        res = interp(nb_slice, dists, weights, dfVec_workspace; scale = scale[d])
        
        # The first `D` elements of the basis are always the linear spatial slopes!
        # E.g., for d=1 (X-direction), res[1] is exactly dFx/dx. 
        return res[d] 
    end
    
    return sum(div_tuple)
end
# =========================================================================
# THE UNIVERSAL MLS INTERPOLATOR
# =========================================================================

function (interp::Interpolator{D, IO, 1})(
    nb_slice::UnitRange{Int},
    distVec::AbstractVector{Space{D}}, 
    wVec::AbstractVector{Float64},
    dfVec::AbstractVector{State{M}};
    scale::Float64=1.0
) where {D, IO, M}
    
    # Resolves the matrix sizes perfectly at compile time
    B_LEN_VAL = basis_length(Val(D), Val(IO))
    
    return _mls_solve(nb_slice, distVec, wVec, dfVec, scale, B_LEN_VAL, Val(IO), Val(D))
end

@inline function _mls_solve(
    nb_slice::UnitRange{Int},
    distVec::AbstractVector{Space{D}}, 
    wVec::AbstractVector{Float64},
    dfVec::AbstractVector{State{M}},
    scale::Float64,
    ::Val{B_LEN}, ::Val{IO}, ::Val{D}
) where {B_LEN, IO, D, M}
    
    invL = 1.0 / scale
    N_s = zero(SMatrix{B_LEN, B_LEN, Float64, B_LEN * B_LEN})
    b_s = zero(SMatrix{B_LEN, M, Float64, B_LEN * M})

    @inbounds for i in nb_slice
        w = wVec[i]
        p_s = build_basis(Val(IO), distVec[i] * invL) 
        
        N_s += w * (p_s * p_s') 
        b_s += w * (p_s * dfVec[i]') 
    end
    
    if abs(det(N_s)) < 1e-14
        return SVector{B_LEN, State{M}}(ntuple(_ -> zero(State{M}), Val(B_LEN)))
    end
    
    c_s = N_s \ b_s 
    scales = build_scale_factors(Val(D), Val(IO), invL)
    
    # Unscale the coefficients into pure physical derivatives (slopes, curves, etc.)
    # Returns SVector{B_LEN, State{M}}
    return SVector{B_LEN, State{M}}(ntuple(k -> State{M}(c_s[k, :] * scales[k]), Val(B_LEN)))
end

# Inside InterpolationUtils.jl (below the standard Universal MLS Interpolator)

function (interp::Interpolator{D, IO, 1})(
    nb_slice::UnitRange{Int}, distVec::AbstractVector{Space{D}}, 
    wVec::AbstractVector{Float64}, dfVec::AbstractVector{State{M}},
    mask::AbstractVector{Bool}; scale::Float64=1.0
) where {D, IO, M}
    
    B_LEN_VAL = basis_length(Val(D), Val(IO))
    return _mls_solve_masked(nb_slice, distVec, wVec, dfVec, mask, scale, B_LEN_VAL, Val(IO), Val(D))
end

@inline function _mls_solve_masked(
    nb_slice::UnitRange{Int}, distVec::AbstractVector{Space{D}}, 
    wVec::AbstractVector{Float64}, dfVec::AbstractVector{State{M}},
    mask::AbstractVector{Bool}, scale::Float64,
    ::Val{B_LEN}, ::Val{IO}, ::Val{D}
) where {B_LEN, IO, D, M}
    
    invL = 1.0 / scale
    N_s = zero(SMatrix{B_LEN, B_LEN, Float64, B_LEN * B_LEN})
    b_s = zero(SMatrix{B_LEN, M, Float64, B_LEN * M})

    @inbounds for i in nb_slice
        if !mask[i]; continue; end
        
        w = wVec[i]
        p_s = build_basis(Val(IO), distVec[i] * invL) 
        N_s += w * (p_s * p_s') 
        b_s += w * (p_s * dfVec[i]') 
    end
    
    if abs(det(N_s)) < 1e-14
        return SVector{B_LEN, State{M}}(ntuple(_ -> zero(State{M}), Val(B_LEN)))
    end
    
    c_s = N_s \ b_s 
    scales = build_scale_factors(Val(D), Val(IO), invL)
    return SVector{B_LEN, State{M}}(ntuple(k -> State{M}(c_s[k, :] * scales[k]), Val(B_LEN)))
end