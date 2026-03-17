# =========================================================================
# BASIS BUILDERS (For Higher Order Interpolations)
# =========================================================================

# 1D: [x, x^2/2]
@inline build_o2_basis(d::SVector{1, Float64}) = SVector(d[1], 0.5 * d[1]^2)

# 2D: [x, y, x^2/2, y^2/2, xy]
@inline build_o2_basis(d::SVector{2, Float64}) = SVector(d[1], d[2], 0.5 * d[1]^2, 0.5 * d[2]^2, d[1]*d[2])

# 3D: [x, y, z, x^2/2, y^2/2, z^2/2, xy, xz, yz]
@inline build_o2_basis(d::SVector{3, Float64}) = SVector(d[1], d[2], d[3], 0.5 * d[1]^2, 0.5 * d[2]^2, 0.5 * d[3]^2, d[1]*d[2], d[1]*d[3], d[2]*d[3])


# =========================================================================
# ORDER 0 INTERPOLATOR (Weighted Average)
# =========================================================================

function (interp::Interpolator{D, 0, 0})(
    nb_slice::UnitRange{Int},
    wVec::AbstractVector{Float64},
    fVec::AbstractVector{T} # T is SVector{NM, Float64}
) where {D, T}
    
    sum_w = 0.0
    sum_wf = zero(T)

    @inbounds for i in nb_slice
        w = wVec[i]
        sum_w += w
        sum_wf += w * fVec[i]
    end

    if abs(sum_w) < 1e-14
        return zero(T) 
    else
        return sum_wf / sum_w 
    end
end

# =========================================================================
# ORDER 1 INTERPOLATOR (Linear MLS)
# =========================================================================

function (interp::Interpolator{D, 1, 1})(
    nb_slice::UnitRange{Int},
    distVec::AbstractVector{SVector{D, Float64}}, 
    wVec::AbstractVector{Float64},
    dfVec::AbstractVector{T}; # T is SVector{NM, Float64}
    scale::Float64=1.0
) where {D, T}
    
    invL = 1.0 / scale
    NM = length(T)
    
    N_s = @SMatrix zeros(Float64, D, D)
    b_s = zero(SMatrix{D, NM, Float64, D * NM})

    @inbounds for i in nb_slice
        w = wVec[i]
        p_s = distVec[i] * invL 
        
        N_s += w * (p_s * p_s') 
        b_s += w * (p_s * dfVec[i]') # Outer product builds the block RHS
    end
    
    if abs(det(N_s)) < 1e-14
        return zero(b_s) 
    end
    
    # StaticArrays solves gradients for ALL macroscopic variables at once
    c_s = N_s \ b_s 
    
    # Returns an SMatrix of size (D x NM)
    return c_s * invL 
end

# =========================================================================
# ORDER 2 INTERPOLATOR (Quadratic MLS)
# =========================================================================

function (interp::Interpolator{D, 2, 1})(
    nb_slice::UnitRange{Int},
    distVec::AbstractVector{SVector{D, Float64}},
    wVec::AbstractVector{Float64},
    dfVec::AbstractVector{T}; # T is SVector{NM, Float64}
    scale::Float64=1.0
) where {D, T}
    
    invL = 1.0 / scale
    invL2 = invL * invL
    NM = length(T)
    
    # Compile-time resolution of basis size based on Dimension
    B_LEN = D == 1 ? 2 : (D == 2 ? 5 : 9)
    
    N_s = @SMatrix zeros(Float64, B_LEN, B_LEN)
    b_s = zero(SMatrix{B_LEN, NM, Float64, B_LEN * NM})

    @inbounds for i in nb_slice
        w = wVec[i]
        p_s = build_o2_basis(distVec[i] * invL) 
        
        N_s += w * (p_s * p_s')
        b_s += w * (p_s * dfVec[i]')
    end
    
    if abs(det(N_s)) < 1e-14
        return zero(SMatrix{D, NM, Float64, D * NM}), zero(SMatrix{B_LEN - D, NM, Float64, (B_LEN - D) * NM})
    end
    
    c_s = N_s \ b_s
    
    # Slice the SMatrix to separate first derivatives from higher-order curves
    # Row indices 1:D are the slopes. Remaining rows are curvatures.
    slopes = c_s[SOneTo(D), :] * invL
    curves = c_s[(D+1):B_LEN, :] * invL2
    
    # Both are returned as SMatrix. 
    # slopes is size (D x NM). curves is size ((B_LEN - D) x NM).
    return slopes, curves
end