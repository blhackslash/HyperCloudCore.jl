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
# MATRIX-VECTORIZED INTERPOLATOR DISPATCH (Stateless)
# =========================================================================

function (interp::Interpolator{D, IO, DO})(
    num_nb::Int,
    dists::AbstractVector{Space{D}},
    weights::AbstractVector{Float64},
    dfFluxVec::AbstractVector{Flux{D, M}},
    dfVec_workspace::AbstractVector{State{M}};
    scale::Space{D}
) where {D, IO, DO, M}
    
    # LLVM unrolls this D-loop at compile time
    div_tuple = ntuple(Val(D)) do d
        

        # Pull the exact column natively! No 2D bounds-checking!
        @inbounds for local_idx in 1:num_nb
            dfVec_workspace[local_idx] = dfFluxVec[local_idx][d] 
        end
        
        scale_d = scale[d]
        
        # 2. Call the BASE scalar/state interpolator! 
        if IO == 1
            res = interp(1:num_nb, dists, weights, dfVec_workspace; scale = scale_d)
            return State{M}(ntuple(c -> res[d, c], Val(M)))
        else
            res_tuple = interp(1:num_nb, dists, weights, dfVec_workspace; scale = scale_d)
            return State{M}(ntuple(c -> res_tuple[1][d, c], Val(M)))
        end
    end
    
    # Return the full aggregated divergence vector
    return sum(div_tuple)
end

# =========================================================================
# ORDER 0 INTERPOLATOR (Weighted Average)
# =========================================================================

function (interp::Interpolator{D, 0, 0})(
    nb_slice::UnitRange{Int},
    wVec::AbstractVector{Float64},
    fVec::AbstractVector{State{M}}
) where {D, M}
    
    sum_w = 0.0
    sum_wf = zeros(SVector{M})

    @inbounds for i in nb_slice
        w = wVec[i]
        sum_w += w
        sum_wf += w * fVec[i]
    end

    if abs(sum_w) < 1e-14
        return zeros(SVector{M})
    else
        return sum_wf / sum_w 
    end
end

# =========================================================================
# ORDER 1 INTERPOLATOR (Linear MLS)
# =========================================================================

function (interp::Interpolator{D, 1, 1})(
    nb_slice::UnitRange{Int},
    distVec::AbstractVector{Space{D}}, 
    wVec::AbstractVector{Float64},
    dfVec::AbstractVector{State{M}};
    scale::Float64=1.0
) where {D, M}
    
    invL = 1.0 / scale
    
    N_s = @SMatrix zeros(Float64, D, D)
    b_s = zero(SMatrix{D, M, Float64, D * M})

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
    
    # Returns an SMatrix of size (D x M)
    return c_s * invL 
end

# =========================================================================
# ORDER 2 INTERPOLATOR (Quadratic MLS)
# =========================================================================

function (interp::Interpolator{D, 2, 1})(
    nb_slice::UnitRange{Int},
    distVec::AbstractVector{Space{D}},
    wVec::AbstractVector{Float64},
    dfVec::AbstractVector{State{M}};
    scale::Float64=1.0
) where {D, M}
    
    invL = 1.0 / scale
    invL2 = invL * invL
    
    # Compile-time resolution of basis size based on Dimension
    B_LEN = D == 1 ? 2 : (D == 2 ? 5 : 9)
    
    N_s = @SMatrix zeros(Float64, B_LEN, B_LEN)
    b_s = zero(SMatrix{B_LEN, M, Float64, B_LEN * M})

    @inbounds for i in nb_slice
        w = wVec[i]
        p_s = build_o2_basis(distVec[i] * invL) 
        
        N_s += w * (p_s * p_s')
        b_s += w * (p_s * dfVec[i]')
    end
    
    if abs(det(N_s)) < 1e-14
        return zero(SMatrix{D, M, Float64, D * M}), zero(SMatrix{B_LEN - D, M, Float64, (B_LEN - D) * M})
    end
    
    c_s = N_s \ b_s
    
    # Slice the SMatrix to separate first derivatives from higher-order curves
    # Row indices 1:D are the slopes. Remaining rows are curvatures.
    slopes = c_s[SOneTo(D), :] * invL
    curves = c_s[(D+1):B_LEN, :] * invL2
    
    # Both are returned as SMatrix. 
    # slopes is size (D x M). curves is size ((B_LEN - D) x M).
    return slopes, curves
end