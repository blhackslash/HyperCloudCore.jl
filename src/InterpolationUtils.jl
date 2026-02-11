module InterpolationUtils

export Interpolator

using LinearAlgebra # For dot, pinv

struct Interpolator{D, IO, DO}

    function Interpolator{D, IO, DO}() where {D, IO, DO}
        new{D, IO, DO}()
    end
end

### Bufferless versions

function (interp::Interpolator{1, 0, 0})(
    nb_slice::UnitRange{Int},
    wVec::AbstractVector{Float64},  # Full AbstractVector
    fVec::AbstractVector{Float64},  # Full AbstractVector
)
    sum_w = 0.0
    dot_wf = 0.0

    @inbounds for i in nb_slice
        w = wVec[i]
        sum_w += w
        dot_wf += w * fVec[i]
    end

    if abs(sum_w) < 1e-14
        return 0.0 # Return the mean value directly
    else
        return dot_wf / sum_w # Return the mean value directly
    end
    # Original code modified interp.res[1], removed to fit pattern
end

function (interp::Interpolator{1, 1, 0})(
    nb_slice::UnitRange{Int},
    dxVec::AbstractVector{Float64}, # Full AbstractVector
    wVec::AbstractVector{Float64},  # Full AbstractVector
    fVec::AbstractVector{Float64},  # Full AbstractVector
)
    # Accumulate matrix and RHS components in a loop
    A11 = 0.0; A12 = 0.0; A22 = 0.0
    b1 = 0.0; b2 = 0.0

    @inbounds for i in nb_slice
        w = wVec[i]
        dx = dxVec[i]
        f = fVec[i]

        # RHS components (b1 = w*f, b2 = w*dx*f)
        b1 += w * f
        wdx = w * dx # Compute w*dx once
        b2 += wdx * f

        # Matrix components (A11 = w, A12 = w*dx, A22 = w*dx*dx)
        A11 += w
        A12 += wdx
        A22 += wdx * dx
    end

    # Solve 2x2 system: [A11 A12; A12 A22] [c0; c1] = [b1; b2]
    D = A11 * A22 - A12 * A12
    if abs(D) < 1e-14
        # Handle singular matrix case (e.g., return zero coefficients)
        return 0.0, 0.0
    else
        invD = 1.0 / D
        # Cramer's rule or explicit inverse
        res0 = (A22 * b1 - A12 * b2) * invD # c0
        res1 = (A11 * b2 - A12 * b1) * invD # c1
        return res0, res1
    end
    # Original code had a slightly different solver, replaced with standard 2x2 solve
end

function (interp::Interpolator{1, 2, 0})(
    nb_slice::UnitRange{Int},
    dxVec::AbstractVector{Float64}, # Full AbstractVector
    wVec::AbstractVector{Float64},  # Full AbstractVector
    fVec::AbstractVector{Float64},  # Full AbstractVector
)
    # Accumulate matrix and RHS components
    # Basis: [1, x, x^2/2]
    A11 = 0.0; A12 = 0.0; A13 = 0.0
    A22 = 0.0; A23 = 0.0; A33 = 0.0
    b1 = 0.0; b2 = 0.0; b3 = 0.0

    @inbounds for i in nb_slice
        w = wVec[i]
        dx = dxVec[i]
        f = fVec[i]
        dx2_2 = dx * dx / 2.0 # Basis function x^2/2

        # RHS: b = A^T W f
        b1 += w * f       # Coeff for basis 1
        b2 += w * dx * f  # Coeff for basis x
        b3 += w * dx2_2 * f # Coeff for basis x^2/2

        # Matrix N = A^T W A (Upper Triangle)
        A11 += w          # (1 * 1)
        A12 += w * dx     # (1 * x)
        A13 += w * dx2_2  # (1 * x^2/2)
        A22 += w * dx * dx # (x * x)
        A23 += w * dx * dx2_2 # (x * x^2/2)
        A33 += w * dx2_2 * dx2_2 # (x^2/2 * x^2/2)
    end
    
    # Fill lower triangle (matrix is symmetric)
    # A21 = A12; A31 = A13; A32 = A23 

    # Hardcoded solve of 3x3 system (Ax=b where A is symmetric N)
    # Using LU decomposition logic from original code
    # Check for potential division by zero
    if abs(A11) < 1e-14; return 0.0, 0.0, 0.0; end
    L21 = A12 / A11
    L31 = A13 / A11
    
    U22 = A22 - L21 * A12
    if abs(U22) < 1e-14; return 0.0, 0.0, 0.0; end
    L32 = (A23 - L31 * A12) / U22
    
    U23 = A23 - L21 * A13 # Original A23 used here
    
    U33 = A33 - L31 * A13 - L32 * U23
    if abs(U33) < 1e-14; return 0.0, 0.0, 0.0; end

    # Forward substitution Ly = b
    y1 = b1
    y2 = b2 - L21 * y1
    y3 = b3 - L31 * y1 - L32 * y2
    
    # Backward substitution Ux = y
    res3 = y3 / U33          # c2 (coefficient for x^2/2)
    res2 = (y2 - U23 * res3) / U22 # c1 (coefficient for x)
    res1 = (y1 - A12 * res2 - A13 * res3) / A11 # c0 (coefficient for 1)

    return res1, res2, res3
end

function (interp::Interpolator{1, 1, 1})(
    nb_slice::UnitRange{Int},
    dxVec::AbstractVector{Float64}, # Full AbstractVector
    wVec::AbstractVector{Float64},  # Full AbstractVector
    dfVec::AbstractVector{Float64}; # Full AbstractVector
    scale::Float64=1.0             # <-- ADDED
)
    # --- 1. Precompute scaling factor ---
    invL = 1.0 / scale
    
    # --- 2. Calculate scaled matrix and RHS ---
    # Basis: p'1 = dx/L
    A11_s = 0.0 # A11_scaled
    b1_s = 0.0  # b1_scaled

    @inbounds for i in nb_slice
        w = wVec[i]
        dx_s = dxVec[i] * invL # p'1
        
        b1_s += w * dx_s * dfVec[i]
        A11_s += w * dx_s * dx_s
    end
    
    # --- 3. Solve scaled system ---
    local c1_s # c1_scaled
    if abs(A11_s) < 1e-14
        c1_s = 0.0
    else
        c1_s = b1_s / A11_s
    end
    
    # --- 4. Return unscaled physical derivative ---
    return c1_s * invL # c1 = c'1 / L
end

function (interp::Interpolator{1, 2, 1})(
    nb_slice::UnitRange{Int},
    dxVec::AbstractVector{Float64}, # Full AbstractVector
    wVec::AbstractVector{Float64},  # Full AbstractVector
    dfVec::AbstractVector{Float64}; # Full AbstractVector
    scale = 1.0
)
    # --- 1. Find a scaling factor L (characteristic length) ---
    invL = 1.0 / scale
    invL2 = invL * invL

    # --- 2. Calculate scaled matrix and RHS components ---
    # We are solving for c1, c2 in: df ~ c1*(dx/L) + c2*(dx/L)^2/2
    # N = [A11, A12; A12, A22]
    # b = [b1; b2]
    A11 = 0.0; A12 = 0.0; A22 = 0.0
    b1 = 0.0; b2 = 0.0

    @inbounds for i in nb_slice
        w = wVec[i]
        df = dfVec[i]
        dx_scaled = dxVec[i] * invL # dx_s = dx/L

        # Basis functions: p1 = dx_s, p2 = 0.5 * dx_s^2
        p1 = dx_scaled
        p2 = 0.5 * dx_scaled * dx_scaled

        # RHS vector: R = A^T W b
        b1 += w * p1 * df # R[1] = sum(w * p1 * df)
        b2 += w * p2 * df # R[2] = sum(w * p2 * df)

        # Normal Matrix: N = A^T W A
        A11 += w * p1 * p1 # N[1,1]
        A12 += w * p1 * p2 # N[1,2]
        A22 += w * p2 * p2 # N[2,2]
    end

    # --- 3. Solve N*c = b using hard-coded Cholesky ---
    
    # a) Cholesky Decomposition (N = LL^T)
    # L = [l11 0; l21 l22]
    l11_sq = A11
    if l11_sq < 1e-14; return 0.0, 0.0; end
    l11 = sqrt(l11_sq)
    inv_l11 = 1.0 / l11
    
    l21 = A12 * inv_l11
    
    # l22_sq = A22 - l21*l21
    # This is (A11*A22 - A12^2) / A11, which is D / A11
    l22_sq = A22 - l21*l21
    if l22_sq < 1e-14; return 0.0, 0.0; end
    l22 = sqrt(l22_sq)
    inv_l22 = 1.0 / l22

    # b) Forward Substitution (Ly = b)
    # [l11 0; l21 l22] * [y1; y2] = [b1; b2]
    y1 = b1 * inv_l11
    y2 = (b2 - l21*y1) * inv_l22

    # c) Backward Substitution (L^T c = y)
    # [l11 l21; 0 l22] * [c1; c2] = [y1; y2]
    res2_scaled = y2 * inv_l22
    res1_scaled = (y1 - l21*res2_scaled) * inv_l11
    
    # --- 4. Return the unscaled, physical derivatives ---
    # The true derivatives are:
    # df/dx = c1 / L
    # d2f/dx2 = c2 / L^2
    return res1_scaled * invL, res2_scaled * invL2
end

function (interp::Interpolator{2, 1, 1})(
    nb_slice::UnitRange{Int},
    dxVec::AbstractVector{Float64}, # Full AbstractVector
    dyVec::AbstractVector{Float64}, # Full AbstractVector
    wVec::AbstractVector{Float64},  # Full AbstractVector
    dfVec::AbstractVector{Float64}; # Full AbstractVector
    scale::Float64=1.0             # <-- ADDED
)
    # --- 1. Precompute scaling factor ---
    invL = 1.0 / scale

    # --- 2. Calculate scaled matrix and RHS ---
    # Basis: p'1 = dx/L, p'2 = dy/L
    A11_s = 0.0; A12_s = 0.0; A22_s = 0.0
    b1_s = 0.0; b2_s = 0.0

    @inbounds for i in nb_slice
        w = wVec[i]
        df = dfVec[i]
        p1_s = dxVec[i] * invL
        p2_s = dyVec[i] * invL
        
        A11_s += w * p1_s * p1_s
        A22_s += w * p2_s * p2_s
        A12_s += w * p1_s * p2_s
        b1_s += w * p1_s * df
        b2_s += w * p2_s * df
    end
    
    # --- 3. Solve scaled system ---
    D_s = A11_s * A22_s - A12_s^2
    local c1_s, c2_s # c1_scaled, c2_scaled
    if abs(D_s) < 1e-14
        c1_s = 0.0
        c2_s = 0.0
    else
        invD_s = 1.0 / D_s
        c1_s = (A22_s * b1_s - A12_s * b2_s) * invD_s
        c2_s = (A11_s * b2_s - A12_s * b1_s) * invD_s
    end
    
    # --- 4. Return unscaled physical derivatives ---
    return c1_s * invL, c2_s * invL # c1 = c'1/L, c2 = c'2/L
end

function (interp::Interpolator{2, 2, 1})(
    nb_slice::UnitRange{Int},
    dxVec::AbstractVector{Float64}, # Full AbstractVector
    dyVec::AbstractVector{Float64}, # Full AbstractVector
    wVec::AbstractVector{Float64},  # Full AbstractVector
    dfVec::AbstractVector{Float64}; # Full AbstractVector,
    scale::Float64=1.0      # <-- ADDED optional scaling
)
    # --- 1. Precompute scaling factors ---
    invL = 1.0 / scale
    invL2 = invL * invL

    # --- 2. Declare local variables for the 5x5 matrix (upper triangle) ---
    N11 = 0.0; N12 = 0.0; N13 = 0.0; N14 = 0.0; N15 = 0.0
    N22 = 0.0; N23 = 0.0; N24 = 0.0; N25 = 0.0
    N33 = 0.0; N34 = 0.0; N35 = 0.0
    N44 = 0.0; N45 = 0.0
    N55 = 0.0
    
    # --- 3. Declare local variables for the 5-element RHS vector ---
    b1 = 0.0; b2 = 0.0; b3 = 0.0; b4 = 0.0; b5 = 0.0

    # --- 4. Construct the Scaled Normal Matrix and RHS ---
    @inbounds for i in nb_slice
        w = wVec[i]
        df = dfVec[i]
        
        # Scale dx and dy
        dx_s = dxVec[i] * invL
        dy_s = dyVec[i] * invL
        
        # Precompute scaled basis functions
        # p'1 = dx/L
        # p'2 = dy/L
        # p'3 = (dx^2/2) / L^2
        # p'4 = (dy^2/2) / L^2
        # p'5 = (dx*dy) / L^2
        basis_1 = dx_s
        basis_2 = dy_s
        basis_3 = (dxVec[i] * dxVec[i] / 2.0) * invL2 # (dx^2/2) * invL2
        basis_4 = (dyVec[i] * dyVec[i] / 2.0) * invL2 # (dy^2/2) * invL2
        basis_5 = dx_s * dy_s                         # (dx*dy) * invL2
        
        # Update the right-hand side b' = A'^T W df
        b1 += w * basis_1 * df
        b2 += w * basis_2 * df
        b3 += w * basis_3 * df
        b4 += w * basis_4 * df
        b5 += w * basis_5 * df
        
        # Update the upper triangle of the symmetric normal matrix N' = A'^T W A'
        N11 += w * basis_1 * basis_1
        N12 += w * basis_1 * basis_2
        N13 += w * basis_1 * basis_3
        N14 += w * basis_1 * basis_4
        N15 += w * basis_1 * basis_5
        
        N22 += w * basis_2 * basis_2
        N23 += w * basis_2 * basis_3
        N24 += w * basis_2 * basis_4
        N25 += w * basis_2 * basis_5
        
        N33 += w * basis_3 * basis_3
        N34 += w * basis_3 * basis_4
        N35 += w * basis_3 * basis_5
        
        N44 += w * basis_4 * basis_4
        N45 += w * basis_4 * basis_5
        
        N55 += w * basis_5 * basis_5
    end
    
    # --- 5. Fully Hardcoded 5x5 Cholesky Solver (on N') ---
    
    # 1. Cholesky Decomposition (N' = L'L'^T)
    l11_sq = N11
    if l11_sq < 1e-14; return (0.0, 0.0, 0.0, 0.0, 0.0); end
    l11 = sqrt(l11_sq)
    inv_l11 = 1.0 / l11
    l21 = N12 * inv_l11
    l31 = N13 * inv_l11
    l41 = N14 * inv_l11
    l51 = N15 * inv_l11

    l22_sq = N22 - l21*l21
    if l22_sq < 1e-14; return (0.0, 0.0, 0.0, 0.0, 0.0); end
    l22 = sqrt(l22_sq)
    inv_l22 = 1.0 / l22
    l32 = (N23 - l31*l21) * inv_l22
    l42 = (N24 - l41*l21) * inv_l22
    l52 = (N25 - l51*l21) * inv_l22

    l33_sq = N33 - l31*l31 - l32*l32
    if l33_sq < 1e-14; return (0.0, 0.0, 0.0, 0.0, 0.0); end
    l33 = sqrt(l33_sq)
    inv_l33 = 1.0 / l33
    l43 = (N34 - l41*l31 - l42*l32) * inv_l33
    l53 = (N35 - l51*l31 - l52*l32) * inv_l33

    l44_sq = N44 - l41*l41 - l42*l42 - l43*l43
    if l44_sq < 1e-14; return (0.0, 0.0, 0.0, 0.0, 0.0); end
    l44 = sqrt(l44_sq)
    inv_l44 = 1.0 / l44
    l54 = (N45 - l51*l41 - l52*l42 - l53*l43) * inv_l44

    l55_sq = N55 - l51*l51 - l52*l52 - l53*l53 - l54*l54
    if l55_sq < 1e-14; return (0.0, 0.0, 0.0, 0.0, 0.0); end
    l55 = sqrt(l55_sq)
    inv_l55 = 1.0 / l55 
    
    # 2. Forward Substitution (solves L'y' = b')
    y1 = b1 * inv_l11
    y2 = (b2 - l21*y1) * inv_l22
    y3 = (b3 - l31*y1 - l32*y2) * inv_l33
    y4 = (b4 - l41*y1 - l42*y2 - l43*y3) * inv_l44
    y5 = (b5 - l51*y1 - l52*y2 - l53*y3 - l54*y4) * inv_l55

    # 3. Backward Substitution (solves L'^T c' = y')
    c5_s = y5 * inv_l55    # c'5
    c4_s = (y4 - l54*c5_s) * inv_l44    # c'4
    c3_s = (y3 - l43*c4_s - l53*c5_s) * inv_l33 # c'3
    c2_s = (y2 - l32*c3_s - l42*c4_s - l52*c5_s) * inv_l22 # c'2
    c1_s = (y1 - l21*c2_s - l31*c3_s - l41*c4_s - l51*c5_s) * inv_l11 # c'1

    # --- 6. Unscale the results ---
    # c_j = c'_j / L^k
    return (
        c1_s * invL,  # c1 = c'1 / L
        c2_s * invL,  # c2 = c'2 / L
        c3_s * invL2, # c3 = c'3 / L^2
        c4_s * invL2, # c4 = c'4 / L^2
        c5_s * invL2  # c5 = c'5 / L^2
    )
end

end