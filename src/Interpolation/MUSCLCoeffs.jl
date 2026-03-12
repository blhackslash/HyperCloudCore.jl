@inline function _zero_coeffs!(nb_slice::UnitRange{Int}, ws::MUSCLWorkspace2D1O)
    @inbounds for k in nb_slice
        ws.alfaijs[k] = 0.0
        ws.betaijs[k] = 0.0
    end
end
@inline function _zero_coeffs!(nb_slice::UnitRange{Int}, ws::MUSCLWorkspace2D2O)
    @inbounds for k in nb_slice
        ws.alfaijs[k] = 0.0
        ws.betaijs[k] = 0.0
        ws.alfaij_bars[k] = 0.0
        ws.betaij_bars[k] = 0.0
        ws.gammaijs[k] = 0.0
    end
end
# --- In MUSCLCoeffs.jl ---

# --- 1D _zero_coeffs! ---
@inline function _zero_coeffs!(nb_slice::UnitRange{Int}, ws::Union{MUSCLWorkspace1D2O,MUSCLWorkspace1D1O})
    @inbounds for k in nb_slice
        ws.alfaij_bars[k] = 0.0
        ws.betaijs[k] = 0.0
    end
end
@inline function _zero_coeffs!(nb_slice::UnitRange{Int}, ws::MUSCLWorkspace1D3O)
    @inbounds for k in nb_slice
        ws.alfaijs[k] = 0.0
        ws.alfaij_bars[k] = 0.0
        ws.betaijs[k] = 0.0
    end
end
@inline function _zero_coeffs!(nb_slice::UnitRange{Int}, ws::MUSCLWorkspace1D4O)
    @inbounds for k in nb_slice
        ws.alfaijs[k] = 0.0
        ws.alfaij_bars[k] = 0.0
        ws.betaijs[k] = 0.0
        ws.gammaijs[k] = 0.0
    end
end

# --- In MUSCLCoeffs.jl ---

# 1D Order 0: Same as Order 1
function _compute_coeffs!(
    ::MUSCLORDER0,
    nb_slice::UnitRange{Int},
    ws::MUSCLWorkspace1D0O,
    pg::ParticleGrid1D
)
    # Re-use the Linear Least Squares logic from Order 1
    # This computes the weights 'c' such that sum(c * dx) = 1
    dx = get_xdistance(pg)
    w = get_weights(pg)
    
    N11 = 0.0
    @inbounds for k in nb_slice
        N11 += w[k] * dx[k]^2
    end
    
    if N11 < 1e-14
        @inbounds for k in nb_slice; ws.alfaij_bars[k] = 0.0; end
        return
    end
    invN11 = 1.0 / N11

    @inbounds for k in nb_slice
        ws.alfaij_bars[k] = (dx[k] * w[k]) * invN11
    end
end

# 2D Order 0: Same as Order 1
function _compute_coeffs!(
    ::MUSCLORDER0,
    nb_slice::UnitRange{Int},
    ws::MUSCLWorkspace2D0O,
    pg::ParticleGrid2D
)
    # Re-use the Linear Least Squares logic from Order 1
    dx = get_xdistance(pg)
    dy = get_ydistance(pg)
    w = get_weights(pg)
    
    N11 = 0.0; N12 = 0.0; N22 = 0.0

    @inbounds for k in nb_slice
        wk = w[k]
        if wk == 0.0; continue; end
        dxk = dx[k]; dyk = dy[k]
        N11 += wk * dxk^2
        N12 += wk * dxk * dyk
        N22 += wk * dyk^2
    end

    D = N11*N22 - N12^2
    if D < 1e-14
        @inbounds for k in nb_slice
            ws.alfaijs[k] = 0.0
            ws.betaijs[k] = 0.0
        end
        return
    end
    invD = 1.0 / D

    @inbounds for k in nb_slice
        wk = w[k]
        dxk = dx[k]; dyk = dy[k]
        
        b1 = dxk * wk
        b2 = dyk * wk
        
        # Cramer's rule / Inverse for 2x2
        # c = N^-1 * b
        # c1 = (N22*b1 - N12*b2) / D
        # c2 = (N11*b2 - N12*b1) / D
        
        ws.alfaijs[k] = (N22 * b1 - N12 * b2) * invD
        ws.betaijs[k] = (N11 * b2 - N12 * b1) * invD
    end
end

# --- ORDER 1 (1D) ---
function _compute_coeffs!(
    ::MUSCLORDER1,
    nb_slice::UnitRange{Int},
    ws::MUSCLWorkspace1D1O,
    pg::ParticleGrid1D
)
    dx = get_xdistance(pg)
    w = get_weights(pg)
    
    # 1. Build 1x1 Normal Matrix N = A^T W A
    N11 = 0.0
    @inbounds for k in nb_slice
        w_k = w[k]
        dx_k = dx[k]
        N11 += w_k * dx_k * dx_k
    end
    
    if N11 < 1e-14
        _zero_coeffs!(nb_slice, ws) # Use helper to zero coeffs
        return
    end
    invN11 = 1.0 / N11

    
    # 2. Solve N*c = b for each neighbor
    @inbounds for k in nb_slice
        b1 = dx[k] * w[k]
        ws.alfaij_bars[k] = invN11 * b1 # Direct assignment
    end
end

# --- ORDER 2 (1D) ---
function _compute_coeffs!(
    ::MUSCLORDER2,
    nb_slice::UnitRange{Int},
    ws::MUSCLWorkspace1D2O,
    pg::ParticleGrid1D
)
    dx = get_xdistance(pg)
    w = get_weights(pg)

    N11 = 0.0; N12 = 0.0; N22 = 0.0
    @inbounds for k in nb_slice
        w_k = w[k]
        dx_k = dx[k]
        p1 = dx_k; p2 = 0.5 * dx_k * dx_k
        N11 += w_k*p1*p1; N12 += w_k*p1*p2; N22 += w_k*p2*p2
    end
    
    l11_sq = N11
    if l11_sq < 1e-14; _zero_coeffs!(nb_slice, ws); return; end
    l11 = sqrt(l11_sq); inv_l11 = 1.0 / l11
    l21 = N12 * inv_l11
    l22_sq = N22 - l21*l21
    if l22_sq < 1e-14; _zero_coeffs!(nb_slice, ws); return; end
    l22 = sqrt(l22_sq); inv_l22 = 1.0 / l22

    @inbounds for k in nb_slice
        w_k = w[k]
        dx_k = dx[k]
        p1_k = dx_k; p2_k = 0.5 * dx_k * dx_k
        b1 = p1_k * w_k; b2 = p2_k * w_k
        y1 = b1 * inv_l11; y2 = (b2 - l21*y1) * inv_l22
        c2 = y2 * inv_l22; c1 = (y1 - l21*c2) * inv_l11
        
        # Direct assignment
        ws.alfaij_bars[k] = c1
        ws.betaijs[k]     = c2
    end
end

# --- ORDER 3 (1D) ---
function _compute_coeffs!(
    ::MUSCLORDER3,
    nb_slice::UnitRange{Int},
    ws::MUSCLWorkspace1D3O,
    pg::ParticleGrid1D
)
    dx = get_xdistance(pg)
    w = get_weights(pg)

    # --- 1. Get Scaling Factor ---
    h_scale = pg.meta.dx
    if h_scale < 1e-14
        _zero_coeffs!(nb_slice, ws); return;
    end
    
    h_inv = 1.0 / h_scale
    h_inv2 = h_inv * h_inv
    h_inv3 = h_inv2 * h_inv

    # --- 2. Build Scaled Normal Matrix N' ---
    N11=0.0; N12=0.0; N13=0.0; N22=0.0; N23=0.0; N33=0.0
    @inbounds for k in nb_slice
        w_k = w[k]; dx_k = dx[k]; dx_k2 = dx_k*dx_k
        
        # Original basis functions
        p1 = dx_k
        p2 = 0.5*dx_k2
        p3 = (1/6)*dx_k2*dx_k
        
        # Scaled basis functions (p'_j = p_j / h^j)
        p1s = p1 * h_inv
        p2s = p2 * h_inv2
        p3s = p3 * h_inv3
        
        # N'_ij = sum(w_k * p'_i * p'_j)
        N11 += w_k*p1s*p1s; N12 += w_k*p1s*p2s; N13 += w_k*p1s*p3s
        N22 += w_k*p2s*p2s; N23 += w_k*p2s*p3s; N33 += w_k*p3s*p3s
    end

    # --- 3. Cholesky Decomposition on N' ---
    l11_sq=N11; if l11_sq<1e-14; _zero_coeffs!(nb_slice, ws); return; end
    l11 = sqrt(l11_sq); inv_l11 = 1/l11; l21 = N12*inv_l11; l31 = N13*inv_l11
    
    l22_sq=N22-l21*l21; if l22_sq<1e-14; _zero_coeffs!(nb_slice, ws); return; end
    l22 = sqrt(l22_sq); inv_l22 = 1/l22; l32 = (N23-l31*l21)*inv_l22
    
    l33_sq=N33-l31*l31-l32*l32; if l33_sq<1e-14; _zero_coeffs!(nb_slice, ws); return; end
    l33 = sqrt(l33_sq); inv_l33 = 1/l33

    # --- 4. Solve for Scaled c' and Unscale to c ---
    @inbounds for k in nb_slice
        w_k = w[k]; dx_k = dx[k]; dx_k2 = dx_k*dx_k

        # Original basis functions
        p1 = dx_k
        p2 = 0.5*dx_k2
        p3 = (1/6)*dx_k2*dx_k

        # Scaled basis functions (p'_j = p_j / h^j)
        p1s = p1 * h_inv
        p2s = p2 * h_inv2
        p3s = p3 * h_inv3
        
        # Scaled RHS vector b'_k = [p'_1*w_k, p'_2*w_k, p'_3*w_k]
        b1s = p1s*w_k; b2s = p2s*w_k; b3s = p3s*w_k
        
        # Solve N'c' = b' (where N' = L'L'^T)
        # a) Forward sub (L'y' = b')
        y1 = b1s*inv_l11; y2 = (b2s-l21*y1)*inv_l22; y3 = (b3s-l31*y1-l32*y2)*inv_l33
        # b) Backward sub (L'^T c' = y')
        c3s = y3*inv_l33; c2s = (y2-l32*c3s)*inv_l22; c1s = (y1-l21*c2s-l31*c3s)*inv_l11
        
        # Unscale coefficients (c_j = c'_j / h^j)
        ws.alfaij_bars[k] = c1s * h_inv  # c1
        ws.betaijs[k]     = c2s * h_inv2 # c2
        ws.alfaijs[k]     = c3s * h_inv3 # c3
    end
end
# --- ORDER 4 (1D) ---
function _compute_coeffs!(
    ::MUSCLORDER4,
    nb_slice::UnitRange{Int},
    ws::MUSCLWorkspace1D4O,
    pg::ParticleGrid1D
)
    dx = get_xdistance(pg)
    w = get_weights(pg)

    # --- 1. Get Scaling Factor ---
    h_scale = pg.meta.dx
    if h_scale < 1e-14
        _zero_coeffs!(nb_slice, ws); return;
    end
    
    h_inv = 1.0 / h_scale
    h_inv2 = h_inv * h_inv
    h_inv3 = h_inv2 * h_inv
    h_inv4 = h_inv3 * h_inv

    # --- 2. Build Scaled Normal Matrix N' ---
    N11=0.0; N12=0.0; N13=0.0; N14=0.0; N22=0.0; N23=0.0; N24=0.0; N33=0.0; N34=0.0; N44=0.0
    @inbounds for k in nb_slice
        w_k = w[k]; dx_k = dx[k]; dx_k2 = dx_k*dx_k; dx_k3 = dx_k2*dx_k
        
        # Original basis functions
        p1 = dx_k
        p2 = 0.5*dx_k2
        p3 = (1/6)*dx_k3
        p4 = (1/24)*dx_k3*dx_k # (1/24)*dx_k2*dx_k2
        
        # Scaled basis functions (p'_j = p_j / h^j)
        p1s = p1 * h_inv
        p2s = p2 * h_inv2
        p3s = p3 * h_inv3
        p4s = p4 * h_inv4
        
        # N'_ij = sum(w_k * p'_i * p'_j)
        N11 += w_k*p1s*p1s; N12 += w_k*p1s*p2s; N13 += w_k*p1s*p3s; N14 += w_k*p1s*p4s
        N22 += w_k*p2s*p2s; N23 += w_k*p2s*p3s; N24 += w_k*p2s*p4s
        N33 += w_k*p3s*p3s; N34 += w_k*p3s*p4s; N44 += w_k*p4s*p4s
    end

    # --- 3. Cholesky Decomposition on N' ---
    l11_sq=N11; if l11_sq<1e-14; _zero_coeffs!(nb_slice, ws); return; end
    l11 = sqrt(l11_sq); inv_l11 = 1/l11; l21=N12*inv_l11; l31=N13*inv_l11; l41=N14*inv_l11
    
    l22_sq=N22-l21*l21; if l22_sq<1e-14; _zero_coeffs!(nb_slice, ws); return; end
    l22 = sqrt(l22_sq); inv_l22 = 1/l22; l32=(N23-l31*l21)*inv_l22; l42=(N24-l41*l21)*inv_l22
    
    l33_sq=N33-l31*l31-l32*l32; if l33_sq<1e-14; _zero_coeffs!(nb_slice, ws); return; end
    l33 = sqrt(l33_sq); inv_l33 = 1/l33; l43=(N34-l41*l31-l42*l32)*inv_l33
    
    l44_sq=N44-l41*l41-l42*l42-l43*l43; if l44_sq<1e-14; _zero_coeffs!(nb_slice, ws); return; end
    l44 = sqrt(l44_sq); inv_l44 = 1/l44

    # --- 4. Solve for Scaled c' and Unscale to c ---
    @inbounds for k in nb_slice
        w_k = w[k]; dx_k = dx[k]; dx_k2 = dx_k*dx_k; dx_k3 = dx_k2*dx_k

        # Original basis functions
        p1 = dx_k
        p2 = 0.5*dx_k2
        p3 = (1/6)*dx_k3
        p4 = (1/24)*dx_k3*dx_k
        
        # Scaled basis functions (p'_j = p_j / h^j)
        p1s = p1 * h_inv
        p2s = p2 * h_inv2
        p3s = p3 * h_inv3
        p4s = p4 * h_inv4
        
        # Scaled RHS vector b'_k
        b1s=p1s*w_k; b2s=p2s*w_k; b3s=p3s*w_k; b4s=p4s*w_k
        
        # Solve N'c' = b'
        # a) Forward sub (L'y' = b')
        y1=b1s*inv_l11; y2=(b2s-l21*y1)*inv_l22; y3=(b3s-l31*y1-l32*y2)*inv_l33; y4=(b4s-l41*y1-l42*y2-l43*y3)*inv_l44
        # b) Backward sub (L'^T c' = y')
        c4s=y4*inv_l44; c3s=(y3-l43*c4s)*inv_l33; c2s=(y2-l32*c3s-l42*c4s)*inv_l22; c1s=(y1-l21*c2s-l31*c3s-l41*c4s)*inv_l11
        
        # Unscale coefficients (c_j = c'_j / h^j)
        ws.alfaij_bars[k] = c1s * h_inv
        ws.betaijs[k]     = c2s * h_inv2
        ws.alfaijs[k]     = c3s * h_inv3
        ws.gammaijs[k]    = c4s * h_inv4
    end
end

# --- In MUSCL.jl ---

"""
(2D Order 1) Calculates coefficients (alfaij, betaij) using Cholesky.
"""
function _compute_coeffs!(
    ::MUSCLORDER1,
    nb_slice::UnitRange{Int},
    ws::MUSCLWorkspace2D1O, # <-- Takes workspace
    pg::ParticleGrid2D
)
    dx = get_xdistance(pg)
    dy = get_ydistance(pg)
    w = get_weights(pg)
    # Get views into workspace coefficient arrays
    alfaij = ws.alfaijs
    betaij = ws.betaijs

    # --- 1. Build the 2x2 Normal Matrix N = A^T W A ---
    N11 = 0.0; N12 = 0.0; N22 = 0.0

    @inbounds for k in nb_slice
        w_k = w[k]
        if w_k == 0.0; continue; end

        dx_k = dx[k]
        dy_k = dy[k]

        # Basis functions
        b1 = dx_k
        b2 = dy_k

        # Add contribution to upper triangle of N
        N11 += w_k * b1 * b1
        N12 += w_k * b1 * b2
        N22 += w_k * b2 * b2
    end

    # --- 2. Hardcoded Cholesky Decomposition (N = LLᵀ) ---
    # L = [l11  0 ]
    #     [l21 l22]

    l11_sq = N11
    if l11_sq < 1e-14
        fill!(alfaij, 0.0)
        fill!(betaij, 0.0)
        return
    end
    l11 = sqrt(l11_sq); inv_l11 = 1.0 / l11
    l21 = N12 * inv_l11

    l22_sq = N22 - l21*l21
    if l22_sq < 1e-14
        fill!(alfaij, 0.0)
        fill!(betaij, 0.0)
        return
    end
    l22 = sqrt(l22_sq); inv_l22 = 1.0 / l22

    # --- 3. Solve for Coefficients for EACH neighbor ---
    @inbounds for k in nb_slice
        w_k = w[k]

        # Build the RHS vector b_k = (A^T W)_k
        dx_k = dx[k]
        dy_k = dy[k]

        b1 = dx_k * w_k
        b2 = dy_k * w_k

        # Solve N*c = b  (where N = LLT)
        # a) Forward Substitution (Ly = b)
        # y1 = b1 / l11
        # y2 = (b2 - l21*y1) / l22
        y1 = b1 * inv_l11
        y2 = (b2 - l21*y1) * inv_l22

        # b) Backward Substitution (Lᵀc = y)
        # L^T = [l11 l21]
        #       [ 0  l22]
        # c2 = y2 / l22
        # c1 = (y1 - l21*c2) / l11
        c2 = y2 * inv_l22
        c1 = (y1 - l21*c2) * inv_l11

        # Store coefficients in the workspace views
        alfaij[k] = c1
        betaij[k] = c2
    end
end

# --- REPLACED _compute_coeffs! for 2D Order 2 ---
"""
(2D Order 2) Calculates coefficients using hard-coded Cholesky.
"""
function _compute_coeffs!(
    ::MUSCLORDER2,
    nb_slice::UnitRange{Int},
    ws::MUSCLWorkspace2D2O, # <-- Takes workspace
    pg::ParticleGrid2D
)
    dx = get_xdistance(pg)
    dy = get_ydistance(pg)
    w = get_weights(pg)
    # Get views into workspace coefficient arrays
    alfaij      = ws.alfaijs
    betaij      = ws.betaijs
    alfaij_bar  = ws.alfaij_bars
    betaij_bar  = ws.betaij_bars
    gammaij     = ws.gammaijs

    # --- 1. Build the 5x5 Normal Matrix N = A^T W A ---
    N11 = 0.0; N12 = 0.0; N13 = 0.0; N14 = 0.0; N15 = 0.0
    N22 = 0.0; N23 = 0.0; N24 = 0.0; N25 = 0.0
    N33 = 0.0; N34 = 0.0; N35 = 0.0
    N44 = 0.0; N45 = 0.0
    N55 = 0.0

    @inbounds for k in nb_slice
        w_k = w[k]
        if w_k == 0.0; continue; end

        dx_k = dx[k]
        dy_k = dy[k]

        # Basis functions
        b1 = dx_k
        b2 = dy_k
        b3 = 0.5 * dx_k^2
        b4 = 0.5 * dy_k^2
        b5 = dx_k * dy_k

        # Add contribution to upper triangle of N = A^T W A
        N11 += w_k * b1 * b1; N12 += w_k * b1 * b2; N13 += w_k * b1 * b3
        N14 += w_k * b1 * b4; N15 += w_k * b1 * b5
        N22 += w_k * b2 * b2; N23 += w_k * b2 * b3; N24 += w_k * b2 * b4
        N25 += w_k * b2 * b5
        N33 += w_k * b3 * b3; N34 += w_k * b3 * b4; N35 += w_k * b3 * b5
        N44 += w_k * b4 * b4; N45 += w_k * b4 * b5
        N55 += w_k * b5 * b5
    end

    # --- 2. Hardcoded Cholesky Decomposition (N = LLᵀ) ---
    l11_sq = N11
    if l11_sq < 1e-14; _zero_coeffs!(nb_slice, ws); return; end
    l11 = sqrt(l11_sq); inv_l11 = 1.0 / l11
    l21 = N12 * inv_l11
    l31 = N13 * inv_l11
    l41 = N14 * inv_l11
    l51 = N15 * inv_l11

    l22_sq = N22 - l21*l21
    if l22_sq < 1e-14; _zero_coeffs!(nb_slice, ws); return; end
    l22 = sqrt(l22_sq); inv_l22 = 1.0 / l22
    l32 = (N23 - l31*l21) * inv_l22
    l42 = (N24 - l41*l21) * inv_l22
    l52 = (N25 - l51*l21) * inv_l22

    l33_sq = N33 - l31*l31 - l32*l32
    if l33_sq < 1e-14; _zero_coeffs!(nb_slice, ws); return; end
    l33 = sqrt(l33_sq); inv_l33 = 1.0 / l33
    l43 = (N34 - l41*l31 - l42*l32) * inv_l33
    l53 = (N35 - l51*l31 - l52*l32) * inv_l33

    l44_sq = N44 - l41*l41 - l42*l42 - l43*l43
    if l44_sq < 1e-14; _zero_coeffs!(nb_slice, ws); return; end
    l44 = sqrt(l44_sq); inv_l44 = 1.0 / l44
    l54 = (N45 - l51*l41 - l52*l42 - l53*l43) * inv_l44

    l55_sq = N55 - l51*l51 - l52*l52 - l53*l53 - l54*l54
    if l55_sq < 1e-14; _zero_coeffs!(nb_slice, ws); return; end
    l55 = sqrt(l55_sq); inv_l55 = 1.0 / l55

    # --- 3. Solve for Coefficients for EACH neighbor ---
    @inbounds for k in nb_slice
        w_k = w[k]

        # Build the RHS vector b_k = (A^T W)_k = A_k^T * w_k
        dx_k = dx[k]
        dy_k = dy[k]

        b1 = dx_k * w_k
        b2 = dy_k * w_k
        b3 = (0.5 * dx_k^2) * w_k
        b4 = (0.5 * dy_k^2) * w_k
        b5 = (dx_k * dy_k) * w_k

        # Solve N*c = b
        # a) Forward Substitution (Ly = b)
        y1 = b1 * inv_l11
        y2 = (b2 - l21*y1) * inv_l22
        y3 = (b3 - l31*y1 - l32*y2) * inv_l33
        y4 = (b4 - l41*y1 - l42*y2 - l43*y3) * inv_l44
        y5 = (b5 - l51*y1 - l52*y2 - l53*y3 - l54*y4) * inv_l55

        # b) Backward Substitution (Lᵀc = y)
        c5 = y5 * inv_l55
        c4 = (y4 - l54*c5) * inv_l44
        c3 = (y3 - l43*c4 - l53*c5) * inv_l33
        c2 = (y2 - l32*c3 - l42*c4 - l52*c5) * inv_l22
        c1 = (y1 - l21*c2 - l31*c3 - l41*c4 - l51*c5) * inv_l11

        # Store coefficients in the workspace views
        alfaij[k]     = c1
        betaij[k]     = c2
        alfaij_bar[k] = c3
        betaij_bar[k] = c4
        gammaij[k]    = c5
    end
end
