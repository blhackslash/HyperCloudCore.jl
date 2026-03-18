# =========================================================================
# WAVE SPEED CALCULATIONS
# =========================================================================

@inline function max_wave_speed(eq::HyperbolicPDE, uL::State{M}, uR::State{M}, d::Int) where {M}
    lamL = max_eigenvalue(eq, uL, d)
    lamR = max_eigenvalue(eq, uR, d)
    return max(lamL, lamR)
end

# --- Default Eigenvalue implementations ---

# Scalars simply return the absolute velocity in dimension `d`
@inline max_eigenvalue(eq::ScalarHyperbolicPDE, u::SVector{1, Float64}, d::Int) = abs(velocity(eq, u)[d][1])

# System PDEs
@inline function max_eigenvalue(eq::Euler1D, U::SVector{3, Float64}, d::Int)
    rho, m, E = U[1], U[2], U[3]
    if rho < 1e-9; return 0.0; end
    u = m / rho
    p = max((GAS_GAMMA_EULER - 1.0) * (E - 0.5 * m^2 / rho), 1e-9)
    c = sqrt(GAS_GAMMA_EULER * p / rho)
    return abs(u) + c
end

@inline function max_eigenvalue(eq::Euler2D, U::SVector{4, Float64}, d::Int)
    rho, mx, my, E = U[1], U[2], U[3], U[4]
    if rho < 1e-9; return 0.0; end
    u_n = d == 1 ? mx / rho : my / rho
    p = max((GAS_GAMMA_EULER - 1.0) * (E - 0.5 * (mx^2 + my^2) / rho), 1e-9)
    c = sqrt(GAS_GAMMA_EULER * p / rho)
    return abs(u_n) + c
end

@inline function max_eigenvalue(eq::LinearAdvection, U::State{M}, d::Int) where {M}
    return maximum(abs.(eq.vel[d]))
end

# =========================================================================
# NUMERICAL FLUXES (Fully Unified)
# =========================================================================

# ---------------------------------------------------------
# RUSANOV FLUX (Handles everything natively via broadcasting!)
# ---------------------------------------------------------
@inline function (rusanov::RusanovFlux)(fL::State{M}, fR::State{M}, eq::HyperbolicPDE, d::Int) where {M}
    F_L = flux(eq, fL)[d] # Extracts SVector{M} for dimension d
    F_R = flux(eq, fR)[d]
    s = max_wave_speed(eq, fL, fR, d)
    
    # SVector broadcasting makes this a single line for both 1D and Multi-D systems!
    return @. 0.5 * (F_L + F_R - s * (fR - fL))
end

# ---------------------------------------------------------
# UPWIND FLUX
# ---------------------------------------------------------
# Scalar Dispatch
@inline function (upwind::UpwindFlux)(fL::SVector{1, Float64}, fR::SVector{1, Float64}, eq::ScalarHyperbolicPDE, d::Int)
    F_L = flux(eq, fL)[d]
    F_R = flux(eq, fR)[d]
    
    uL, uR = fL[1], fR[1]
    
    if uL == uR
        a = velocity(eq, fL)[d][1]
    else
        a = (F_L[1] - F_R[1]) / (uL - uR)
    end
    
    return SVector{1, Float64}(0.5 * (F_L[1] + F_R[1] - abs(a) * (uR - uL)))
end

# System Dispatch (Systems fall back to Rusanov for safety)
@inline function (upwind::UpwindFlux)(fL::State{M}, fR::State{M}, eq::HyperbolicPDESystem, d::Int=1) where {M}
    return RusanovFlux()(fL, fR, eq, d)
end