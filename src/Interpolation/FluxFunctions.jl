# =========================================================================
# HELPER FUNCTIONS FOR UNIFIED DIRECTIONAL ACCESS
# =========================================================================

# --- Directional Flux Extractors ---
# Scalar 1D (Returns SVector{1, Float64})
@inline _directional_flux(eq::ScalarHyperbolicPDE{1}, u::SVector{1, Float64}, d::Int) = flux(eq, u)
# Scalar Multi-D (Extracts the d-th dimension and repackages as SVector{1, Float64})
@inline _directional_flux(eq::ScalarHyperbolicPDE{D}, u::SVector{1, Float64}, d::Int) where {D} = SVector{1, Float64}(flux(eq, u)[d])

# System 1D (Returns SVector{NM, Float64})
@inline _directional_flux(eq::HyperbolicPDESystem{1}, U::SVector{NM, Float64}, d::Int) where {NM} = flux(eq, U)
# System Multi-D (Extracts the SVector{NM, Float64} for dimension d)
@inline _directional_flux(eq::HyperbolicPDESystem{D}, U::SVector{NM, Float64}, d::Int) where {D, NM} = flux(eq, U)[d]


# --- Directional Velocity Extractors (Scalar Only) ---
@inline _directional_velocity(eq::ScalarHyperbolicPDE{1}, u::SVector{1, Float64}, d::Int) = velocity(eq, u)[1]
@inline _directional_velocity(eq::ScalarHyperbolicPDE{D}, u::SVector{1, Float64}, d::Int) where {D} = velocity(eq, u)[d]


# =========================================================================
# WAVE SPEED CALCULATIONS
# =========================================================================

# Scalar PDEs 
@inline function max_wave_speed(eq::ScalarHyperbolicPDE, uL::SVector{1, Float64}, uR::SVector{1, Float64}, d::Int)
    vL = _directional_velocity(eq, uL, d)
    vR = _directional_velocity(eq, uR, d)
    return max(abs(vL), abs(vR))
end

# System PDEs 
@inline function max_wave_speed(eq::HyperbolicPDESystem, UL::SVector{NM, Float64}, UR::SVector{NM, Float64}, d::Int) where {NM}
    # Notice we pass the SVector directly into max_eigenvalue now!
    lamL = max_eigenvalue(eq, UL, d) 
    lamR = max_eigenvalue(eq, UR, d)
    return max(lamL, lamR)
end

# --- Default Eigenvalue implementations for Euler Equations ---
@inline function max_eigenvalue(eq::Euler1D, U::Tuple, d::Int)
    rho, m, E = U
    if rho < 1e-9; return 0.0; end
    u = m / rho
    p = pressure_from_euler_conserved(rho, m, E)
    c = sqrt(GAS_GAMMA_EULER * p / rho)
    return abs(u) + c
end

@inline function max_eigenvalue(eq::Euler2D, U::Tuple, d::Int)
    rho, mx, my, E = U
    if rho < 1e-9; return 0.0; end
    # Get velocity along the required dimension (d=1 is X, d=2 is Y)
    u_n = d == 1 ? mx / rho : my / rho
    p = pressure_from_euler_conserved(U)
    c = sqrt(GAS_GAMMA_EULER * p / rho)
    return abs(u_n) + c
end


# =========================================================================
# NUMERICAL FLUXES (Unified for Scalars & Systems, 1D & Multi-D)
# =========================================================================

# ---------------------------------------------------------
# RUSANOV FLUX
# ---------------------------------------------------------

# Scalar Dispatch
@inline function (rusanov::RusanovFlux)(fL::SVector{1, Float64}, fR::SVector{1, Float64}, eq::ScalarHyperbolicPDE, d::Int=1)
    uL = fL[1]
    uR = fR[1]
    
    fx_L = _directional_flux(eq, uL, d)
    fx_R = _directional_flux(eq, uR, d)
    s = max_wave_speed(eq, uL, uR, d)
    
    return SVector{1, Float64}(0.5 * (fx_L + fx_R - s * (uR - uL)))
end

# System Dispatch (Calculates flux for all NM variables simultaneously)
@inline function (rusanov::RusanovFlux)(fL::SVector{NM, Float64}, fR::SVector{NM, Float64}, eq::HyperbolicPDESystem, d::Int=1) where {NM}

    F_L = _directional_flux(eq, fL, d)
    F_R = _directional_flux(eq, fR, d)
    s = max_wave_speed(eq, fL, fR, d)
    
    return SVector{NM, Float64}(ntuple(c -> 0.5 * (F_L[c] + F_R[c] - s * (fR[c] - fL[c])), Val(NM)))
end


# ---------------------------------------------------------
# UPWIND FLUX
# ---------------------------------------------------------

# Scalar Dispatch
@inline function (upwind::UpwindFlux)(fL::SVector{1, Float64}, fR::SVector{1, Float64}, eq::ScalarHyperbolicPDE, d::Int=1)
    uL = fL[1]
    uR = fR[1]
    
    fx_L = _directional_flux(eq, uL, d)
    fx_R = _directional_flux(eq, uR, d)
    
    # Calculate local wave speed (a)
    if uL == uR
        a = _directional_velocity(eq, uL, d)
    else
        a = (fx_L - fx_R) / (uL - uR)
    end
    
    return SVector{1, Float64}(0.5 * (fx_L + fx_R - abs(a) * (uR - uL)))
end

# System Dispatch
@inline function (upwind::UpwindFlux)(fL::SVector{NM, Float64}, fR::SVector{NM, Float64}, eq::HyperbolicPDESystem, d::Int=1) where {NM}
    # True upwinding for systems requires Roe-averaging or full eigensystem decomposition.
    # As a safe fallback for systems, we automatically route to Rusanov.
    return RusanovFlux()(fL, fR, eq, d)
end