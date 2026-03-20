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
@inline max_eigenvalue(eq::ScalarHyperbolicPDE, u::SVector{1, Float64}, d::Int) = maximum(abs.(eq.vel[d]))

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
@inline function max_eigenvalues(eq::HyperbolicPDE{D, M}, f_L::Flux{D, M}, f_R::Flux{D, M}) where {D, M}
    return SVector{D, Float64}(ntuple(Val(D)) do d
        # Extract the d-th column state
        lamL = max_eigenvalue(eq, f_L[d], d)
        lamR = max_eigenvalue(eq, f_R[d], d)
        max(lamL, lamR)
    end)
end
# =========================================================================
# NUMERICAL FLUXES (Fully Unified)
# =========================================================================
@inline function (rusanov::RusanovFlux)(f_L::Flux{D, M}, f_R::Flux{D, M}, F_L::Flux{D, M}, F_R::Flux{D, M}, eq::HyperbolicPDE{D, M}) where {D, M}
    s_vec = max_eigenvalues(eq, f_L, f_R)
    
    # Loop over dimensions and construct the Flux
    return Flux{D, M}(ntuple(Val(D)) do d
        dissipation = s_vec[d] * (f_R[d] - f_L[d])
        0.5 * (F_L[d] + F_R[d] - dissipation)
    end)
end

@inline function (upwind::UpwindFlux)(f_L::Flux{D, 1}, f_R::Flux{D, 1}, F_L::Flux{D, 1}, F_R::Flux{D, 1}, eq::ScalarHyperbolicPDE{D}) where {D}
    return Flux{D, 1}(ntuple(Val(D)) do d
        du = f_R[d][1] - f_L[d][1]
        
        if abs(du) < 1e-14
            s = abs(velocity(eq, f_L[d])[d])
        else
            s = abs((F_R[d][1] - F_L[d][1]) / du)
        end
        
        0.5 * (F_L[d] + F_R[d] - s * (f_R[d] - f_L[d]))
    end)
end

# System Fallback
@inline function (upwind::UpwindFlux)(f_L::Flux{D, M}, f_R::Flux{D, M}, F_L::Flux{D, M}, F_R::Flux{D, M}, eq::HyperbolicPDESystem{D, M}) where {D, M}
    return RusanovFlux()(f_L, f_R, F_L, F_R, eq)
end