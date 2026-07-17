# =========================================================================
# MAXIMUM EIGENVALUES (Wave Speeds)
# =========================================================================

# --- Generalized Scalar PDEs (Fallback for any M=1 equation) ---
@inline function max_eigenvalue(eq::HyperbolicPDE{D, 1, R}, u::State{1}, d::Int) where {D, R}
    return abs(velocity(eq, u)[d])
end

# --- Linear Advection ---
# 1. Disambiguation for M = 1 (Fixes the MethodError)
@inline function max_eigenvalue(eq::LinearAdvection{D, 1, R}, U::State{1}, d::Int) where {D, R}
    return abs(eq.vel[d][1])
end

# 2. General case for Systems (M > 1)
@inline function max_eigenvalue(eq::LinearAdvection{D, M, R}, U::State{M}, d::Int) where {D, M, R}
    return maximum(abs.(eq.vel[d]))
end

# --- Euler Equation (D-Dimensional Unified) ---

# 1. Conservative Variables (ρ, m, E)
@inline function max_eigenvalue(eq::EulerEquation{D, M, <:Conservative}, U::State{M}, d::Int) where {D, M}

    rho = if U[1] < 1e-9; @debug "Vanishing/Negative Density Found! This is expected for posteriori limiter like MOOD!"; 1e-9; else; U[1] end
    m_d = U[1+d]
    E = U[M]
    
    # Calculate full kinetic energy for pressure natively in D-dimensions
    m_sq = sum(abs2, ntuple(i -> U[1+i], Val(D)))
    
    p = max((GAS_GAMMA_EULER - 1.0) * (E - 0.5 * m_sq / rho), 1e-9)
    c = sqrt(GAS_GAMMA_EULER * p / rho)
    
    return abs(m_d / rho) + c
end

# 2. Primitive/Lagrangian Variables (ρ, u, p)
@inline function max_eigenvalue(eq::EulerEquation{D, M, <:NCRepresentation}, V::State{M}, d::Int) where {D, M}
    rho = max(V[1], 1e-9)
    u_d = V[1+d]
    p = max(V[M], 1e-9)
    
    c = sqrt(GAS_GAMMA_EULER * p / rho)
    
    return abs(u_d) + c
end

# --- The Interface Aggregator ---
@inline function max_eigenvalues(eq::HyperbolicPDE{D, M, R}, f_L::Flux{D, M}, f_R::Flux{D, M}) where {D, M, R}
    return SVector{D, Float64}(ntuple(Val(D)) do d
        # Extract the d-th column state natively
        lamL = max_eigenvalue(eq, f_L[d], d)
        lamR = max_eigenvalue(eq, f_R[d], d)
        max(lamL, lamR)
    end)
end

# =========================================================================
# NUMERICAL FLUXES (Fully Unified)
# =========================================================================

@inline function (rusanov::RusanovFlux)(
    f_L::Flux{D, M}, f_R::Flux{D, M}, F_L::Flux{D, M}, F_R::Flux{D, M}, eq::HyperbolicPDE{D, M, R}
) where {D, M, R}
    
    s_vec = max_eigenvalues(eq, f_L, f_R)
    
    # Loop over dimensions and construct the Flux
    return Flux{D, M}(ntuple(Val(D)) do d
        dissipation = s_vec[d] * (f_R[d] - f_L[d])
        0.5 * (F_L[d] + F_R[d] - dissipation)
    end)
end

# System Fallback (If Upwind is called on a system, drop to Rusanov)
@inline function (upwind::UpwindFlux)(
    f_L::Flux{D, M}, f_R::Flux{D, M}, F_L::Flux{D, M}, F_R::Flux{D, M}, eq::HyperbolicPDE{D, M, R}
) where {D, M, R}
    
    return RusanovFlux()(f_L, f_R, F_L, F_R, eq)
end

@inline function (upwind::UpwindFlux)(
    f_L::Flux{D, 1}, f_R::Flux{D, 1}, F_L::Flux{D, 1}, F_R::Flux{D, 1}, eq::HyperbolicPDE{D, 1, R}
) where {D, R}
    
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

