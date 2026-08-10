# =========================================================================
# MAXIMUM EIGENVALUES (Wave Speeds)
# =========================================================================

# --- Generalized Scalar PDEs (Fallback for any M=1 equation) ---
@inline function max_eigenvalue(eq::HyperbolicPDE{D, 1, T, R}, u::State{1, T}, d::Int) where {D, T, R}
    return abs(velocity(eq, u)[d])
end

# --- Linear Advection ---
# 1. Disambiguation for M = 1
@inline function max_eigenvalue(eq::LinearAdvection{D, 1, T, R}, U::State{1, T}, d::Int) where {D, T, R}
    return abs(eq.vel[d][1])
end

# 2. General case for Systems (M > 1)
@inline function max_eigenvalue(eq::LinearAdvection{D, M, T, R}, U::State{M, T}, d::Int) where {D, M, T, R}
    return maximum(abs.(eq.vel[d]))
end

# --- Euler Equation (D-Dimensional Unified) ---

# 1. Conservative Variables (ρ, m, E)
@inline function max_eigenvalue(eq::EulerEquation{D, M, T, <:Conservative}, U::State{M, T}, d::Int) where {D, M, T}
    rho = if U[1] < T(1e-9); @debug "Vanishing/Negative Density Found!"; T(1e-9); else; U[1] end
    m_d = U[1+d]
    E = U[M]
    
    m_sq = sum(abs2, ntuple(i -> U[1+i], Val(D)))
    
    p = max((eq.gamma - one(T)) * (E - T(0.5) * m_sq / rho), T(1e-9))
    c = sqrt(eq.gamma * p / rho)
    
    return abs(m_d / rho) + c
end

# 2. Primitive/Lagrangian Variables (ρ, u, p)
@inline function max_eigenvalue(eq::EulerEquation{D, M, T, <:NCRepresentation}, V::State{M, T}, d::Int) where {D, M, T}
    rho = max(V[1], T(1e-9))
    u_d = V[1+d]
    p = max(V[M], T(1e-9))
    
    c = sqrt(eq.gamma * p / rho)
    
    return abs(u_d) + c
end

# --- The Interface Aggregator ---
@inline function max_eigenvalues(eq::HyperbolicPDE{D, M, T, R}, f_L::Flux{D, M, T}, f_R::Flux{D, M, T}) where {D, M, T, R}
    return SVector{D, T}(ntuple(Val(D)) do d
        lamL = max_eigenvalue(eq, f_L[d], d)
        lamR = max_eigenvalue(eq, f_R[d], d)
        max(lamL, lamR)
    end)
end

# =========================================================================
# NUMERICAL FLUXES (Fully Unified)
# =========================================================================

@inline function (rusanov::RusanovFlux)(
    f_L::Flux{D, M, T}, f_R::Flux{D, M, T}, F_L::Flux{D, M, T}, F_R::Flux{D, M, T}, eq::HyperbolicPDE{D, M, T, R}
) where {D, M, T, R}
    
    s_vec = max_eigenvalues(eq, f_L, f_R)
    
    return Flux{D, M, T}(ntuple(Val(D)) do d
        dissipation = s_vec[d] * (f_R[d] - f_L[d])
        T(0.5) * (F_L[d] + F_R[d] - dissipation)
    end)
end

# System Fallback (If Upwind is called on a system, drop to Rusanov)
@inline function (upwind::UpwindFlux)(
    f_L::Flux{D, M, T}, f_R::Flux{D, M, T}, F_L::Flux{D, M, T}, F_R::Flux{D, M, T}, eq::HyperbolicPDE{D, M, T, R}
) where {D, M, T, R}
    
    return RusanovFlux()(f_L, f_R, F_L, F_R, eq)
end

@inline function (upwind::UpwindFlux)(
    f_L::Flux{D, 1, T}, f_R::Flux{D, 1, T}, F_L::Flux{D, 1, T}, F_R::Flux{D, 1, T}, eq::HyperbolicPDE{D, 1, T, R}
) where {D, T, R}
    
    return Flux{D, 1, T}(ntuple(Val(D)) do d
        du = f_R[d][1] - f_L[d][1]
        
        if abs(du) < T(1e-14)
            s = abs(velocity(eq, f_L[d])[d])
        else
            s = abs((F_R[d][1] - F_L[d][1]) / du)
        end
        
        T(0.5) * (F_L[d] + F_R[d] - s * (f_R[d] - f_L[d]))
    end)
end