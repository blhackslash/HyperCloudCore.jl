# =========================================================================
# UNIVERSAL UTILITIES
# =========================================================================
@inline function sort_flux(f_i::State{M, T}, f_j::State{M, T}, F_i::Flux{D, M, T}, F_j::Flux{D, M, T}, dist_k::Space{D, T}) where {D, M, T}
    f_L = Flux{D, M, T}(ntuple(d -> dist_k[d] > zero(T) ? f_i : f_j, Val(D)))
    f_R = Flux{D, M, T}(ntuple(d -> dist_k[d] > zero(T) ? f_j : f_i, Val(D)))
    
    F_L = Flux{D, M, T}(ntuple(d -> dist_k[d] > zero(T) ? F_i[d] : F_j[d], Val(D)))
    F_R = Flux{D, M, T}(ntuple(d -> dist_k[d] > zero(T) ? F_j[d] : F_i[d], Val(D)))
    
    return f_L, f_R, F_L, F_R
end

# =========================================================================
# STATE CONVERSIONS
# =========================================================================

# --- Burgers & Linear Advection ---
@inline prim2cons(::BurgersEquation, u::State{1, T}) where {T} = u
@inline cons2prim(::BurgersEquation, w::State{1, T}) where {T} = w

@inline prim2cons(::LinearAdvection, U::State{M, T}) where {M, T} = U
@inline cons2prim(::LinearAdvection, W::State{M, T}) where {M, T} = W

# --- Euler Equation ---
@inline function prim2cons(eq::EulerEquation{D, M, T}, V::State{M, T}) where {D, M, T}
    rho = V[1]
    u = Space{D, T}(ntuple(d -> V[1+d], Val(D))) 
    p = V[M]
    
    m = rho .* u
    E = p / (eq.gamma - T(1)) + T(0.5) * rho * sum(abs2, u) 
    
    return State{M, T}(rho, m..., E)
end

@inline function cons2prim(eq::EulerEquation{D, M, T}, U::State{M, T}) where {D, M, T}
    rho = max(U[1], T(1e-7))
    m = Space{D, T}(ntuple(d -> U[1+d], Val(D)))
    E = U[M]
    
    u = m ./ rho
    p = (eq.gamma - T(1)) * (E - T(0.5) * sum(abs2, m) / rho)
    
    return State{M, T}(rho, u..., max(p, T(1e-7)))
end

@inline function flux(eq::HyperbolicPDE{D, M, T, NCR}, u::State{M, T}) where {D, M, T, NCR <: NCRepresentation}
    return zero(Flux{D, M, T})
end

# =========================================================================
# CONSERVATIVE FLUXES
# =========================================================================

# --- Burgers Equation MD ---
@inline function flux(eq::BurgersEquation{D, T, <:Conservative}, u::State{1, T}) where {D, T}
    return Flux{D, 1, T}(ntuple(_ -> State{1, T}(T(0.5) * u[1]^2), Val(D)))
end

# --- Linear Advection ---
@inline function flux(eq::LinearAdvection{D, M, T, <:Conservative}, U::State{M, T}) where {M, D, T}
    return Flux{D, M, T}(ntuple(d -> eq.vel[d] .* U, Val(D)))
end

# --- Test U3 Equation ---
@inline function flux(::TestU3Equation{a, T, <:Conservative}, u::State{1, T}) where {a, T}
    val = T(1/3) * (one(T) - T(a)) * u[1]^3
    return Flux{1, 1, T}( (State{1, T}(val),) )
end

# --- Euler Equation ---
@inline function flux(eq::EulerEquation{D, M, T, <:Conservative}, U::State{M, T}) where {D, M, T}
    V = cons2prim(eq, U)
    rho = V[1]
    u = Space{D, T}(ntuple(d -> V[1+d], Val(D)))
    p = V[M]
    E = U[M]
    
    return Flux{D, M, T}(ntuple(Val(D)) do d
        ud = u[d]
        mass_flux = rho * ud
        mom_flux = ntuple(i -> rho * u[i] * ud + (i == d ? p : zero(T)), Val(D))
        energy_flux = ud * (E + p)
        
        State{M, T}(mass_flux, mom_flux..., energy_flux)
    end)
end

# =========================================================================
# VELOCITIES & EIGENVALUES
# =========================================================================

@inline velocity(eq::LinearAdvection{D, M, T}, U::State{M, T}) where {M, D, T} = eq.vel
@inline velocity(eq::LinearAdvection{D, 1, T}, U::State{1, T}) where {D, T} = Space{D, T}(ntuple(i->eq.vel[i][1], Val(D)))

@inline velocity(eq::BurgersEquation{D, T}, u::State{1, T}) where {D, T} = Space{D, T}(ntuple(_ -> u[1], Val(D)))

@inline velocity(::TestU3Equation{a, T}, u::State{1, T}) where {a, T} = State{1, T}((T(1) - T(a)) * u[1]^2)

# =========================================================================
# NON-CONSERVATIVE MATVECS
# =========================================================================

# --- Burgers Equation ---
@inline function A_matrix_times_vector(::BurgersEquation{D, T, <:NCRepresentation}, u::State{1, T}, du::State{1, T}) where {D, T}
    return State{1, T}(u[1] * du[1])
end

# --- Linear Advection (1D) ---
@inline function A_matrix_times_vector(eq::LinearAdvection{1, M, T, <:NCRepresentation}, U::State{M, T}, dU::State{M, T}) where {M, T}
    return eq.vel[1] .* dU
end