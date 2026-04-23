# =========================================================================
# UNIVERSAL UTILITIES
# =========================================================================
@inline function sort_flux(f_i::State{M}, f_j::State{M}, F_i::Flux{D, M}, F_j::Flux{D, M}, dist_k::Space{D}) where {D, M}
    # Builds the arrays natively column-by-column
    f_L = Flux{D, M}(ntuple(d -> dist_k[d] > 0 ? f_i : f_j, Val(D)))
    f_R = Flux{D, M}(ntuple(d -> dist_k[d] > 0 ? f_j : f_i, Val(D)))
    
    F_L = Flux{D, M}(ntuple(d -> dist_k[d] > 0 ? F_i[d] : F_j[d], Val(D)))
    F_R = Flux{D, M}(ntuple(d -> dist_k[d] > 0 ? F_j[d] : F_i[d], Val(D)))
    
    return f_L, f_R, F_L, F_R
end

# =========================================================================
# STATE CONVERSIONS
# =========================================================================

# --- Burgers Equation MD ---
@inline prim2cons(::BurgersEquation, u::State{1}) = u
@inline cons2prim(::BurgersEquation, w::State{1}) = w

# --- Linear Advection ---
@inline prim2cons(::LinearAdvection, U::State{M}) where {M} = U
@inline cons2prim(::LinearAdvection, W::State{M}) where {M} = W

# --- Euler Equation ---
@inline function prim2cons(eq::EulerEquation{D, M}, V::State{M}) where {D, M}
    rho = V[1]
    u = SVector{D, Float64}(ntuple(d -> V[1+d], Val(D))) 
    p = V[M]
    
    m = rho .* u
    E = p / (GAS_GAMMA_EULER - 1.0) + 0.5 * rho * sum(abs2, u)
    
    return State{M}(rho, m..., E)
end

@inline function cons2prim(eq::EulerEquation{D, M}, U::State{M}) where {D, M}
    rho = max(U[1], 1e-7)
    m = SVector{D, Float64}(ntuple(d -> U[1+d], Val(D)))
    E = U[M]
    
    u = m ./ rho
    p = (GAS_GAMMA_EULER - 1.0) * (E - 0.5 * sum(abs2, m) / rho)
    
    return State{M}(rho, u..., max(p, 1e-7))
end

@inline function flux(eq::HyperbolicPDE{D,M,NCR}, u) where {D, M, NCR <: NCRepresentation}
    return zero(Flux{D, M})
end

# =========================================================================
# CONSERVATIVE FLUXES
# =========================================================================

# --- Burgers Equation MD ---
@inline function flux(eq::BurgersEquation{D, <:Conservative}, u::State{1}) where {D}
    return Flux{D, 1}(ntuple(_ -> 0.5 * u.^2, Val(D)))
end
#@inline flux(::BurgersEquation{D, <:NCRepresentation}, u::State{1}) where {D} = zero(Flux{D, 1})

# --- Linear Advection ---
@inline function flux(eq::LinearAdvection{D, M, <:Conservative}, U::State{M}) where {M, D}
    return Flux{D, M}(ntuple(d -> eq.vel[d] .* U, Val(D)))
end
#@inline flux(::LinearAdvection{D, M, <:NCRepresentation}, U::State{M}) where {D, M} = zero(Flux{D, M})

@inline function flux(::TestU3Equation{a}, u::SVector{1, Float64}) where {a}
    return SVector{1, Float64}(0.33333 * (1.0 - a) * u[1]^3)
end

# --- Euler Equation ---
@inline function flux(eq::EulerEquation{D, M, <:Conservative}, U::State{M}) where {D, M}
    V = cons2prim(eq, U)
    rho = V[1]
    u = SVector{D, Float64}(ntuple(d -> V[1+d], Val(D)))
    p = V[M]
    E = U[M]
    
    return Flux{D, M}(ntuple(Val(D)) do d
        ud = u[d]
        mass_flux = rho * ud
        mom_flux = ntuple(i -> rho * u[i] * ud + (i == d ? p : 0.0), Val(D))
        energy_flux = ud * (E + p)
        
        State{M}(mass_flux, mom_flux..., energy_flux)
    end)
end

# =========================================================================
# VELOCITIES & EIGENVALUES
# =========================================================================

@inline velocity(eq::LinearAdvection{D, M}, U::State{M}) where {M, D} = eq.vel
@inline velocity(eq::LinearAdvection{D, 1}, U::State{1}) where {D} = Space{D}(ntuple(i->eq.vel[i][1], Val(D)))

@inline velocity(eq::BurgersEquation{D}, u::State{1}) where {D} = Space{D}(ntuple(_ -> u[1], Val(D)))

@inline velocity(::TestU3Equation{a}, u::SVector{1, Float64}) where {a} = SVector{1, Float64}((1.0 - a) * u[1]^2)

# =========================================================================
# NON-CONSERVATIVE MATVECS
# =========================================================================

# --- Burgers Equation ---
# A(u) = u
@inline function A_matrix_times_vector(::BurgersEquation{D, <:NCRepresentation}, u::State{1}, du::State{1}) where {D}
    return State{1}(u[1] * du[1])
end

# --- Linear Advection (1D) ---
# A(U) = v
@inline function A_matrix_times_vector(eq::LinearAdvection{1, M, <:NCRepresentation}, U::State{M}, dU::State{M}) where {M}
    # eq.vel[1] safely extracts the State{M} vector from the 1D Flux tensor
    return eq.vel[1] .* dU
end