@inline function sort_flux(f_i::State{M}, f_j::State{M}, F_i::Flux{M, D}, F_j::Flux{M, D}, dist_k::Space{D}) where {D, M}
    # Build Tuple of Columns based on direction
    f_L_cols = ntuple(d -> dist_k[d] > 0 ? f_i : f_j, Val(D))
    f_R_cols = ntuple(d -> dist_k[d] > 0 ? f_j : f_i, Val(D))
    
    F_L_cols = ntuple(d -> dist_k[d] > 0 ? F_i[:, d] : F_j[:, d], Val(D))
    F_R_cols = ntuple(d -> dist_k[d] > 0 ? F_j[:, d] : F_i[:, d], Val(D))
    
    # hcat fuses the D SVectors into an MxD SMatrix natively!
    f_L_mat = hcat(f_L_cols...)
    f_R_mat = hcat(f_R_cols...)
    F_L_mat = hcat(F_L_cols...)
    F_R_mat = hcat(F_R_cols...)
    
    return f_L_mat, f_R_mat, F_L_mat, F_R_mat
end

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
@inline function max_eigenvalues(eq::HyperbolicPDE{D, M}, f_L::Flux{M, D}, f_R::Flux{M, D}) where {D, M}
    return SVector{D, Float64}(ntuple(Val(D)) do d
        # Extract the d-th column state
        lamL = max_eigenvalue(eq, State{M}(f_L[:, d]), d)
        lamR = max_eigenvalue(eq, State{M}(f_R[:, d]), d)
        max(lamL, lamR)
    end)
end
# =========================================================================
# NUMERICAL FLUXES (Fully Unified)
# =========================================================================
# ---------------------------------------------------------
# RUSANOV FLUX (Matrix Form)
# ---------------------------------------------------------
@inline function (rusanov::RusanovFlux)(f_L::Flux{M, D}, f_R::Flux{M, D}, F_L::Flux{M, D}, F_R::Flux{M, D}, eq::HyperbolicPDE{D, M}) where {D, M}
    
    s_vec = max_eigenvalues(eq, f_L, f_R) # Returns SVector{D, Float64}
    
    # Broadcast multiply the columns by their respective wave speeds
    dissipation = (f_R - f_L) .* s_vec'
    
    return 0.5 * (F_L + F_R - dissipation)
end

# ---------------------------------------------------------
# UPWIND FLUX (Matrix Form)
# ---------------------------------------------------------
@inline function (upwind::UpwindFlux)(f_L::Flux{1, D}, f_R::Flux{1, D}, F_L::Flux{1, D}, F_R::Flux{1, D}, eq::ScalarHyperbolicPDE{D}) where {D}
    
    delta_u = f_R - f_L # 1xD Matrix
    
    s_vec = SVector{D, Float64}(ntuple(Val(D)) do d
        du = delta_u[1, d]
        if abs(du) < 1e-14
            abs(velocity(eq, State{1}(f_L[:, d]))[d])
        else
            abs((F_R[1, d] - F_L[1, d]) / du)
        end
    end)
    
    dissipation = delta_u .* s_vec'
    return 0.5 * (F_L + F_R - dissipation)
end

# System Fallback
@inline function (upwind::UpwindFlux)(f_L::Flux{M, D}, f_R::Flux{M, D}, F_L::Flux{M, D}, F_R::Flux{M, D}, eq::HyperbolicPDESystem{D, M}) where {D, M}
    return RusanovFlux()(f_L, f_R, F_L, F_R, eq)
end