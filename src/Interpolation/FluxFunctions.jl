export UpwindFlux, RusanovFlux

"""
    max_eigenvalues(eq::HyperbolicPDE, f_L::Flux, f_R::Flux) -> SVector{D, T}

Computes the maximum wave speed (eigenvalue) between left and right reconstructed states across each spatial dimension.

# Arguments
- `eq`: The hyperbolic PDE system being solved.
- `f_L`: Reconstructed state values at the left of the interface across each dimension.
- `f_R`: Reconstructed state values at the right of the interface across each dimension.

# Returns
- `SVector{D, T}`: Dimensional vector containing `max(|λ_L|, |λ_R|)` for each spatial axis.
"""
@inline function max_eigenvalues(eq::HyperbolicPDE{D, M, T, R}, f_L::Flux{D, M, T}, f_R::Flux{D, M, T}) where {D, M, T, R}
    return SVector{D, T}(ntuple(Val(D)) do d
        lamL = max_eigenvalue(eq, f_L[d], d)
        lamR = max_eigenvalue(eq, f_R[d], d)
        max(lamL, lamR)
    end)
end


# =========================================================================
# NUMERICAL FLUXES
# =========================================================================

"""
    UpwindFlux <: NumericalFluxFunction

An upwind numerical interface flux evaluator.

# Details
- **Scalar Systems (`M = 1`)**: Evaluates the Rankine-Hugoniot shock speed `s = |ΔF / Δu|`. If states are nearly coincident (`|Δu| < 1e-14`), it extracts the wave speed directly from the scalar flux Jacobian.
- **Vector Systems (`M > 1`)**: Automatically falls back to dispatching `RusanovFlux`
"""
struct UpwindFlux <: NumericalFluxFunction end

"""
    RusanovFlux <: NumericalFluxFunction

A Rusanov (local Lax-Friedrichs) numerical interface flux evaluator.

# Details
Evaluates the interface flux with localized numerical dissipation scaled by the maximum characteristic wave speed:

    F_num = 0.5 * (F_L + F_R - s_max * (u_R - u_L))

where `s_max = max(|λ_L|, |λ_R|)` is evaluated along each spatial dimension via `max_eigenvalues`
"""
struct RusanovFlux <: NumericalFluxFunction end

"""
    (rusanov::RusanovFlux)(f_L, f_R, F_L, F_R, eq) -> Flux{D, M, T}

Evaluates the Rusanov (local Lax-Friedrichs) interface flux across all spatial dimensions.

# Arguments
- `f_L`: Reconstructed left states across all spatial dimensions.
- `f_R`: Reconstructed right states across all spatial dimensions.
- `F_L`: Physical fluxes evaluated at `f_L`.
- `F_R`: Physical fluxes evaluated at `f_R`.
- `eq`: The hyperbolic PDE system.
"""
@inline function (rusanov::RusanovFlux)(
    f_L::Flux{D, M, T}, f_R::Flux{D, M, T}, F_L::Flux{D, M, T}, F_R::Flux{D, M, T}, eq::HyperbolicPDE{D, M, T, R}
) where {D, M, T, R}
    
    s_vec = max_eigenvalues(eq, f_L, f_R)
    
    return Flux{D, M, T}(ntuple(Val(D)) do d
        dissipation = s_vec[d] * (f_R[d] - f_L[d])
        T(0.5) * (F_L[d] + F_R[d] - dissipation)
    end)
end

"""
    (upwind::UpwindFlux)(f_L, f_R, F_L, F_R, eq) -> Flux{D, M, T}

Evaluates the numerical flux using the upwind scheme. Dispatches to `RusanovFlux` for general systems (`M > 1`), or exact scalar upwinding when `M == 1`.

# Arguments
- `f_L`: Reconstructed left states across all spatial dimensions.
- `f_R`: Reconstructed right states across all spatial dimensions.
- `F_L`: Physical fluxes evaluated at `f_L`.
- `F_R`: Physical fluxes evaluated at `f_R`.
- `eq`: The hyperbolic PDE system.
"""
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
            # Extract the scalar speed from the 1x1 Jacobian SMatrix
            s = abs(velocity(eq, f_L[d], d)[1, 1])
        else
            s = abs((F_R[d][1] - F_L[d][1]) / du)
        end
        
        T(0.5) * (F_L[d] + F_R[d] - s * (f_R[d] - f_L[d]))
    end)
end