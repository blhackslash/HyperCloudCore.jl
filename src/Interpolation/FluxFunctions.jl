export UpwindFlux, RusanovFlux

"""
    max_eigenvalues(eq::HyperbolicPDE, f_L::Flux, f_R::Flux)

Computes the maximum eigenvalue across all spatial dimensions for a given set of left and right fluxes.

# Returns
- An `SVector{D, T}` containing the maximum observed eigenvalue between the left and right states for each respective dimension.
"""
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
"""
    UpwindFlux
    RusanovFlux

Struct definitions for generalized numerical flux functions utilized by the solver.
"""
struct UpwindFlux <: NumericalFluxFunction end
struct RusanovFlux <: NumericalFluxFunction end

"""
    (rusanov::RusanovFlux)(f_L, f_R, F_L, F_R, eq)

Functor evaluating the Rusanov (local Lax-Friedrichs) numerical flux.

# Details
- Calculates numerical dissipation using the maximum eigenvalue bounded by the left and right states. 
- Returns the flux evaluation as `0.5 * (F_L + F_R - dissipation)` for each spatial dimension.
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
    (upwind::UpwindFlux)(f_L, f_R, F_L, F_R, eq)

Functor evaluating the Upwind numerical flux. 

# Details
- **Scalar Execution:** For scalar equations, it calculates the wave speed `s`. If the difference between left and right states is computationally zero (`< 1e-14`), it extracts the speed directly from the 1x1 Jacobian `SMatrix`. Otherwise, it computes the ratio of flux differences to state differences.
- **System Fallback:** If `UpwindFlux` is executed on a system of equations, it automatically falls back to dispatching the `RusanovFlux` algorithm.
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
            # NEW: Extract the scalar speed from the 1x1 Jacobian SMatrix
            s = abs(velocity(eq, f_L[d], d)[1, 1])
        else
            s = abs((F_R[d][1] - F_L[d][1]) / du)
        end
        
        T(0.5) * (F_L[d] + F_R[d] - s * (f_R[d] - f_L[d]))
    end)
end