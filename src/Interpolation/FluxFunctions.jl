export UpwindFlux, RusanovFlux, AlignedRusanovFlux

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
@inline function max_eigenvalues(eq::HyperbolicPDE{D, M, T}, f_L::Flux{D, M, T}, f_R::Flux{D, M, T}) where {D, M, T}
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
    f_L::Flux{D, M, T}, f_R::Flux{D, M, T}, F_L::Flux{D, M, T}, F_R::Flux{D, M, T}, eq::HyperbolicPDE{D, M, T}
) where {D, M, T}
    
    s_vec = max_eigenvalues(eq, f_L, f_R)
    
    return Flux{D, M, T}(ntuple(Val(D)) do d
        dissipation = s_vec[d] * (f_R[d] - f_L[d])
        T(0.5) * (F_L[d] + F_R[d] - dissipation)
    end)
end

"""
    AlignedRusanovFlux <: NumericalFluxFunction

The Rusanov flux with its dissipation aligned with the divergence weights where an order was dropped. On
a pair whose particles both have the full order it is the plain [`RusanovFlux`](@ref) (states sorted into
"left" and "right" per axis by the sign of the distance). Where either particle is below the full order
the states are sorted by the sign of the particle's own divergence weight `c_ij` for that neighbour
instead; the rule is symmetric in the pair, so both sides evaluate the same flux. The dissipation then
enters the divergence of particle `i` with the coefficient `½ Σ_d s_d |c_ij,d| ≥ ½ |c_ij · f'|`, so the
first-order update (constant reconstructions, zero-sum weights) is local-extremum-diminishing for *any*
divergence weights, including high-degree fits on one-sided stencils, whose weights change sign over the
stencil and make the geometric dissipation anti-dissipative there. That allows a uniform divergence degree
(`div_order`) with MOOD acting on the reconstruction only, which removes the mixed-degree pairs and their
mass defect; the schemes that use it (`MUSCL`) keep the per-slot weight signs with their geometry caches.
Away from drops the scheme is identical to the plain flux, so accuracy and error constant are unchanged.
"""
struct AlignedRusanovFlux <: NumericalFluxFunction end

@inline (::AlignedRusanovFlux)(f_L::Flux{D, M, T}, f_R::Flux{D, M, T}, F_L::Flux{D, M, T}, F_R::Flux{D, M, T},
                                eq::HyperbolicPDE{D, M, T}) where {D, M, T} = RusanovFlux()(f_L, f_R, F_L, F_R, eq)

# Whether a flux sorts the interface states by the signs of the divergence weights (`AlignedRusanovFlux`) or
# by the geometry (every other flux).
@inline _aligned_dissipation(::NumericalFluxFunction) = false
@inline _aligned_dissipation(::AlignedRusanovFlux) = true

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
    f_L::Flux{D, M, T}, f_R::Flux{D, M, T}, F_L::Flux{D, M, T}, F_R::Flux{D, M, T}, eq::HyperbolicPDE{D, M, T}
) where {D, M, T}
    
    return RusanovFlux()(f_L, f_R, F_L, F_R, eq)
end

@inline function (upwind::UpwindFlux)(
    f_L::Flux{D, 1, T}, f_R::Flux{D, 1, T}, F_L::Flux{D, 1, T}, F_R::Flux{D, 1, T}, eq::HyperbolicPDE{D, 1, T}
) where {D, T}
    
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