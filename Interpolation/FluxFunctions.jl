export UpwindFlux, RusanovFlux

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
struct UpwindFlux <: NumericalFluxFunction end
struct RusanovFlux <: NumericalFluxFunction end

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
            # NEW: Extract the scalar speed from the 1x1 Jacobian SMatrix
            s = abs(velocity(eq, f_L[d], d)[1, 1])
        else
            s = abs((F_R[d][1] - F_L[d][1]) / du)
        end
        
        T(0.5) * (F_L[d] + F_R[d] - s * (f_R[d] - f_L[d]))
    end)
end