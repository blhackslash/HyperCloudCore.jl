struct Interpolator{D, IO, DO}

    function Interpolator{D, IO, DO}() where {D, IO, DO}
        new{D, IO, DO}()
    end
end

## ------------------------------- Flux Functions -------------------------------
abstract type NumericalFluxFunction end
struct UpwindFlux <: NumericalFluxFunction end
struct RusanovFlux <: NumericalFluxFunction end
struct RoeDiffusiveFlux <: NumericalFluxFunction end

"""
    MOODCriterion

Abstract MOOD Criterion type. Each MOOD criterion should overload the ()-operator, checks if the MOOD criterion at that cell is satisfied.
Returns true for a MOOD event.
"""
abstract type MOODCriterion end

# --- MOODu1 (Simple DMP Check) ---
struct MOODu1 <: MOODCriterion 
    d::Float64
end
# --- MOODu2 (DMP Check + Conditional Curvature Relaxation) ---
struct MOODu2 <: MOODCriterion 
    d::Float64
end
struct NoMOOD <: MOODCriterion end
struct OnlyMOOD <: MOODCriterion end

"""
    GradientInterpolator

In case of unstructured grids, the spatial gradient is approximated using a moving least squares (MLS) method based on Taylor polynomials.
These algorithms are implemented as follows. Each method is a struct that is a subtype of GradientInterpolator. The gradient 
at a gridpoint can then be computed using the ()-operator; see for example UpwindGradient and CentralGradient. These objects select
the correct stencil and then call the MLS routine (gradInterpolation).
"""
abstract type GradientInterpolator end

# Fallback Gradient interpolator for no fallback
struct NoFallbackGrad <: GradientInterpolator end

## ------------------------------- MUSCL -------------------------------
abstract type MUSCLORDER end
struct MUSCLORDER0 <: MUSCLORDER end
struct MUSCLORDER1 <: MUSCLORDER end
struct MUSCLORDER2 <: MUSCLORDER end
struct MUSCLORDER3 <: MUSCLORDER end
struct MUSCLORDER4 <: MUSCLORDER end

abstract type AbstractSlopeLimiter end
abstract type RealSlopeLimiter <: AbstractSlopeLimiter end
struct BarthJespersenLimiter <: RealSlopeLimiter end
struct VenkatakrishnanLimiter <: RealSlopeLimiter end
struct SuperbeeLimiter <: RealSlopeLimiter end
struct MinmodLimiter <: RealSlopeLimiter end
struct NoLimiter <: AbstractSlopeLimiter end

abstract type MUSCLWorkspace end

# --- NEW: 1D Workspaces split by order ---
abstract type MUSCLWorkspace1D <: MUSCLWorkspace end

# 1D Workspace for Order 0
struct MUSCLWorkspace1D0O <: MUSCLWorkspace1D
    # Only stores geometric coefficients for divergence
    alfaij_bars::Vector{Float64} 
end

"""
Workspace for 1D, 1st/2nd Order MUSCL.
Stores 1st/2nd order coefficients and derivatives.
(O1 and O2 are combined, as O1 slope limiting (MOOD) needs curvature).
"""
struct MUSCLWorkspace1D1O <: MUSCLWorkspace1D
    # --- FLATTENED per-interaction coefficient storage ---
    alfaij_bars::Vector{Float64} # for 1st-order slope
    betaijs::Vector{Float64}     # for 2nd-order curve

    # --- PER-PARTICLE derivative storage (already flat) ---
    slopes::Vector{Float64}
    curves_xx::Vector{Float64}
end

"""
Workspace for 1D, 1st/2nd Order MUSCL.
Stores 1st/2nd order coefficients and derivatives.
(O1 and O2 are combined, as O1 slope limiting (MOOD) needs curvature).
"""
struct MUSCLWorkspace1D2O <: MUSCLWorkspace1D
    # --- FLATTENED per-interaction coefficient storage ---
    alfaij_bars::Vector{Float64} # for 1st-order slope
    betaijs::Vector{Float64}     # for 2nd-order curve

    # --- PER-PARTICLE derivative storage (already flat) ---
    slopes::Vector{Float64}
    curves_xx::Vector{Float64}
end

"""
Workspace for 1D, 3rd Order MUSCL.
Includes thread-local buffers for stable QR decomposition.
"""
struct MUSCLWorkspace1D3O <: MUSCLWorkspace1D
    # --- Coeffs ---
    alfaijs::Vector{Float64}     # for d3 [cite: 9]
    alfaij_bars::Vector{Float64} # for d1 [cite: 9]
    betaijs::Vector{Float64}     # for d2 [cite: 9]
    
    # --- Derivatives ---
    slopes::Vector{Float64}
    curves_xx::Vector{Float64}
    d3fdx3::Vector{Float64}

end 

"""
Workspace for 1D, 4th Order MUSCL.
"""
struct MUSCLWorkspace1D4O <: MUSCLWorkspace1D
    # --- Coeffs ---
    alfaijs::Vector{Float64}     # for d3
    alfaij_bars::Vector{Float64} # for d1
    betaijs::Vector{Float64}     # for d2
    gammaijs::Vector{Float64}    # for d4
    
    # --- Derivatives ---
    slopes::Vector{Float64}
    curves_xx::Vector{Float64}
    d3fdx3::Vector{Float64}
    d4fdx4::Vector{Float64} # Field for 4th derivative


end

# --- NEW: 2D Workspaces split by order ---
abstract type MUSCLWorkspace2D <: MUSCLWorkspace end

# 2D Workspace for Order 0
struct MUSCLWorkspace2D0O <: MUSCLWorkspace2D
    # Only stores geometric coefficients for divergence
    alfaijs::Vector{Float64}
    betaijs::Vector{Float64}
end

"""
Workspace for 2D, 1st Order MUSCL.
Contains flat buffers for coefficients and per-particle slope storage.
"""
struct MUSCLWorkspace2D1O <: MUSCLWorkspace2D
    # --- FLATTENED per-interaction coefficient storage ---
    alfaijs::Vector{Float64}
    betaijs::Vector{Float64}

    # --- PER-PARTICLE slope storage (already flat) ---
    slopes_x::Vector{Float64}
    
    slopes_y::Vector{Float64}

end

"""
Workspace for 2D, 2nd Order MUSCL.
Contains extended flat buffers for coefficients, per-particle derivative storage,
and a temporary matrix buffer for the pseudo-inverse calculation.
"""
struct MUSCLWorkspace2D2O <: MUSCLWorkspace2D
    # --- FLATTENED per-interaction coefficient storage ---
    alfaijs::Vector{Float64}     # for fx
    betaijs::Vector{Float64}     # for fy
    alfaij_bars::Vector{Float64} # for fxx
    betaij_bars::Vector{Float64} # for fyy
    gammaijs::Vector{Float64}    # for fxy

    # --- PER-PARTICLE derivative storage (already flat) ---
    slopes_x::Vector{Float64}
    slopes_y::Vector{Float64}
    curves_xx::Vector{Float64} 
    curves_yy::Vector{Float64} 
    curves_xy::Vector{Float64} 

end

struct MUSCL{D,ORDER<:MUSCLORDER, L<:AbstractSlopeLimiter, NFF <: NumericalFluxFunction, WS<:MUSCLWorkspace, M<:MOODCriterion} <: GradientInterpolator
    order::ORDER
    limiter::L
    res::Vector{Float64}
    numericalFlux::NFF
    workspace::WS
    mood::M
end

## ------------------------------- Upwind -------------------------------
abstract type UpwindAlgorithm end  
abstract type TiwariAlgorithm <: UpwindAlgorithm end 
abstract type PraveenAlgorithm <: UpwindAlgorithm end  
abstract type NonLinearPraveenAlgorithm <: UpwindAlgorithm end  
abstract type ClassicAlgorithm <: UpwindAlgorithm end 

abstract type UpwindWorkspace end

struct UpwindWorkspaceTA{D, M} <: UpwindWorkspace
    distVec::Vector{Space{D}}
    dfVec::Vector{State{M}} 
    wVec::Vector{Float64}
    xWindow::BitVector
    yWindow::BitVector
end
# Inside Upwind.jl (Around line 15)
struct UpwindWorkspaceCA{D, M} <: UpwindWorkspace
    distVec::Vector{Space{D}}
    dfVec::Vector{State{M}}
    wVec::Vector{Float64}
    dfFluxVec::Vector{Flux{D, M}} # <-- NEW MATRIX BUFFER
end

struct UpwindWorkspacePA{D} <: UpwindWorkspace
    coeff_Vec::Vector{Space{D}}
    cijVec::Vector{Float64}     
end

struct UpwindGradient{D, WS <: UpwindWorkspace, I <: Interpolator, Algorithm <: UpwindAlgorithm} <: GradientInterpolator
    order::Int
    numericalFlux::NumericalFluxFunction
    workspaces::Vector{WS}
    interpolator::I
end

## ------------------------------- WENO -------------------------------

abstract type WENOWorkspace end
abstract type WENOGI <:GradientInterpolator end

struct WENOWorkspace1D <: WENOWorkspace
    # Scratch space for one-sided stencil calculations
    dx_stencil::Vector{Float64}
    df_stencil::Vector{Float64}
    w_stencil::Vector{Float64}

    function WENOWorkspace1D(max_neighbors::Int=30)
        new(
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors)
        )
    end
end

"""
A minimal, thread-local workspace for the 2D WENO algorithm.
Holds a single set of "scratch" buffers to build stencils in.
"""
struct WENOWorkspace2D <: WENOWorkspace
    # Scratch space for stencil calculations
    dx_stencil::Vector{Float64}
    dy_stencil::Vector{Float64}
    df_stencil::Vector{Float64}
    w_stencil::Vector{Float64}

    function WENOWorkspace2D(max_neighbors::Int=30)
        new(
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors)
        )
    end
end

"""
Refactored WENO struct to hold thread-local workspaces.
"""
struct WENO{D,WS <: WENOWorkspace, I <: Interpolator, NFF <: NumericalFluxFunction} <: WENOGI
    order::Int
    workspaces::Vector{WS}
    interpolator::I
    numericalFlux::NFF
end

## ------------------------------- Central Gradient -------------------------------


struct CentralGradient{D, I <: Interpolator} <: GradientInterpolator
    order::Int
    interpolator::I
end