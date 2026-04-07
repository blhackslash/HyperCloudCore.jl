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
abstract type AbstractSlopeLimiter end
abstract type RealSlopeLimiter <: AbstractSlopeLimiter end
struct BarthJespersenLimiter <: RealSlopeLimiter end
struct VenkatakrishnanLimiter <: RealSlopeLimiter end
struct SuperbeeLimiter <: RealSlopeLimiter end
struct MinmodLimiter <: RealSlopeLimiter end
struct NoLimiter <: AbstractSlopeLimiter end

struct MUSCLWorkspace{D, M, B_LEN}
    distVec::Vector{Space{D}}
    wVec::Vector{Float64}
    dfVec::Vector{State{M}}
    dfFluxVec::Vector{Flux{D, M}}
    
    # The ultimate unified storage: 
    # Holds ALL derivatives (slopes, curves, etc.) in a single, strictly-typed SVector per particle!
    gradients::Vector{SVector{B_LEN, State{M}}} 
end


struct MUSCL{D, M, B_LEN, ORDER, I <: Interpolator, L <: AbstractSlopeLimiter, NF, MOOD} <: GradientInterpolator
    interpolator::I
    limiter::L
    numericalFlux::NF
    mood::MOOD
    workspaces::Vector{MUSCLWorkspace{D, M, B_LEN}}
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

struct UpwindWorkspacePA{D,M} <: UpwindWorkspace end

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