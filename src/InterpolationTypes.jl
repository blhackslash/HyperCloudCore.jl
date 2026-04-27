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

struct MUSCL{D, M, B_LEN, ORDER, I <: Interpolator, L <: AbstractSlopeLimiter, NF, MOOD} <: GradientInterpolator
    interpolator::I
    limiter::L
    numericalFlux::NF
    mood::MOOD
    gradients::Vector{SVector{B_LEN, State{M}}} 
end

function MUSCL(
    dimension::Int, M::Int, order::Int; 
    limiter=NoLimiter(), numericalFlux=RusanovFlux(), mood=NoMOOD()
)
    # Statically determine the basis length
    B_LEN_VAL = basis_length(Val(dimension), Val(order))
    B_LEN = typeof(B_LEN_VAL).parameters[1] 
    
    interp = Interpolator{dimension, order, 1}()
    # Pass `order` directly into the type signature!
    return MUSCL{dimension, M, B_LEN, order, typeof(interp), typeof(limiter), typeof(numericalFlux), typeof(mood)}(
        interp, limiter, numericalFlux, mood, SVector{B_LEN,State{M}}[]
    )
end

## ------------------------------- Upwind -------------------------------
abstract type UpwindAlgorithm end  
abstract type TiwariAlgorithm <: UpwindAlgorithm end 
abstract type PraveenAlgorithm <: UpwindAlgorithm end  
abstract type NonLinearPraveenAlgorithm <: UpwindAlgorithm end  
abstract type ClassicAlgorithm <: UpwindAlgorithm end 

struct UpwindGradient{D, I <: Interpolator, Algorithm <: UpwindAlgorithm} <: GradientInterpolator
    order::Int
    numericalFlux::NumericalFluxFunction
    interpolator::I
end

## ------------------------------- WENO -------------------------------

## ------------------------------- WENO -------------------------------

struct WENO{D, I <: Interpolator} <: GradientInterpolator
    order::Int
    interpolator::I
end

## ------------------------------- Central Gradient -------------------------------


struct CentralGradient{D, I <: Interpolator} <: GradientInterpolator
    order::Int
    interpolator::I
end