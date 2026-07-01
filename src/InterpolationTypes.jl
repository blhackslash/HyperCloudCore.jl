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
# =========================================================================
# MOOD STRATEGIES & CRITERIA (Pure Dispatch Tags & Data Holders)
# =========================================================================
abstract type MOODStrategy end
struct EPD1 <: MOODStrategy end
struct EPD2 <: MOODStrategy end
struct EPD0 <: MOODStrategy end
struct StrictEPD0 <: MOODStrategy end

abstract type MOODCriterion end
abstract type RealMOOD <: MOODCriterion end

# --- Concrete Criteria ---
struct MOODu1 <: RealMOOD 
    d::Float64
end

struct MOODu2 <: RealMOOD 
    d::Float64
end

struct NoMOOD <: MOODCriterion end
struct OnlyMOOD <: RealMOOD end

# =========================================================================
# THE MASTER MOOD FUNCTOR STRUCT
# =========================================================================
struct MOOD{S <: MOODStrategy, C <: MOODCriterion}
    strategy::S
    criterion::C
end

# Convenience Constructors (Defaulting to EPD1 and NoMOOD)
MOOD(criterion::MOODCriterion) = MOOD(EPD1(), criterion)
MOOD() = MOOD(EPD1(), NoMOOD())
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

struct BarthJespersenLimiter{Mode} <: RealSlopeLimiter 
    BarthJespersenLimiter(mode::Symbol=:soft) = new{mode}()
end

struct VenkatakrishnanLimiter{Mode} <: RealSlopeLimiter 
    VenkatakrishnanLimiter(mode::Symbol=:soft) = new{mode}()
end

struct SuperbeeLimiter{Mode} <: RealSlopeLimiter 
    SuperbeeLimiter(mode::Symbol=:soft) = new{mode}()
end

struct MinmodLimiter{Mode} <: RealSlopeLimiter 
    MinmodLimiter(mode::Symbol=:soft) = new{mode}()
end

struct NoLimiter <: AbstractSlopeLimiter end

# Singleton for Order 1 (Degree 0)
struct ConstantReconstruction end

struct MUSCL{D, M, B_LEN, MAX_ORDER, DIV_ORDER, MOOD, INTERPS, L, NF} <: GradientInterpolator
    interpolators::INTERPS
    limiter::L
    numericalFlux::NF
    mood::MOOD
    gradients::Vector{SVector{B_LEN, State{M}}} 
    particle_orders::Vector{Int} 
    mood_triggered::Vector{Bool} 
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