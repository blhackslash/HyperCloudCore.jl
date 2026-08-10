struct Interpolator{D, IO, DO}
    function Interpolator{D, IO, DO}() where {D, IO, DO}
        new{D, IO, DO}()
    end
end

abstract type NumericalFluxFunction end
struct UpwindFlux <: NumericalFluxFunction end
struct RusanovFlux <: NumericalFluxFunction end
struct RoeDiffusiveFlux <: NumericalFluxFunction end

# =========================================================================
# MOOD STRATEGIES & CRITERIA
# =========================================================================
abstract type MOODStrategy end
struct EPD1 <: MOODStrategy end
struct EPD2 <: MOODStrategy end
struct EPD0 <: MOODStrategy end
struct StrictEPD0 <: MOODStrategy end

abstract type MOODCriterion end
abstract type RealMOOD <: MOODCriterion end

struct MOODu1{T} <: RealMOOD 
    d::T
end
struct MOODu2{T} <: RealMOOD 
    d::T
end

struct NoMOOD <: MOODCriterion end
struct OnlyMOOD <: RealMOOD end

struct MOOD{S <: MOODStrategy, C <: MOODCriterion}
    strategy::S
    criterion::C
end
MOOD(criterion::MOODCriterion) = MOOD(EPD1(), criterion)
MOOD() = MOOD(EPD1(), NoMOOD())

"""
    DivergenceInterpolator
"""
abstract type DivergenceInterpolator end

# =========================================================================
# SLOPE LIMITERS
# =========================================================================
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

struct ConstantReconstruction end

struct MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, MOOD, INTERPS, L, NF} <: DivergenceInterpolator
    interpolators::INTERPS
    limiter::L
    flux::NF
    mood::MOOD
    gradients::Vector{SVector{B_LEN, State{M, T}}} 
    particle_orders::Vector{Int} 
    mood_triggered::Vector{Bool} 
end

## ------------------------------- Upwind -------------------------------
abstract type UpwindAlgorithm end  
abstract type TiwariAlgorithm <: UpwindAlgorithm end 
abstract type PraveenAlgorithm <: UpwindAlgorithm end  
abstract type NonLinearPraveenAlgorithm <: UpwindAlgorithm end  
abstract type ClassicAlgorithm <: UpwindAlgorithm end 
## ------------------------------- Upwind -------------------------------
struct UpwindDivergence{D, M, T, I <: Interpolator, Algorithm <: UpwindAlgorithm} <: DivergenceInterpolator
    order::Int
    flux::NumericalFluxFunction
    interpolator::I
end

## ------------------------------- WENO -------------------------------
struct WENO{D, M, T, I <: Interpolator} <: DivergenceInterpolator
    order::Int
    interpolator::I
end

## ------------------------------- Central Divergence -------------------------------
struct CentralDivergence{D, M, T, I <: Interpolator} <: DivergenceInterpolator
    order::Int
    interpolator::I
end