export MLSWeightFunction, AbstractBoundaryCondition, NumericalFluxFunction, MOODStrategy, MOODCriterion
export RealMOOD, AbstractSlopeLimiter, RealSlopeLimiter, DivergenceInterpolator, AbstractPath, EquationRepresentation, NCRepresentation, Conservative
export HyperbolicPDE, UpwindAlgorithm, NoSourceTerm, AbstractSourceTerm, NoGridMover, GridMover
export RKButcherTableau, IMEXButcherTableau, GeneralIMEXTimeStepper, GeneralRKTimeStepper, TimeStepper

"""
    MLSWeightFunction
    GridMover
    AbstractBoundaryCondition
    NumericalFluxFunction
    MOODStrategy
    MOODCriterion
    AbstractSlopeLimiter
    DivergenceInterpolator
    UpwindAlgorithm
    HyperbolicPDE
    TimeStepper
    AbstractSourceTerm

Core abstract types defining the extensible architecture of the mesh-free solver.
"""
abstract type MLSWeightFunction end
abstract type GridMover end
struct NoGridMover <: GridMover end
abstract type AbstractBoundaryCondition end

abstract type NumericalFluxFunction end

export MOODStrategy, Halo, MOODCriterion, MOOD

abstract type MOODStrategy end
abstract type Halo{N} <: MOODStrategy end

abstract type MOODCriterion end

"""
    MOOD{S <: MOODStrategy, C <: MOODCriterion}

A concrete structure pairing a `MOODStrategy` (dictating the depth of order reduction via `Halo{N}`) 
with a `MOODCriterion` (dictating when order reduction is triggered).
"""
struct MOOD{S <: MOODStrategy, C <: MOODCriterion}
    strategy::S
    criterion::C
end

abstract type AbstractSlopeLimiter end
abstract type RealSlopeLimiter <: AbstractSlopeLimiter end

abstract type DivergenceInterpolator end

abstract type AbstractPath end
abstract type PathIntegrator end
struct PathIntegral{P <: AbstractPath, I <: PathIntegrator}
    path::P
    integrator::I
end
"""
    Conservative
    NCRepresentation{P <: AbstractPath}

Representations of the governing equations. 
- `Conservative` indicates a standard divergence form. 
- `NCRepresentation` designates systems containing non-conservative products evaluated along a specific path.
"""
abstract type EquationRepresentation end
struct Conservative <: EquationRepresentation end
abstract type NCRepresentation{P <: AbstractPath} <: EquationRepresentation end

abstract type HyperbolicPDE{D, M, T, R <: EquationRepresentation} end

export AbstractSourceTerm, AbstractExplicitSourceTerm, AbstractImplicitSourceTerm
export NoExplicitSource, NoImplicitSource

abstract type AbstractSourceTerm end
abstract type AbstractExplicitSourceTerm <: AbstractSourceTerm end
abstract type AbstractImplicitSourceTerm <: AbstractSourceTerm end

struct NoExplicitSource <: AbstractExplicitSourceTerm end
struct NoImplicitSource <: AbstractImplicitSourceTerm end

abstract type TimeStepper end