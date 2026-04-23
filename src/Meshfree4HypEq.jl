module Meshfree4HypEq

export runSimulation, GAS_GAMMA_EULER, DEBUG_TARGET_PARTICLE, DEBUG_TARGET_STEP

using LinearAlgebra
using Logging
using Printf
using ProgressMeter
using Random
using StaticArrays
using Base.Threads
using Polyester: @batch

## ------------------------------- SVector Types -------------------------------
const Space{D} = SVector{D, Float64}
const State{M} = SVector{M, Float64}
const Flux{D, M} = SVector{D, State{M}}
const Kinetic{K} = SVector{K, Float64}



# Order important!
include("PDETypes.jl")
include("GridTypes.jl")
include("InterpolationTypes.jl")
include("TimestepperTypes.jl")

include("PDEUtils.jl")
include("CoreUtils.jl")
include("PathIntegrals.jl")


include("../Grid/ParticleGrids.jl")
include("../Interpolation/Interpolations.jl")
include("../TimeStepping/TimeIntegration.jl")



# ==============================================================================
# --- PUBLIC API EXPORTS ---
# ==============================================================================

# 1. Geometries & State (from CoreTypes.jl)
export Space, State, Flux, Kinetic
export DifferentialOrder, Order0, Order1, DO0, DO1

# Conversions
export param2uvec, param2xvec, param2svec, param2fvec, prim2cons, cons2prim

# 2. Physics Equations (from CoreTypes.jl)
export HyperbolicPDE, EquationRepresentation, Primitive, Conservative, Lagrangian
export LinePath, NaiveAveragePath, MappedPath
export LinearAdvection, BurgersEquation, TestU3Equation, EulerEquation, flux, flux_dot

# 4. Numerics & Integration (from TimeIntegration.jl)
export TimeStepper, MeshfreeTimeStepper
export initTS!, initTSBuffer!, saveData!, mainTimeIntegrator!

# 5. Core Simulation Structs (Assuming these exist in your other files)
export ParticleGrid, SimSetting, CustomGridMover, NoGridMover, PhysicalGridMover, createParticleGrid, getTimeStep
export exponentialWeightFunction, inverseWeightFunction
# Interpolation
export Interpolator, GradientInterpolator, MUSCL, UpwindGradient, WENO, CentralGradient, NoFallbackGrad
export NumericalFluxFunction, UpwindFlux, RusanovFlux, RoeDiffusiveFlux
export MOODCriterion, MOODu1, MOODu2, NoMOOD, OnlyMOOD
export AbstractSlopeLimiter, BarthJespersenLimiter, VenkatakrishnanLimiter, SuperbeeLimiter, MinmodLimiter, NoLimiter

# 6. Specific Time Steppers (Export the ones you actually use)
export SimpleSplitting, ARS233, PRSSP3, ARS222, ARS232, RKButcherTableau, GeneralRKTimeStepper, IMEXButcherTableau
export EulerUpwind, LinearizedRelaxationImplicitSolver, GeneralIMEXTimeStepper, Kin2Macro
export EulerTableau, RalstonRK2Tableau, SSPRK3Tableau, RK4Tableau
export PR_IMEX_SSP3_ButcherTableau, IMEXButcherTableau, IMEXARS233ButcherTableau, ARS222_ButcherTableau, SSP2332ButcherTableau

export NoSourceTerm, AbstractSourceTerm, RelaxationSourceTerm, NonLocalRelaxationSourceTerm
export LinearizedRelaxationImplicitSolver, PicardIterationSolver, AbstractImplicitSolver

end # module
