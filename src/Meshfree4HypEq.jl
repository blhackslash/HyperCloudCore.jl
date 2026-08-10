module Meshfree4HypEq

export run_simulation, DEBUG_TARGET_PARTICLE, DEBUG_TARGET_STEP

using LinearAlgebra
using Logging
using Printf
using ProgressMeter
using Random
using StaticArrays
using CPUSummary
using Base.Threads
using Polyester: @batch

## ------------------------------- SVector Types -------------------------------
const Space{D, T} = SVector{D, T}
const State{M, T} = SVector{M, T}
const Flux{D, M, T} = SVector{D, State{M, T}}

# Order important!
include("PDETypes.jl")
include("GridTypes.jl")
include("InterpolationTypes.jl")
include("TimestepperTypes.jl")

include("PDEUtils.jl")
include("CoreUtils.jl")
include("PathIntegrals.jl")


include("../Grid/ParticleGrids.jl")
include("../Interpolation/Interpolators.jl")
include("../TimeStepping/TimeIntegration.jl")



# ==============================================================================
# --- PUBLIC API EXPORTS ---
# ==============================================================================

# 1. Geometries & State
export Space, State, Flux
export DifferentialOrder, Order0, Order1, DO0, DO1

# Conversions
export param2uvec, param2xvec, param2svec, param2fvec, prim2cons, cons2prim

# 2. Physics Equations
export HyperbolicPDE, EquationRepresentation, Primitive, Conservative, Lagrangian
export LinePath, NaiveAveragePath, MappedPath
export LinearAdvection, BurgersEquation, TestU3Equation, EulerEquation, flux, flux_dot

# 3. Numerics & Integration
export TimeStepper, MeshfreeTimeStepper, FixedGridTimeStepper, MeshfreeSystemTimeStepper
export saveData!, mainTimeIntegrator!

# 4. Core Simulation Structs 
export ParticleGrid, CustomGridMover, NoGridMover, PhysicalGridMover, createParticleGrid, getTimeStep
export ExponentialWeightFunction, InverseWeightFunction

# 5. Interpolation
export Interpolator, DivergenceInterpolator, MUSCL, UpwindDivergence, WENO, CentralDivergence
export NumericalFluxFunction, UpwindFlux, RusanovFlux, RoeDiffusiveFlux
export MOODCriterion, MOODu1, MOODu2, NoMOOD, OnlyMOOD, MOOD, EPD1, EPD2, EPD0, StrictEPD0
export AbstractSlopeLimiter, BarthJespersenLimiter, VenkatakrishnanLimiter, SuperbeeLimiter, MinmodLimiter, NoLimiter

# 6. Time Steppers & Tableaus
export RKButcherTableau, GeneralRKTimeStepper, IMEXButcherTableau
export GeneralIMEXTimeStepper, Kin2Macro

export RK1_Euler_Tableau, RK2_Ralston_Tableau, RK3_SSP_Tableau, RK4_Classical_Tableau
export IMEX_Euler_Tableau, IMEX_ARS233_Tableau, IMEX_ARS222_Tableau, IMEX_PRSSP3_Tableau, IMEX_SSP2332_Tableau

# 7. Source Terms & Solvers
export NoSourceTerm, AbstractSourceTerm, RelaxationSourceTerm, NonLocalRelaxationSourceTerm
export LinearizedRelaxationImplicitSolver, PicardIterationSolver, AbstractImplicitSolver

end # module