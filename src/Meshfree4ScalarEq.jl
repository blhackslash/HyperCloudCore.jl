module Meshfree4ScalarEq

export runSimulation, GAS_GAMMA_EULER, DEBUG_TARGET_PARTICLE, DEBUG_TARGET_STEP

using Distributed
using LinearAlgebra
using Logging
using Printf
using ProgressMeter
using Random
using StaticArrays
using Statistics
using Base.Threads
using Polyester: @batch


include("Core/CoreTypes.jl")
include("Grid/ParticleGrids.jl")
include("Interpolation/Interpolations.jl")
include("TimeStepping/TimeIntegration.jl")

include("InitialConditions.jl")

# ==============================================================================
# --- PUBLIC API EXPORTS ---
# ==============================================================================

# 1. Geometries & State (from CoreTypes.jl)
export Space, State, Flux, Kinetic
export DifferentialOrder, Order0, Order1, DO0, DO1

# 2. Physics Equations (from CoreTypes.jl)
export HyperbolicPDE, ScalarHyperbolicPDE, HyperbolicPDESystem, NCHyperbolicPDESystem, DiagonalHyperbolicSystem
export LinearAdvection, BurgersEquation, TestU3Equation, Euler1D, BurgersEquation2D

# 3. Initial Conditions (from CoreTypes.jl)
export InitialCondition, SmoothInitialCondition
export Gauss, Box, Sine, Riemann, SRiemann, QuadrantRiemann

# 4. Numerics & Integration (from TimeIntegration.jl)
export TimeStepper, MeshfreeTimeStepper
export initTS!, initTSBuffer!, saveData!, time_integration_loop!

# 5. Core Simulation Structs (Assuming these exist in your other files)
export ParticleGrid, SimSetting, setInitialConditions!

# 6. Specific Time Steppers (Export the ones you actually use)
export SimpleSplitting, ARS233, PRSSP3, ARS222, ARS232
# export EulerUpwind, MainGrad, FallbackGrad, LinearizedRelaxationImplicitSolver ...

end # module
