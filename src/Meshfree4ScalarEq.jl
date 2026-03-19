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

using IPlotPDESols

include("../SimulationFunctions/runSimulation.jl")

end  # module 
