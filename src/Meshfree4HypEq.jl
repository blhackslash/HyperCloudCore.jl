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

include("CoreUtils.jl")
include("CoreTypes.jl")
include("API.jl")
include("PathIntegrals.jl")
include("LinearAdvection.jl")


include("../Grid/ParticleGrids.jl")
include("../Interpolation/Interpolators.jl")
include("../TimeStepping/TimeIntegration.jl")



# ==============================================================================
# --- PUBLIC API EXPORTS ---
# ==============================================================================

# 1. Geometries & State
export Space, State, Flux

end # module