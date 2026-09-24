module HyperCloud

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
"""
    Space{D, T}
    State{M, T}
    Flux{D, M, T}

The core domain-specific `SVector` aliases used pervasively throughout the `HyperCloud` module.
- `Space`: Represents `D`-dimensional spatial coordinates.
- `State`: Represents `M`-dimensional equation variables for a single particle.
- `Flux`: Represents the evaluated fluxes across all `D` dimensions for all `M` equations.
"""
const Space{D, T} = SVector{D, T}
const State{M, T} = SVector{M, T}
const Flux{D, M, T} = SVector{D, State{M, T}}

include("CoreUtils.jl")
include("CoreTypes.jl")
include("API.jl")
include("PathIntegrals.jl")
include("LinearAdvection.jl")


include("./Grid/ParticleGrids.jl")
include("./Interpolation/Interpolators.jl")
include("./TimeStepping/TimeIntegration.jl")



# ==============================================================================
# --- PUBLIC API EXPORTS ---
# ==============================================================================

# 1. Geometries & State
export Space, State, Flux

end # module