module HyperCloudCore

using LinearAlgebra
using Logging
using Printf
using Random
using StaticArrays
using Base.Threads
using Polyester: @batch

## ------------------------------- SVector Types -------------------------------
export Space, State, Flux, Velocity

"""
    Space{D, T}

Static vector alias representing `D`-dimensional spatial coordinates with element type `T`.
"""
const Space{D, T} = SVector{D, T}

"""
    State{M, T}

Static vector alias representing an `M`-dimensional vector of equation variables for a single particle with element type `T`.
"""
const State{M, T} = SVector{M, T}

"""
    Flux{D, M, T}

Static vector alias representing evaluated fluxes across all `D` spatial dimensions for an `M`-variable system (`SVector{D, State{M, T}}`).
"""
const Flux{D, M, T} = SVector{D, State{M, T}}

"""
    Velocity{D, M, T, L}

Static vector alias representing flux Jacobians across all `D` spatial dimensions for an `M`-variable system (`SVector{D, SMatrix{M, M, T, L}}`), where `L = M * M`.
"""
const Velocity{D, M, T, L} = SVector{D, SMatrix{M, M, T, L}}

include("Utils.jl")
include("Types.jl")
include("API.jl")


include("./Grid/ParticleGrids.jl")

include("./TimeStepping/_main.jl")

include("./Interpolation/Interpolators.jl")



end # module