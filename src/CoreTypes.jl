module CoreTypes

using StaticArrays
using ..HyperbolicPDEs

include("GridTypes.jl")

## Meshfree Timestepper Structs

abstract type TimeStepper end
abstract type MeshfreeTimeStepper <: TimeStepper end
abstract type FixedGridTimeStepper <: TimeStepper end
abstract type MeshfreeSystemTimeStepper <: MeshfreeTimeStepper end

struct EulerUpwind{G1 <: GradientInterpolator, G2 <: GradientInterpolator, M <: MOODCriterion, GM <: GridMover} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::M
    moveGrid::GM
    
    # Buffers are now part of the struct to be reused
    rhoInit::Vector{Float64}      # Stores the state at the beginning of the step
    neighbor_fs::Vector{Float64}  # Pre-gathered neighbor values
    neighbor_dfs::Vector{Float64} # Pre-gathered neighbor differences

    function EulerUpwind(gradientInterpolator::G1, gm::GM; fallbackInterpolator::G2 = NoFallbackGrad(), mood::M = NoMOOD()) where {G1 <: GradientInterpolator, G2 <: GradientInterpolator, M <: MOODCriterion, GM <: GridMover}
        # Initialize with empty buffers
        new{G1, G2, M, GM}(gradientInterpolator, fallbackInterpolator, mood, gm, Float64[], Float64[], Float64[])
    end
end

# No longer needs Nx, Ny. Buffers are sized based on the grid passed during the call.
struct RalstonRK2{G1, G2, MOOD, GM} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    moveGrid::GM
    
    # Buffers are now part of the struct to be reused
    rhoInit::Vector{Float64}
    rhos::Vector{Float64}
    div1::Vector{Float64}

    # Buffers for efficient calculations
    neighbor_fs::Vector{Float64}
    neighbor_dfs::Vector{Float64}

    function RalstonRK2(grad::G1, fallback::G2, mood::M, gm::GM) where {G1 <: GradientInterpolator, G2 <: GradientInterpolator, M <: MOODCriterion, GM <: GridMover}
        # Initialize with empty buffers, they will be resized on the first step
        new{G1, G2, M, GM}(grad, fallback, mood, gm, Float64[], Float64[], Float64[], Float64[], Float64[])
    end
end

end