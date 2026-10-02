"""
    InteractionBuffer{D, M, T}

A thread-safe, pre-allocated workspace designed to hold neighbor interaction data during flux evaluations.

# Fields
- `f::Vector{State{M, T}}`: Stores the direct state of neighboring particles.
- `df::Vector{State{M, T}}`: Stores the raw state differences between neighbors and the target particle.
- `df_flux::Vector{Flux{D, M, T}}`: Stores computed numerical flux differences or non-conservative jumps.
- `df_scratch::Vector{State{M, T}}`: An auxiliary buffer for intermediate moving least squares operations.
- `mask::Vector{Bool}`: A boolean array used to filter specific neighbors dynamically during directional stencil building.
"""
struct InteractionBuffer{D, M, T}
    f::Vector{State{M, T}}
    df::Vector{State{M, T}}
    df_flux::Vector{Flux{D, M, T}}
    df_scratch::Vector{State{M, T}} 
    mask::Vector{Bool} 
    
    InteractionBuffer{D, M, T}() where {D, M, T} = new{D, M, T}(
        State{M, T}[], State{M, T}[], Flux{D, M, T}[], State{M, T}[], Bool[]
    )
end

"""
    update_size!(ib::InteractionBuffer, num_interactions)
    update_size!(ts::GeneralIMEXTimeStepper, N_particles, M_neighbors)
    update_size!(ts::GeneralRKTimeStepper, N_particles, M_neighbors)

Dynamically resizes internal buffers and state arrays to accommodate the current number of particles and neighbor interactions.

# Details
- For `InteractionBuffer`, it ensures sufficient capacity for fields like `f`, `df`, `df_flux`, and `mask`.
- For time steppers, it resizes the target RK/IMEX stage arrays (e.g., `Y_stages`, `K_stages`) and automatically cascades the update to the internal neighbor `InteractionBuffer`.
"""
function update_size!(ib::InteractionBuffer, num_interactions::Int)
    ensure_capacity!(ib.f, num_interactions)
    ensure_capacity!(ib.df, num_interactions)
    ensure_capacity!(ib.df_flux, num_interactions)
    ensure_capacity!(ib.df_scratch, num_interactions)
    ensure_capacity!(ib.mask, num_interactions)
    return nothing
end

include("ButcherTableaus.jl")
include("MeshfreeRKTimeSteppers.jl")
include("MeshfreeIMEXTimeSteppers.jl")
include("MOOD.jl")