mutable struct LocalVoxels
    num_bins::Int
    half_bins::Int
    voxel_size::Float64
    occupation::Vector{Bool}
    function LocalVoxels(min_nb::Int, R::Float64)
        num_bins = 2 * min_nb + 1
        voxel_size = (2.0 * R) / num_bins
        occupation = zeros(Bool, num_bins)
        new(num_bins, min_nb, voxel_size, occupation)
    end
end

## ------------------------------- Weight Functions -------------------------------
abstract type MLSWeightFunction end

"""
    exponentialWeightFunction(alpha::Real, range::Real)

Functor that calculates an exponential weight based on distance.
The parameters `alpha` (shape parameter) and `range` (normalization distance)
are stored directly in the struct.
"""
struct exponentialWeightFunction <: MLSWeightFunction
    alpha::Float64
    range::Float64
    inv_range_sq::Float64
end

"""
    inverseWeightFunction(alpha::Real=0.0, range::Real=0.0)

Functor that calculates an inverse-square distance weight.
The parameters `alpha` and `range` are included for a consistent
interface but are not used in the calculation.
"""
struct inverseWeightFunction <: MLSWeightFunction
    alpha::Float64
    range::Float64
end

struct Kin2Macro{M}
    ranges::NTuple{M, UnitRange{Int}}
end

abstract type GridMover end

# ---------------------------------------------------------
# 1. NoGridMover
# ---------------------------------------------------------
struct NoGridMover <: GridMover end

# ---------------------------------------------------------
# 2. CustomGridMover
# ---------------------------------------------------------
struct CustomGridMover{F, P} <: GridMover
    vel_func::F
    params::P
end

# ---------------------------------------------------------
# 3. PhysicalGridMover
# ---------------------------------------------------------
mutable struct PhysicalGridMover{D} <: GridMover
    vel_indices::Space{D}
    grid_velocities::Vector{Space{D}} # Pre-allocated workspace buffer
end

# ---------------------------------------------------------
# 1. Grid Metadata
# ---------------------------------------------------------
mutable struct GridMetadata{D}
    N::Int                  
    N_interior::Int         
    N_ghost::Int            
    mins::Space{D}
    maxs::Space{D}
    inner_mins::Space{D}
    inner_maxs::Space{D}
    R::Float64
    r::Float64
    dx::Space{D} 
    regular::Bool
    bc::Symbol              
    range_factor::Float64
    max_nb::Int       
end

# ---------------------------------------------------------
# 2. Shared Workspace Buffers
# ---------------------------------------------------------
mutable struct SharedBuffers{D, M}
    rho_buffer::Vector{State{M}}      
    pos_buffer::Vector{Space{D}} 
    bit_buffer::Vector{Bool}
    int_buffer::Vector{Int}
end

# # ---------------------------------------------------------
# # 3. Neighbor Search Context
# # ---------------------------------------------------------
# mutable struct NeighborData{D, S, WF}
#     system::S      
#     weight_func::WF

#     # CSR format using native UnitRanges
#     ranges::Vector{UnitRange{Int}}
#     indices::Vector{Int}
    
#     # --- THE MASSIVE CHANGE ---
#     weights::Vector{Float64}
#     distances::Vector{Space{D}} 

#     atomic_counts::Vector{Atomic{Int}}
#     atomic_offsets::Vector{Atomic{Int}}
# end

# ---------------------------------------------------------
# 3. Neighbor Search Context
# ---------------------------------------------------------
mutable struct NeighborData{D, S, WF}
    system::S      
    weight_func::WF

    # CSR format using native UnitRanges
    ranges::Vector{UnitRange{Int}}
    indices::Vector{Int}
    
    weights::Vector{Float64}
    distances::Vector{Space{D}} 

    # --- Pure Serial Buffers ---
    counts::Vector{Int}
    offsets::Vector{Int}
end

# ---------------------------------------------------------
# 4. Reordering / Sorting Context
# ---------------------------------------------------------
struct ReorderData{D}
    permutation::Vector{Int}          
    inv_permutation::Vector{Int}      
    new_permutation_buffer::Vector{Int} 
    seen_buffer::Vector{Bool}            
end

# ---------------------------------------------------------
# 5. Particle Management Context
# ---------------------------------------------------------
struct ManagementData{D, M}
    local_voxels::LocalVoxels
end

# ---------------------------------------------------------
# 6. Particle Grid Core (Geometry & Topology)
# ---------------------------------------------------------
mutable struct ParticleGridCore{D}
    positions::Vector{Space{D}}
    is_boundary::Vector{Bool}
    volumes::Vector{Float64}
end

# ---------------------------------------------------------
# 7. The Top-Level Particle Grid
# ---------------------------------------------------------
mutable struct ParticleGrid{D, M, S, WF, GM}
    meta::GridMetadata{D}
    core::ParticleGridCore{D}
    shared::SharedBuffers{D, M}
    neighbor::NeighborData{D, S, WF}
    reorder::ReorderData{D}
    manage::ManagementData{D, M}
    kin2macro::Kin2Macro{M}
    mover::GM

    rhos::Vector{State{M}}
    mood_events::Vector{SVector{M, Bool}}
    curvatures::Vector{State{M}}
    
end

# --- Aliases for convenience ---
const ParticleGrid1D{M, S, WF, GM} = ParticleGrid{1, M, S, WF, GM}
const ParticleGrid2D{M, S, WF, GM} = ParticleGrid{2, M, S, WF, GM}

