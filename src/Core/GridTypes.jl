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

# ---------------------------------------------------------
# 1. Grid Metadata
# ---------------------------------------------------------
mutable struct GridMetadata{D}
    N::Int                  
    N_interior::Int         
    N_ghost::Int            
    mins::SVector{D, Float64}
    maxs::SVector{D, Float64}
    h::Float64              
    dx::SVector{D, Float64} 
    regular::Bool
    bc::Symbol              
    range_factor::Float64
    max_nb::Int       
end

# ---------------------------------------------------------
# 2. Shared Workspace Buffers
# ---------------------------------------------------------
mutable struct SharedBuffers{D, M}
    rho_buffer::Matrix{Float64}      
    pos_buffer::Vector{SVector{D, Float64}} 
    bit_buffer::Vector{Bool}
    int_buffer::Vector{Int}
end

# ---------------------------------------------------------
# 3. Neighbor Search Context
# ---------------------------------------------------------
mutable struct NeighborData{D, S, WF}
    system::S      
    weight_func::WF

    # CSR format using native UnitRanges
    ranges::Vector{UnitRange{Int}}
    indices::Vector{Int}
    
    # Matrix holding (Weight, dx, [dy, dz])
    data::Matrix{Float64} 

    atomic_counts::Vector{Atomic{Int}}
    atomic_offsets::Vector{Atomic{Int}}
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
    merge_flags::Vector{Bool}
    split_targets::Vector{Int}
    split_buffer_pos::Vector{SVector{D, Float64}}
    split_buffer_rho::Vector{NTuple{M, Float64}} # M-sized Tuples for N x M matrix rows
    local_voxels::LocalVoxels
end

# ---------------------------------------------------------
# 6. Particle Grid Core (Geometry & Topology)
# ---------------------------------------------------------
mutable struct ParticleGridCore{D}
    positions::Vector{SVector{D, Float64}}
    is_boundary::Vector{Bool}
    volumes::Vector{Float64}
end

# ---------------------------------------------------------
# 7. The Top-Level Particle Grid
# ---------------------------------------------------------
mutable struct ParticleGrid{D, M, S, WF}
    meta::GridMetadata{D}
    core::ParticleGridCore{D}
    shared::SharedBuffers{D, M}
    neighbor::NeighborData{D, S, WF}
    reorder::ReorderData{D}
    manage::ManagementData{D,M}
    
    rhos::Matrix{Float64}
    mood_events::Matrix{Bool}
    curvatures::Matrix{Float64}
end

# --- Aliases for convenience ---
const ParticleGrid1D{M, S, WF} = ParticleGrid{1, M, S, WF}
const ParticleGrid2D{M, S, WF} = ParticleGrid{2, M, S, WF}



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
mutable struct PhysicalGridMover{E, I, V, D} <: GridMover
    pde::E
    interpolator::I
    vel_kinetic_indices::V                       # Indices of the driving kinetic variables
    grid_velocities::Vector{SVector{D, Float64}} # Pre-allocated workspace buffer

    # Constructor 1: For Scalar Equations (No indices needed)
    function PhysicalGridMover(pde::E, interp::I) where {E, I}
        new{E, I, Nothing, 1}(pde, interp, nothing, Vector{SVector{1, Float64}}(undef, 0))
    end

    # Constructor 2: For Systems (Takes driving indices and Dimension)
    function PhysicalGridMover(pde::E, interp::I, vel_indices::V, ::Val{D}) where {E, I, V, D}
        new{E, I, V, D}(pde, interp, vel_indices, Vector{SVector{D, Float64}}(undef, 0))
    end
end