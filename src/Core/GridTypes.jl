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
# Inside GridTypes.jl
struct PhysicalGridMover{D} <: GridMover
    # Maps spatial dimensions to the macroscopic state index representing velocity
    # e.g., for a 1D scalar it's (1,), for 2D Euler it might be (2, 3)
    vel_indices::NTuple{D, Int} 
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
    a::Float64
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

# ---------------------------------------------------------
# 3. Neighbor Search Context
# ---------------------------------------------------------
mutable struct NeighborData{D, WF}    
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
# 6. Particle Grid Core (Geometry & Topology)
# ---------------------------------------------------------
mutable struct ParticleGridCore{D}
    positions::Vector{Space{D}}
    is_boundary::Vector{Bool}
    volumes::Vector{Float64}
end

# Add BC parameter to GlobalBins
struct GlobalBins{D, BC}
    mins::Space{D}
    maxs::Space{D}
    coarse_size::Float64
    coarse_dims::NTuple{D, Int}
    head::Vector{Int}
    next::Vector{Int}
    fine_size::Float64
    fine_dims::NTuple{D, Int}
    fine_occupation::Vector{Bool}
    fine_type::Vector{UInt8}
end

# Add BC parameter to ParticleGrid and pass it to GlobalBins
mutable struct ParticleGrid{D, M, WF, GM, BC}
    meta::GridMetadata{D}
    core::ParticleGridCore{D}
    shared::SharedBuffers{D, M}
    neighbor::NeighborData{D, WF}
    reorder::ReorderData{D}
    bins::GlobalBins{D, BC}    # <-- Now type-linked
    mover::GM

    rhos::Vector{State{M}}
    mood_events::Vector{SVector{M, Bool}}
    curvatures::Vector{State{M}}
end

# Update the Aliases
const ParticleGrid1D{M, WF, GM, BC} = ParticleGrid{1, M, WF, GM, BC}
const ParticleGrid2D{M, WF, GM, BC} = ParticleGrid{2, M, WF, GM, BC}


