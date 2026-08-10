## ------------------------------- Weight Functions ------------------------------
abstract type MLSWeightFunction end

"""
    ExponentialWeightFunction(alpha::Real, range::Real)
"""
struct ExponentialWeightFunction{T} <: MLSWeightFunction
    alpha::T
    range::T
    inv_range_sq::T
end

"""
    InverseWeightFunction(alpha::Real=0.0, range::Real=0.0)
"""
struct InverseWeightFunction{T} <: MLSWeightFunction
    alpha::T
    range::T
end

abstract type GridMover end
struct NoGridMover <: GridMover end
struct CustomGridMover{F, P} <: GridMover
    vel_func::F
    params::P
end
struct PhysicalGridMover{D} <: GridMover
    vel_indices::NTuple{D, Int} 
end

# ---------------------------------------------------------
# 1. Grid Metadata
# ---------------------------------------------------------
mutable struct GridMetadata{D, T}
    N::Int                  
    N_interior::Int         
    N_ghost::Int            
    mins::Space{D, T}
    maxs::Space{D, T}
    inner_mins::Space{D, T}
    inner_maxs::Space{D, T}
    L::Space{D, T}
    L_inv::Space{D, T}
    R::T
    r::T
    a::T
    dx::Space{D, T} 
    regular::Bool
    bc::Symbol              
    range_factor::T
    max_nb::Int       
end

# ---------------------------------------------------------
# 2. Shared Workspace Buffers
# ---------------------------------------------------------
mutable struct SharedBuffers{D, M, T}
    rho_buffer::Vector{State{M, T}}      
    pos_buffer::Vector{Space{D, T}}
    float_buffer::Vector{T}
    bit_buffer::Vector{Bool}
    int_buffer::Vector{Int}
end

# ---------------------------------------------------------
# 3. Neighbor Search Context
# ---------------------------------------------------------
mutable struct NeighborData{D, T, WF}    
    weight_func::WF
    ranges::Vector{UnitRange{Int}}
    indices::Vector{Int}
    weights::Vector{T}
    distances::Vector{Space{D, T}} 
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
mutable struct ParticleGridCore{D, T}
    positions::Vector{Space{D, T}}
    is_boundary::Vector{Bool}
    volumes::Vector{T}
end

@inline get_n_offsets(::Val{1}) = 3
@inline get_n_offsets(::Val{2}) = 9
@inline get_n_offsets(::Val{3}) = 27

struct GlobalBins{D, T, BC, N_OFF}
    mins::Space{D, T}
    maxs::Space{D, T}
    coarse_size::Space{D, T}
    coarse_dims::NTuple{D, Int}
    head::Vector{Int}
    next::Vector{Int}
    fine_size::Space{D, T}
    fine_dims::NTuple{D, Int}
    fine_occupation::Vector{Bool}
    fine_type::Vector{UInt8}
    bin_neighbors::Vector{SVector{N_OFF, Int}} 
end

mutable struct ParticleGrid{D, M, T, WF, GM, BC, N_OFF}
    meta::GridMetadata{D, T}
    core::ParticleGridCore{D, T}
    shared::SharedBuffers{D, M, T}
    neighbor::NeighborData{D, T, WF}
    reorder::ReorderData{D}
    bins::GlobalBins{D, T, BC, N_OFF} 
    mover::GM

    rhos::Vector{State{M, T}}
    mood_events::Vector{SVector{M, Bool}}
    curvatures::Vector{State{M, T}}
end