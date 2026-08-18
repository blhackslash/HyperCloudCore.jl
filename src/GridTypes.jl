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

abstract type AbstractBoundaryCondition end

# =========================================================================
# CONCRETE BOUNDARY CONDITIONS
# =========================================================================

struct FixedDirichlet <: AbstractBoundaryCondition end
struct OutflowBC <: AbstractBoundaryCondition end

# ---------------------------------------------------------
# 1. Grid Metadata
# ---------------------------------------------------------
mutable struct GridMetadata{D, T}
    N::Int
    N_interior::Int
    N_ghost::Int
    
    R::T
    max_dx::T
    dx::Space{D, T}
    
    interp_range_factor::T
    max_nb::Int
end

# 1. Base Domain Abstract Type
abstract type AbstractDomain{D, T} end

# 2. General Custom/Rectangular Domain Struct
struct Domain{Shape, D, T, F_Valid, F_Interior, F_Tag} <: AbstractDomain{D, T}
    canvas_mins::Space{D, T}
    canvas_maxs::Space{D, T}
    
    interior_mins::Space{D, T}
    interior_maxs::Space{D, T}
    
    is_periodic::SVector{D, Bool}
    L::Space{D, T}
    L_inv::Space{D, T}
    L_wrap::Space{D, T}
    invL_wrap::Space{D, T}
    
    # Geometry Closures
    is_valid::F_Valid        # True if inside canvas (interior + ghosts)
    is_interior::F_Interior  # NEW: True if strictly inside the physical fluid domain
    get_tag::F_Tag           # Returns >0 for specific walls
    
    bc_map::Dict{Int, AbstractBoundaryCondition}
end

const RectangularDomain{D, T, F_Valid, F_Interior, F_Tag} = Domain{Val{:rectangular}, D, T, F_Valid, F_Interior, F_Tag}

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
    tags::Vector{Int}
end

@inline get_n_offsets(::Val{1}) = 3
@inline get_n_offsets(::Val{2}) = 9
@inline get_n_offsets(::Val{3}) = 27

struct GlobalBins{D, T, N_OFF}
    mins::Space{D, T}
    maxs::Space{D, T}
    coarse_size::Space{D, T}
    coarse_dims::NTuple{D, Int}
    head::Vector{Int}
    next::Vector{Int}
    bin_neighbors::Vector{SVector{N_OFF, Int}} 
end

function GlobalBins(
    ::Type{T}, domain_mins::Space{D, T}, domain_maxs::Space{D, T}, 
    R::Real, max_particles::Int
) where {D, T}
    
    # We no longer need to convert to Space{D,T} since they are already passed as such
    domain_size = domain_maxs .- domain_mins

    coarse_dims = ntuple(d -> max(1, ceil(Int, domain_size[d] / R)), Val(D))
    coarse_size = domain_size ./ coarse_dims

    total_coarse_bins = prod(coarse_dims)
    head = zeros(Int, total_coarse_bins)
    next = zeros(Int, ceil(Int, max_particles * 1.25))
    
    N_OFF = get_n_offsets(Val(D))
    bin_neighbors = Vector{SVector{N_OFF, Int}}(undef, 0)

    # Note: Ensure your GlobalBins struct definition also had BC removed!
    # i.e., struct GlobalBins{D, T, N_OFF}
    return GlobalBins{D, T, N_OFF}(
        domain_mins, domain_maxs, coarse_size, coarse_dims, head, next, bin_neighbors
    )
end

struct ParticleGrid{D, M, T, WF, GM, N_OFF, Dom <: AbstractDomain{D, T}}
    meta::GridMetadata{D, T}
    domain::Dom                  # <--- NEW FIELD
    core::ParticleGridCore{D, T}
    shared::SharedBuffers{D, M, T}
    neighbor::NeighborData{D, T, WF}
    reorder::ReorderData{D}
    bins::GlobalBins{D, T, N_OFF} # (Note: GlobalBins might also need its BC parameter removed)
    mover::GM
    
    rhos::Vector{State{M, T}}
    mood_events::Vector{SVector{M, Bool}}
    curvatures::Vector{State{M, T}}
end