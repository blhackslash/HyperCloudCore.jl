export @pebug, DEBUG_TARGET_PARTICLE

# 1. Define a global, type-stable Reference to hold our debug state.
# We default it to false.
const IS_DEBUG = Ref(false)

const DEBUG_TARGET_PARTICLE = Ref(-1)
const DEBUG_TARGET_STEP = Ref(-1)     # Optional: filter by time step

function __init__()    
    # 2. Update the global Ref based on the logger.
    # Now this state is saved and globally accessible.
    IS_DEBUG[] = Logging.min_enabled_level(Logging.current_logger()) <= Logging.Debug
end

macro pebug(p_idx, msg, args...)
    debug_call = Expr(:macrocall, Symbol("@debug"), __source__, msg, args...)
    
    return quote
        # 3. Check our extremely fast IS_DEBUG reference first.
        # If it's false, the CPU skips the rest immediately.
        if IS_DEBUG[] && $(esc(p_idx)) == DEBUG_TARGET_PARTICLE[]
            $(esc(debug_call))
        end
    end
end

function safe_resize!(vec::AbstractVector, N::Int)
    if length(vec) < N
        N_new = N + N ÷ 4
        resize!(vec, N_new)
    end
end
"""
Ensures a vector `v` has at least capacity `n`.
Resizes if `length(v) < n`.
"""
function _ensure_capacity!(v::AbstractVector, n::Int)
    if length(v) < n
        n = n + n ÷ 4
        resize!(v, n)
    end
    return nothing
end
# Case 1: Single Float/Int -> 1D Space, 1 Component (Scalar PDE in 1D)
# Example input: 2.5
# Output: SVector{1, SVector{1, Float64}}([ [2.5] ])
@inline param2svec(v::Real) = SVector{1, SVector{1, Float64}}((SVector{1, Float64}(Float64(v)),))

# Case 2: 1D Tuple -> D-Dimensional Space, 1 Component (Scalar PDE in Multi-D)
# Example input: (1.5, 2.0)
# Output: SVector{2, SVector{1, Float64}}([ [1.5], [2.0] ])
@inline param2svec(v::NTuple{D, <:Real}) where {D} = 
    SVector{D, SVector{1, Float64}}(ntuple(i -> SVector{1, Float64}(Float64(v[i])), Val(D)))

# Case 3: Tuple of Tuples -> D-Dimensional Space, M Components (System PDE in Multi-D)
# Example input: ((1.0, 0.0), (0.0, 1.0))
# Output: SVector{2, SVector{2, Float64}}([ [1.0, 0.0], [0.0, 1.0] ])
@inline param2fvec(v::NTuple{D, NTuple{M, <:Real}}) where {D, M} = 
    Flux{D,M}(ntuple(i -> State{M}(Float64.(v[i])), Val(D)))
@inline param2fvec(v::Float64)=Flux{1,1}(((v,),))
# Handles the (Vector{Float64},) or (Vector, Vector) format from runSimulation.jl
function param2fvec(v::NTuple{D, Vector{T}}) where {D, T <: Real}
    # NK is the number of kinetic components (length of the vector)
    NK = length(v[1])
    return SVector{D}(ntuple(d -> SVector{NK, Float64}(v[d]), Val(D)))
end

# Handles a raw Vector{Vector{Float64}} if passed directly
function param2fvec(v::Vector{Vector{T}}) where {T <: Real}
    D = length(v)
    NK = length(v[1])
    return SVector{D}(ntuple(d -> SVector{NK, Float64}(v[d]), Val(D)))
end

# Handles a single Vector (for 1D, NK-component systems)
function param2fvec(v::Vector{T}) where {T <: Real}
    NK = length(v)
    return SVector{1}( (SVector{NK, Float64}(v),) )
end

# Case 4: Fallback if it is already correctly formatted
@inline param2svec(v::Flux{D,M}) where {D, M} = v

# --- System State Conversion (M components) ---
@inline param2uvec(v::Real) = SVector{1, Float64}(Float64(v))
@inline param2uvec(v::NTuple{M, <:Real}) where {M} = State{M}(Float64.(v))
@inline param2uvec(v::State{M}) where {M} = v

# --- Spatial Geometry Conversion (D dimensions) ---
@inline param2xvec(x::Real) = SVector{1, Float64}(Float64(x))
@inline param2xvec(x::NTuple{D, <:Real}) where {D} = Space{D}(Float64.(x))
@inline param2xvec(x::Space{D}) where {D} = x

# =========================================================================
# MATHEMATICAL BRANCHLESS SIMD HELPERS
# =========================================================================
@inline math_max(a::Float64, b::Float64) = 0.5 * (a + b + abs(a - b))
@inline math_min(a::Float64, b::Float64) = 0.5 * (a + b - abs(a - b))
@inline math_max(a::AbstractVector, b::AbstractVector) = 0.5 * (a + b + abs.(a - b))
@inline math_min(a::AbstractVector, b::AbstractVector) = 0.5 * (a + b - abs.(a - b))