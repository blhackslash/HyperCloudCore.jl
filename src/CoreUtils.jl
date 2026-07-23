export @pebug, DEBUG_TARGET_PARTICLE, set_thread_tolerance!

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

const _THREAD_THRESHOLD = Ref(50000)
const _THREAD_TOLERANCE = Ref(80)
const _USE_THREADS      = Ref(false)

_use_threads() = _USE_THREADS[]
_set_threads!(N::Int) = _THREAD_THRESHOLD[] < N ? _USE_THREADS[] = true : nothing
set_thread_tolerance!(N::Int) = _THREAD_TOLERANCE[] = N/100


"""
    @smart_parallel condition for_loop

Dynamically routes a loop to `Threads.@threads :dynamic` if `condition` is true, 
otherwise routes it to Polyester's `@batch`.
"""
macro smart_parallel(condition, loop)
    if loop.head !== :for
        error("@smart_parallel requires a for loop expression")
    end
    
    # Escape the entire quote block so the raw loop is visible to the inner macros
    return esc(quote
        if $condition
            Base.Threads.@threads :dynamic $loop
        else
            @batch $loop
        end
    end)
end

"""
    calculate_thread_threshold(pg::ParticleGrid, main_grad; l3_cache_mb=24)

Calculates the maximum number of particles that can comfortably fit in the CPU L3 Cache 
before memory bandwidth saturation requires switching to `Threads.@threads`.
"""
function calculate_thread_threshold(
    pg::ParticleGrid{D, M}, 
    main_grad::MUSCL{D, M, B_LEN}
) where {D, M, B_LEN}
    
    l3_cache_bytes = Int(CPUSummary.cache_size(Val(3))) * CPUSummary.num_cores()
    
    # Estimate average neighbors per particle
    N_nb = length(pg.neighbor.indices) / max(1, pg.meta.N)
    
    # 1. Base Arrays (rhos, stage candidates, divergences) ≈ 3 vectors of State{M}
    # 8 bytes per Float64 * M components
    base_bytes = 3 * 8 * M
    
    # 2. High-Order Gradients
    # SVector{B_LEN, State{M}}
    grad_bytes = 8 * B_LEN * M
    
    # 3. Neighborhood Access (Indices, Weights, Space{D} Distances)
    nb_bytes = N_nb * 8 * (1 + 1 + D)
    
    bytes_per_particle = base_bytes + grad_bytes + nb_bytes
    
    return floor(Int, l3_cache_bytes / bytes_per_particle * _THREAD_TOLERANCE[])
end

# Fallback for standard fixed-order or geometric methods without a dynamic B_LEN
function calculate_thread_threshold(pg::ParticleGrid{D, M}, ::Any; l3_cache_mb::Real=24.0) where {D, M}
    l3_cache_bytes = Int(CPUSummary.cache_size(Val(3))) * CPUSummary.num_cores()
    N_nb = length(pg.neighbor.indices) / max(1, pg.meta.N)
    
    base_bytes = 3 * 8 * M
    nb_bytes = N_nb * 8 * (1 + 1 + D)
    
    return floor(Int, l3_cache_bytes / (base_bytes + nb_bytes))
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
@inline function ensure_capacity!(vec::AbstractVector, req_capacity::Int)
    if length(vec) < req_capacity
        resize!(vec, ceil(Int, req_capacity * 1.25))
    end
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

@inline param2uvec(v::Real) = SVector{1, Float64}(v)
@inline param2uvec(v::Tuple) = SVector{length(v), Float64}(v)
@inline param2uvec(v::SVector) = v # Pass-through
@inline param2uvec(v::AbstractVector) = SVector{length(v), Float64}(v)

# --- For Space{D} (Spatial Coordinates & Dimensions) ---
@inline param2xvec(x::Real) = SVector{1, Float64}(x)
@inline param2xvec(x::Tuple) = SVector{length(x), Float64}(x)
@inline param2xvec(x::SVector) = x # Pass-through
@inline param2xvec(x::AbstractVector) = SVector{length(x), Float64}(x)

# =========================================================================
# MATHEMATICAL BRANCHLESS SIMD HELPERS
# =========================================================================
@inline math_max(a::Float64, b::Float64) = 0.5 * (a + b + abs(a - b))
@inline math_min(a::Float64, b::Float64) = 0.5 * (a + b - abs(a - b))
@inline math_max(a::AbstractVector, b::AbstractVector) = 0.5 * (a + b + abs.(a - b))
@inline math_min(a::AbstractVector, b::AbstractVector) = 0.5 * (a + b - abs.(a - b))

# Defaults
@inline function prim2cons(eq, U)
    return U
end

@inline function cons2prim(eq, U)
    return U
end