
# Conversions
export param2uvec, param2xvec, param2svec, param2fvec, prim2cons, cons2prim
export set_threads!, @pebug, DEBUG_TARGET_PARTICLE, set_thread_tolerance!


const IS_DEBUG = Ref(false)
const DEBUG_TARGET_PARTICLE = Ref(-1)
const DEBUG_TARGET_STEP = Ref(-1)

function __init__()    
    IS_DEBUG[] = Logging.min_enabled_level(Logging.current_logger()) <= Logging.Debug
end

"""
    @pebug p_idx msg args...

A targeted debugging macro that conditionally prints debug information.
- Only executes if the global `IS_DEBUG` flag is active and the provided `p_idx` strictly matches the globally set `DEBUG_TARGET_PARTICLE`.
"""
macro pebug(p_idx, msg, args...)
    debug_call = Expr(:macrocall, Symbol("@debug"), __source__, msg, args...)
    return quote
        if IS_DEBUG[] && $(esc(p_idx)) == DEBUG_TARGET_PARTICLE[]
            $(esc(debug_call))
        end
    end
end

const _THREAD_THRESHOLD = Ref(50000)
const _THREAD_TOLERANCE = Ref(100)
const _USE_THREADS      = Ref(false)

_use_threads() = _USE_THREADS[]
set_threads!(N::Int) = _THREAD_THRESHOLD[] < N ? _USE_THREADS[] = true : nothing
set_threads!(B::Bool) = (_USE_THREADS[] = B)
set_thread_tolerance!(N::Int) = _THREAD_TOLERANCE[] = N/100

"""
    @smart_parallel condition loop

A scheduling macro that dynamically dispatches for-loops to different multithreading backends based on a boolean condition.
- Dispatches to `Base.Threads.@threads :dynamic` if the condition is true.
- Dispatches to `Polyester.@batch` for lower-overhead threading if the condition is false.
"""
macro smart_parallel(condition, loop)
    if loop.head !== :for
        error("@smart_parallel requires a for loop expression")
    end
    return esc(quote
        if $condition
            Base.Threads.@threads :dynamic $loop
        else
            @batch $loop
        end
    end)
end


function safe_resize!(vec::AbstractVector, N::Int)
    if length(vec) < N
        N_new = N + N ÷ 4
        resize!(vec, N_new)
    end
end

@inline function ensure_capacity!(vec::AbstractVector, req_capacity::Int)
    if length(vec) < req_capacity
        resize!(vec, ceil(Int, req_capacity * 1.25))
    end
end

# =========================================================================
# PARAMETER CASTING (Fully Parameterized)
# =========================================================================
"""
    param2uvec(v)
    param2xvec(v)
    param2svec(v)
    param2fvec(v)

Type-casting utilities that convert generalized user inputs (Scalars, Tuples, or standard Arrays) into the highly optimized, strictly typed `StaticArrays` utilized internally by the solver.
- `param2uvec` and `param2xvec`: Convert inputs into standard `SVector` formulations.
- `param2svec`: Nests values into vectors of vectors representing multidimensional data.
- `param2fvec`: Specifically constructs `Flux{D, M, T}` types from tuple or array representations.
"""
@inline param2svec(v::T) where {T <: Real} = SVector{1, SVector{1, T}}((SVector{1, T}(v),))
@inline param2svec(v::NTuple{D, T}) where {D, T <: Real} = SVector{D, SVector{1, T}}(ntuple(i -> SVector{1, T}(v[i]), Val(D)))

@inline param2fvec(v::NTuple{D, NTuple{M, T}}) where {D, M, T <: Real} = Flux{D, M, T}(ntuple(i -> State{M, T}(v[i]), Val(D)))
@inline param2fvec(v::T) where {T <: Real} = Flux{1, 1, T}(((v,),))

function param2fvec(v::NTuple{D, Vector{T}}) where {D, T <: Real}
    NK = length(v[1])
    return SVector{D}(ntuple(d -> SVector{NK, T}(v[d]), Val(D)))
end

function param2fvec(v::Vector{Vector{T}}) where {T <: Real}
    D = length(v)
    NK = length(v[1])
    return SVector{D}(ntuple(d -> SVector{NK, T}(v[d]), Val(D)))
end

function param2fvec(v::Vector{T}) where {T <: Real}
    NK = length(v)
    return SVector{1}((SVector{NK, T}(v),))
end

@inline param2svec(v::Flux{D, M, T}) where {D, M, T} = v

@inline param2uvec(v::T) where {T <: Real} = SVector{1, T}(v)
@inline param2uvec(v::Tuple) = SVector{length(v), eltype(v)}(v)
@inline param2uvec(v::SVector) = v 
@inline param2uvec(v::AbstractVector) = SVector{length(v), eltype(v)}(v)

@inline param2xvec(x::T) where {T <: Real} = SVector{1, T}(x)
@inline param2xvec(x::Tuple) = SVector{length(x), eltype(x)}(x)
@inline param2xvec(x::SVector) = x 
@inline param2xvec(x::AbstractVector) = SVector{length(x), eltype(x)}(x)

# =========================================================================
# MATHEMATICAL BRANCHLESS SIMD HELPERS
# =========================================================================

"""
    math_max(a, b)
    math_min(a, b)

Branchless, SIMD-friendly helper functions computing maximums and minimums.
- Calculates the result algebraically using `0.5 * (a + b ± abs(a - b))` to avoid branching overhead during hot-loop execution.
"""
@inline math_max(a::T, b::T) where {T <: Real} = T(0.5) * (a + b + abs(a - b))
@inline math_min(a::T, b::T) where {T <: Real} = T(0.5) * (a + b - abs(a - b))
@inline math_max(a::AbstractVector{T}, b::AbstractVector{T}) where {T} = T(0.5) * (a + b + abs.(a - b))
@inline math_min(a::AbstractVector{T}, b::AbstractVector{T}) where {T} = T(0.5) * (a + b - abs.(a - b))

"""
    sort_flux(f_i, f_j, F_i, F_j, dist_k)

Resolves the left and right state and flux interfaces required for numerical flux evaluation.
- Utilizes the spatial directionality of the distance vector `dist_k` to appropriately assign `f_i` and `f_j` to the left (`f_L`) or right (`f_R`) interface slots.
"""
@inline function sort_flux(f_i::State{M, T}, f_j::State{M, T}, F_i::Flux{D, M, T}, F_j::Flux{D, M, T}, dist_k::Space{D, T}) where {D, M, T}
    f_L = Flux{D, M, T}(ntuple(d -> dist_k[d] > zero(T) ? f_i : f_j, Val(D)))
    f_R = Flux{D, M, T}(ntuple(d -> dist_k[d] > zero(T) ? f_j : f_i, Val(D)))
    
    F_L = Flux{D, M, T}(ntuple(d -> dist_k[d] > zero(T) ? F_i[d] : F_j[d], Val(D)))
    F_R = Flux{D, M, T}(ntuple(d -> dist_k[d] > zero(T) ? F_j[d] : F_i[d], Val(D)))
    
    return f_L, f_R, F_L, F_R
end