export @pebug, DEBUG_TARGET_PARTICLE, set_thread_tolerance!

const IS_DEBUG = Ref(false)
const DEBUG_TARGET_PARTICLE = Ref(-1)
const DEBUG_TARGET_STEP = Ref(-1)

function __init__()    
    IS_DEBUG[] = Logging.min_enabled_level(Logging.current_logger()) <= Logging.Debug
end

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

function calculate_thread_threshold(
    pg::ParticleGrid{D, M, T}, 
    main_grad::MUSCL{D, M, T, B_LEN}
) where {D, M, T, B_LEN}
    
    l3_cache_bytes = Int(CPUSummary.cache_size(Val(3))) * CPUSummary.num_cores()
    N_nb = length(pg.neighbor.indices) / max(1, pg.meta.N)
    
    base_bytes = 3 * sizeof(T) * M
    grad_bytes = sizeof(T) * B_LEN * M
    nb_bytes = N_nb * sizeof(T) * (1 + 1 + D)
    
    bytes_per_particle = base_bytes + grad_bytes + nb_bytes
    return floor(Int, l3_cache_bytes / bytes_per_particle * _THREAD_TOLERANCE[])
end

function calculate_thread_threshold(pg::ParticleGrid{D, M, T}, ::Any; l3_cache_mb::Real=24.0) where {D, M, T}
    l3_cache_bytes = Int(CPUSummary.cache_size(Val(3))) * CPUSummary.num_cores()
    N_nb = length(pg.neighbor.indices) / max(1, pg.meta.N)
    
    base_bytes = 3 * sizeof(T) * M
    nb_bytes = N_nb * sizeof(T) * (1 + 1 + D)
    return floor(Int, l3_cache_bytes / (base_bytes + nb_bytes))
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
@inline math_max(a::T, b::T) where {T <: Real} = T(0.5) * (a + b + abs(a - b))
@inline math_min(a::T, b::T) where {T <: Real} = T(0.5) * (a + b - abs(a - b))
@inline math_max(a::AbstractVector{T}, b::AbstractVector{T}) where {T} = T(0.5) * (a + b + abs.(a - b))
@inline math_min(a::AbstractVector{T}, b::AbstractVector{T}) where {T} = T(0.5) * (a + b - abs.(a - b))

@inline prim2cons(eq, U) = U
@inline cons2prim(eq, U) = U