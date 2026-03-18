export @pebug, DEBUG_TARGET_PARTICLE

# 1. Define a global, type-stable Reference to hold our debug state.
# We default it to false.
const IS_DEBUG = Ref(false)

const DEBUG_TARGET_PARTICLE = Ref(-1)
const DEBUG_TARGET_STEP = Ref(-1)     # Optional: filter by time step

function __init__()
    # This code will run once when the module is loaded.
    min_level_to_show = Logging.Info
    global_logger(ConsoleLogger(stderr, min_level_to_show))
    println("Logger initialized to show ", min_level_to_show, "-Level.")
    
    # 2. Update the global Ref based on the logger.
    # Now this state is saved and globally accessible.
    IS_DEBUG[] = Logging.min_enabled_level(Logging.current_logger()) <= Logging.Debug
end

# function __init__()
#     min_level_to_show = Logging.Debug
    
#     # 1. Open a file stream in write ("w") or append ("a") mode
#     log_io = open("./logs/simulation_debug.log", "w")
    
#     # 2. Pass the file stream to SimpleLogger
#     file_logger = SimpleLogger(log_io, min_level_to_show)
    
#     # 3. Set it as the global logger
#     global_logger(file_logger)
    
#     println("Logger initialized to write ", min_level_to_show, "-Level to file.")
#     IS_DEBUG[] = Logging.min_enabled_level(Logging.current_logger()) <= Logging.Debug
#     DEBUG_TARGET_PARTICLE[] = 28
# end

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
@inline param2svec(v::NTuple{D, NTuple{M, <:Real}}) where {D, M} = 
    Flux{D,M}(ntuple(i -> State{M}(Float64.(v[i])), Val(D)))

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