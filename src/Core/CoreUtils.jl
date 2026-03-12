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