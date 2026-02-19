module Meshfree4ScalarEq

export runScalarSimulation, runSystemSimulation, GAS_GAMMA_EULER

# Imports
using Random
using Logging



# --- NEW: Add this function at the end of your module ---
function __init__()
    # This code will run once when the module is loaded.
    # It sets the logger for the entire application.
    min_level_to_show = Logging.Info
    global_logger(ConsoleLogger(stderr, min_level_to_show))
    println("Logger initialized to show ",min_level_to_show,"-Level.")
end
# ---------------------------------------------------------

# No internal Dependencies
include("HyperbolicPDEs.jl")
using .HyperbolicPDEs

include("InterpolationUtils.jl")
using .InterpolationUtils

include("MLSWeightFunctions.jl")
using .MLSWeightFunctions

include("SimSettings.jl")
using .SimSettings

# Minimal internal Dependencies (HyperbolicPDEs)

include("FluxFunctions.jl")
using .FluxFunctions

# Needs Particle Grids

include("ParticleGrids.jl")
using .ParticleGrids

include("SourceTerms.jl")
using .SourceTerms

include("ImplicitSolvers.jl")
using .ImplicitSolvers

include("InitialConditions.jl")
using .InitialConditions

include("MOOD.jl")
using .MOOD

include("GridMovement.jl")
using .GridMovement

# Significant internal Dependencies

include("Interpolations.jl")
using .Interpolations

include("TimeIntegration.jl")
using .TimeIntegration

using IPlotPDESols

include("../SimulationFunctions/runScalarSimulation.jl")
include("../SimulationFunctions/runSystemSimulation.jl")

end  # module 