module Meshfree4ScalarEq

export runSimulation, GAS_GAMMA_EULER, DEBUG_TARGET_PARTICLE, DEBUG_TARGET_STEP

include("CoreUtils.jl")
using .CoreUtils
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

include("../SimulationFunctions/runSimulation.jl")

end  # module 