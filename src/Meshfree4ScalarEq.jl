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

using Test
using LinearAlgebra

# --- Setup needed for the user's snippet ---
struct Order0 end; struct Order1 end
const DO0 = Order0(); const DO1 = Order1()
const GAS_GAMMA_EULER = 1.4

# Mock/Ensure types exist as defined in your snippet
# (Assuming LEuler1D and LinePath are already in your module)

function run_integration_tests(eq)
    println("--- Testing Numerical Path Integral for LEuler1D ---")
    
    # Tolerance for 5-point Gauss-Lobatto on these specific functions
    # Note: GL5 is exact for polynomials up to degree 7. 
    # Logarithmic terms (1/rho) will have slight quadrature errors.
    tol = 1e-10

    @testset "Path Integral Validation" begin

        # Test 1: Only Velocity Jump (Constant rho and p)
        # Analytical: (rho_avg * du, 0, gamma * p_avg * du)
        uL = (1.0, 0.0, 1.0)
        uR = (1.0, 1.0, 1.0)
        num_res = path_integral(eq, uL, uR)
        ana_res = (1.0, 0.0, 1.4)
        @test all(isapprox.(num_res, ana_res, atol=tol))
        println("Test 1 (du only): Passed. Result: $num_res")

        # Test 2: Only Pressure Jump (Constant rho and u)
        # Analytical: (0, dp/rho, 0)
        uL = (1.0, 0.0, 1.0)
        uR = (1.0, 0.0, 2.0)
        num_res = path_integral(eq, uL, uR)
        ana_res = (0.0, 1.0, 0.0)
        @test all(isapprox.(num_res, ana_res, atol=tol))
        println("Test 2 (dp only): Passed. Result: $num_res")

        # Test 3: Only Density Jump
        # Analytical: (0, 0, 0) because du=0 and dp=0
        uL = (1.0, 0.0, 1.0)
        uR = (2.0, 0.0, 1.0)
        num_res = path_integral(eq, uL, uR)
        ana_res = (0.0, 0.0, 0.0)
        @test all(isapprox.(num_res, ana_res, atol=tol))
        println("Test 3 (drho only): Passed. Result: $num_res")

        # Test 4: Full Jump (Similar to a shock tube interface)
        # Analytical components derived from logarithmic integration
        rhoL, uL_val, pL = 1.0, 0.0, 1.0
        rhoR, uR_val, pR = 0.5, 0.0, 0.1
        uL = (rhoL, uL_val, pL)
        uR = (rhoR, uR_val, pR)
        
        # Analytic Comp 2: (dp/drho) * ln(rhoR/rhoL)
        dp = pR - pL
        drho = rhoR - rhoL
        ana_v2 = (dp / drho) * log(rhoR / rhoL)
        
        num_res = path_integral(eq, uL, uR)
        ana_res = (0.0, ana_v2, 0.0)
        
        # We use a slightly larger tolerance here as 1/rho is not a polynomial
        @test all(isapprox.(num_res, ana_res, atol=1e-6))
        println("Test 4 (Full jump): Passed. Numerical: $(num_res[2]), Analytical: $ana_v2")
    end
end

# To run:
#pde = LEuler1D(path = LinePath{3}())
#run_integration_tests(pde)

end  # module 