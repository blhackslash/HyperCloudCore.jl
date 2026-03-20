using Meshfree4ScalarEq
using IPlotPDESols
using InteractiveUtils

# 2D simulation using your `runScalar2DSim` function.
function main()
    params = ParamDict(
        # --- Shared Parameters ---
        "tmax" => 5.0,
        "Nx" => 250,
        "Ny" => 250,
        "xmin" => -5.0,
        "xmax" => 5.0,
        "ymin" => -5.0,
        "ymax" => 5.0,
        "CFL" => 0.4,
        #"dt" => .005,
        "snapshots" => 20,
        "interp_alpha" => 1.0,
        "interp_range" => 3.5,
        "init_func" => "riemann", # Using the unified struct
        "init_params" => (1.,0.,(-2.,-2.),(1.,1.)),
        #"init_params" => ((1.,1.),(0.,0.),(-2.,-2.),(1.,1.)),
        #"init_params" => (1.0, (0.0, 0.0), 1.5), # (amplitude, (centerX, centerY), width)
        #"init_params" => (0.,1.,-2.,2.,-2.,2.),
        "randomness_factor" => (0.2, 0.2), # (x_rand_factor, y_rand_factor)
        "SEED_value" => 42,
        "PDE" => "linear",
        "PDE_params" => ((.5,),(1.,)), # 2D velocity vector (vx, vy)
        "bc" => :outflow,
        "sim_function" => "runSimulation",
        "weight_function" => "exponential",
        #"limiter" => "VK",

        # --- Method-Specific Parameters for "RK4-MUSCL2-2D" ---
        "timestepper" => "RK3",
        "main_gradient" => "Upwind",
        "order" => 1,
        "upwind_alg_2d" => "Classic",
        "main_flux" => "Rusanov",
        "remove_ghosts" => true
        #"relax_velocities" => [[ (2.0, 0.), (-2., 0.), (0., -2.), (0., 2.) ]], "relax_epsilon" => 1e-6, "save_relax" => false,
        #"MOOD" => "U2", "delta_relax" => 0., "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov",
        #"mood" => "none", # No MOOD for this run
    )

    # params = ParamDict(
    #     # --- Shared Parameters ---
    #     "tmax" => 5.,
    #     "N" => 200,
        
    #     "xmin" => -5.0,
    #     "xmax" => 5.0,
    #     #"CFL" => .1,
    #     "dt" => .01,
    #     "snapshots" => 20,
    #     "interp_alpha" => 1.0,
    #     "interp_range" => 3.5,
    #     "init_func" => "riemann", # Using the unified struct
    #     "init_params" => ((1.,1.),(0.,0.),-2., 1.),
    #     #"init_params" => (1.0, 0., 1.5), # (amplitude, (centerX, centerY), width)
    #     #"init_params" => (0.,1.,-2.,2.),
    #     "randomness_factor" => 0.0, # (x_rand_factor, y_rand_factor)
    #     "SEED_value" => 42,
    #     "PDE" => "linear", "PDE_params" => ((1.,.5),), #"merge_factor" => .3,
    #     "weight_function" => "exponential",
    #     "grid_mover" => "none",
    #     "bc" => :outflow,
    #     "sim_function" => "runSimulation",

    #     # --- Method-Specific Parameters for "RK4-MUSCL2-2D" ---
    #     "timestepper" => "RalstonRK2",
    #     #"relax_velocities" => [[1.,-1.]],  "relax_epsilon" => 1e-6, "save_relax" => false,
    #     "main_gradient" => "Upwind",
    #     "order" =>1,
    #     "main_flux" => "Rusanov",
    #     "remove_ghosts" => false
    #     #"MOOD" => "U2", "delta_relax" => 0., "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov",
    #     #"switch_tol" => 5.
    #     #"limiter" => "minmod",
    #     #"MOOD" => "U1", "delta_relax" => 0. # No MOOD for this run
    # )

    # --- How to use this for testing ---
    #
    # 1. Make sure all your project modules and the `runScalar2DSim.jl` file are loaded.
    #
    # 2. You can then call your function directly:
    #
    # 1. Configure the profiler to sample ALL threads
    #    We also give it a larger buffer (n) and a reasonable delay
    sim_data = nothing
    sim_data = runSimulation(params);
    @profview runSimulation(params)
    # 3. `sim_data` will now hold the results (a SimData2D object), which you can inspect.
    #
    test_config = SimulationConfig(
        runSimulation,
        params,
        MethodDict(
            "Test(MOOD)" => ParamDict("MOOD" => "U2", "delta_relax" => 0., "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov",),
            "Test" => ParamDict(),
            "TestMUSCL(SmSw)" => ParamDict("timestepper" => "RalstonRK2SmoothSwitch","MOOD" => "U2", "delta_relax" => 0., "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", "switch_tol" => 5.),
            #"TestMUSCLRelax" => ParamDict("timestepper" => "PRSSP3", "relax_velocities" => [-1.,1.], "save_relax" => false, "relax_epsilon" => 1e-6),
        ),
        #"all"
        ["Test"]

    );
    return sim_data, test_config
end

sim_data, test_config = main()

#show1DSolutionFig(test_config)
#scene_options = Dict{String, Any}("line_vector" => (1.,0.), "deviation" => 2)
#show2DCutFig(test_config;scene_options = scene_options)
#show2DSolutionFig(test_config)