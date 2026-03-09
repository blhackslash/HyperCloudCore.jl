using Meshfree4ScalarEq
using IPlotPDESols

# --- Example SimulationConfig for 1D Euler System ---
euler_smooth_params = (
    (amp=0.1, mean=0.0, width=0.5, off=1.0),
    (amp=0.2, mean=0.0, width=0.5, off=0.5),
    (amp=0.1, mean=0.0, width=0.5, off=1.0)
)

sod_euler_params = ( # Sod shock tube for 1D Euler
    (1.0, 0.0, 1.6665),    # rho, u, p
    (0.125, 0.0, 0.16666), 
    0.0 
    # Note: Your plotting range and tmax should be suitable for Sod's problem evolution.
    # Typical Sod domain [-0.5, 0.5], tmax ~ 0.2
)
sod_euler_params = ( # Sod shock tube for 1D Euler
    (1.0, 0.0, 1.),    # rho, u, p
    (0.125, 0.0, 0.1), 
    0.0 
    # Note: Your plotting range and tmax should be suitable for Sod's problem evolution.
    # Typical Sod domain [-0.5, 0.5], tmax ~ 0.2
)
# sod_euler_params = (
#     (0.001, 0.0, 1.6665e-3),       # rho_L, u_L, p_L
#     (1/8 * 1e-3, 0.0, 1.6666e-4),  # rho_R, u_R, p_R
#     0.0                            # Discontinuity position x0
# )
SEED_value = 10
as1 = collect(0.05:0.05:3.)
as = [as1; -as1]
function main()


    sim_config_euler1d_system = SimulationConfig(
        runSystemSimulation,
        ParamDict(
            "tmax" => 0.2, "N" => 600, "bc" => :outflow,
            "xmin" => -0.5, "xmax" => .5, 
            "dt" => 0.001, "snapshots" => 21, 
            "interp_alpha" => 1.0, "interp_range" => 3.5, # Factor for dx
            #"init_func" => "eulerShockTube",
            "init_func" => "riemann", 
            "PDE" => "leuler1d", "sim_function" => "runSystemSimulation",
            #"init_params" => euler_smooth_params, 
            "init_params" => sod_euler_params, 
            "randomness_factor" => 0., "grid_mover" => "none", "merge_factor" => .3,
            "SEED" => SEED_value, "save_relax" => false, "weight_function" => "exponential",
            #"relax_velocities" => [as, as, as]
            "relax_velocities" => [ [2.0, -2.0], [2.0, -2.0], [2.0, -2.0] ], # Pairs for rho, m, E kinetic components
        ),
        MethodDict( 
            "ARS222MUSCL2(minmod)" => ParamDict(
                "timestepper" => "ARS222",
                "main_gradient" => "MUSCL", "order" => 2, # MUSCLlimited recon order is 1. this order param is for general MUSCL
                "main_flux" => "Rusanov", "limiter" => "minmod",
                "MOOD" => "none",
                "relax_epsilon" => 1e-6
            ),
            "ARS222MUSCL2(VK)" => ParamDict(
                "timestepper" => "ARS222",
                "main_gradient" => "MUSCL", "order" => 2, # MUSCLlimited recon order is 1. this order param is for general MUSCL
                "main_flux" => "Rusanov", "limiter" => "VK",
                "MOOD" => "none",
                "relax_epsilon" => 1e-6
            ),
            "SSMUSCL2(minmod)" => ParamDict(
                "timestepper" => "SimpleSplitting",
                "main_gradient" => "MUSCL", "order" => 2, # MUSCLlimited recon order is 1. this order param is for general MUSCL
                "main_flux" => "Rusanov", "limiter" => "minmod",
                "MOOD" => "none", 
                "relax_epsilon" => 1e-6
            ),
            "SSMUSCL2MOOD" => ParamDict(
                "timestepper" => "SimpleSplitting",
                "main_gradient" => "MUSCL", "order" => 2, # MUSCLlimited recon order is 1. this order param is for general MUSCL
                "main_flux" => "Rusanov", #"limiter" => "minmod",
                "MOOD" => "U1", "delta_relax" => 0., #"MOOD" => "none",
                "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov",
                "relax_epsilon" => 1e-6 
            ),
            "SSMUSCL4MOOD" => ParamDict(
                "timestepper" => "SimpleSplitting",
                "main_gradient" => "MUSCL", "order" => 4, # MUSCLlimited recon order is 1. this order param is for general MUSCL
                "main_flux" => "Rusanov", #"limiter" => "minmod",
                "MOOD" => "U1", "delta_relax" => 0.,
                "relax_epsilon" => 1e-6
            ),
            "ARS222WENO2" => ParamDict(
                "timestepper" => "ARS222",
                "main_gradient" => "WENO", "order" => 2,
                "MOOD" => "none",
                "relax_epsilon" => 1e-6
            ),
            "ARS222MUSCL2MOOD" => ParamDict(
                "timestepper" => "ARS222",
                "main_flux" => "Rusanov",
                "main_gradient" => "MUSCL", "order" => 2, # MUSCL(1) for 2nd order spatial
                "MOOD" => "U1", "delta_relax" => 0., #"MOOD" => "none",
                        "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
                "relax_epsilon" => 1e-10,
                #"relax_velocities" => [ [10.0, -10.0,], [10.0, -10.0], [10.0, -10.0] ]
            ),
            "ARS222MUSCL2MOOD2" => ParamDict(
                "timestepper" => "ARS222",
                "main_flux" => "Rusanov",
                "main_gradient" => "MUSCL", "order" => 2, # MUSCL(1) for 2nd order spatial
                "MOOD2" => "U1", "delta_relax" => 0., "MOOD" => "none",
                        "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
                "relax_epsilon" => 1e-10,
                #"relax_velocities" => [ [10.0, -10.0,], [10.0, -10.0], [10.0, -10.0] ]
            ),
            "SSP3MUSCL2MOOD" => ParamDict(
                "timestepper" => "PRSSP3",
                "main_flux" => "Rusanov",
                "main_gradient" => "MUSCL", "order" => 2, # MUSCL(1) for 2nd order spatial
                "MOOD" => "U1", "delta_relax" => 0., #"MOOD" => "none",
                        "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
                "relax_epsilon" => 1e-10,
                #"relax_velocities" => [ [10.0, -10.0,], [10.0, -10.0], [10.0, -10.0] ]
            ),
            "ARS222MUSCL2" => ParamDict(
                "timestepper" => "ARS222",
                "main_flux" => "Rusanov",
                "main_gradient" => "MUSCL", "order" => 2, # MUSCL(1) for 2nd order spatial
                "MOOD" => "none", 
                        "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
                "relax_epsilon" => 1e-6
            ),
            "ARS233MUSCL2" => ParamDict(
                "timestepper" => "ARS233",
                "main_flux" => "Rusanov",
                "main_gradient" => "MUSCL", "order" => 2, # MUSCL(1) for 2nd order spatial
                "MOOD" => "none", 
                        "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
                "relax_epsilon" => 1e-6
            ),
            "ARS233MUSCL3" => ParamDict(
                "timestepper" => "ARS233",
                "main_flux" => "Rusanov",
                "main_gradient" => "MUSCL", "order" => 3, # MUSCL(1) for 2nd order spatial
                "MOOD" => "none", 
                        "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
                "relax_epsilon" => 1e-6
            ),
            "ARS233MUSCL4" => ParamDict(
                "timestepper" => "ARS233",
                "main_flux" => "Rusanov",
                "main_gradient" => "MUSCL", "order" => 4, # MUSCL(1) for 2nd order spatial
                "MOOD" => "none", 
                        "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
                "relax_epsilon" => 1e-6
            ),
            "SSP2MUSCL2MOOD" => ParamDict(
                "timestepper" => "SSP2",
                "main_flux" => "Rusanov",
                        "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
                "main_gradient" => "MUSCL", "order" => 2, # MUSCL(1) for 2nd order spatial
                "MOOD" => "U2", "delta_relax" => 0., 
                "relax_epsilon" => 1e-6
            ),
            "ARS222Upwind" => ParamDict(
                "timestepper" => "ARS222",
                "main_flux" => "Rusanov",
                "main_gradient" => "Upwind", "order" => 1, # MUSCL(1) for 2nd order spatial
                "MOOD" => "none",
                "relax_epsilon" => 1e-6
            ),
            "ARS222Upwind(fixedGrid)" => ParamDict(
                "timestepper" => "ARS222",
                "main_flux" => "Rusanov",
                "main_gradient" => "Upwind", "order" => 1, # MUSCL(1) for 2nd order spatial
                "MOOD" => "none",
                "randomness_factor" => 0.,
                "interp_range" => 1.5,
                "relax_epsilon" => 1e-6
            ),
            "ARS233MUSCL5MOOD" => ParamDict(
                "timestepper" => "ARS233",
                        "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
                "main_flux" => "Rusanov",
                "main_gradient" => "MUSCL", "order" => 5, # MUSCL(1) for 2nd order spatial
                "MOOD" => "U2", "delta_relax" => 0., # More aggressive MOOD
                "relax_epsilon" => 1e-6
            ),
            "ARS222MUSCL5MOOD" => ParamDict(
                "timestepper" => "ARS222",
                        "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
                "main_flux" => "Rusanov",
                "main_gradient" => "MUSCL", "order" => 5, # MUSCL(1) for 2nd order spatial
                "MOOD" => "U2", "delta_relax" => 0., # More aggressive MOOD
                "relax_epsilon" => 1e-6
            ),
            "SSP3MUSCL5MOOD" => ParamDict(
                "timestepper" => "PRSSP3",
                        "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov", # Fallback for MOOD inside ARS2IMEX
                "main_flux" => "Rusanov",
                "main_gradient" => "MUSCL", "order" => 5, # MUSCL(1) for 2nd order spatial
                "MOOD" => "U2", "delta_relax" => 0., # More aggressive MOOD
                "relax_epsilon" => 1e-6
            ),
            "Analytical Solution" => ParamDict(
                "timestepper" => "Analytic",
                "ignore" => ["interp_range", "interp_alpha","weight_function", "randomness_factor", "SEED", "order", "relax_velocities"]
                # No randomness_factor needed when regular=true
            ),
            "Reference" => ParamDict(
                "timestepper" => "SimpleSplitting",
                "main_flux" => "Rusanov",
                "MOOD" => "none", 
                "N" => (:const, 30000),
                "relax_epsilon" => 1e-6,
                "ignore" => ["interp_range", "interp_alpha", "randomness_factor", "SEED", "order"]
            )
        ),
        #"Analytical Solution"
        #"ARS222Upwind"
        ["ARS222MUSCL2MOOD","Analytical Solution","ARS222MUSCL2","ARS222Upwind"]
        #["ARS233MUSCL2","ARS233MUSCL3","ARS233MUSCL4"]
        #["ARS222MUSCL2MOOD2","ARS222MUSCL2(VK)", "SSMUSCL4MOOD","SSP3MUSCL2MOOD", "ARS222Upwind","Analytical Solution", "SSMUSCL2MOOD","ARS222MUSCL2", "ARS222MUSCL2(minmod)", "ARS222MUSCL2MOOD"]
        #["ARS222Upwind(fixedGrid)", "ARS222MUSCL2limiter", "ARS233MUSCL5MOOD", "ARS222MUSCL2MOOD", "ARS222MUSCL5MOOD","ARS222MUSCL2"]
    )

    params =
        ParamDict(
            "tmax" => 0.2, "N" => 50, "bc" => :fixed_dirichlet,
            "xmin" => -0.5, "xmax" => .5, 
            "dt" => 0.01, "snapshots" => 21, 
            "interp_alpha" => 1.0, "interp_range" => 3.5, # Factor for dx
            #"init_func" => "eulerShockTube",
            "init_func" => "riemann", 
            "PDE" => "euler1d", "sim_function" => "runSystemSimulation",
            #"init_params" => euler_smooth_params, 
            "init_params" => sod_euler_params, 
            "randomness_factor" => 0., "grid_mover" => "none", "merge_factor" => .5,
            "SEED" => SEED_value, "save_relax" => false, "weight_function" => "exponential",
            #"relax_velocities" => [as, as, as]
            "relax_velocities" => [ [2.0, -2.0], [2.0, -2.0], [2.0, -2.0] ], # Pairs for rho, m, E kinetic components
            "timestepper" => "ARS222",
            "main_flux" => "Rusanov",
            "main_gradient" => "Upwind", "order" => 1, # MUSCL(1) for 2nd order spatial
            "MOOD" => "none",
            "relax_epsilon" => 1e-6
            #"PDE_params" => (1.0, 1.0) # 2D velocity vector (vx, vy)
        )
    #runSystemSimulation(params)
    @profview runSystemSimulation(params)
    return sim_config_euler1d_system
end

sim_config_euler1d_system = main()
#show1DSolutionFig(sim_config_euler1d_system; ui_options = :publication) 
#showDynamicDependence(sim_config_euler1d_system; calc_stats = true)
#showConvergencePlot(sim_config_euler1d_system, "N", 10. .^(1.:.25:2.5); force_int_param = true, initial_calc = true, ui_options = :default)
#showConvergencePlot(sim_config_euler1d_system, "SEED", 100:50:10000; force_int_param = true, initial_calc = true, ui_options = :default)
# This will require show1DSolutionFig to be adapted to handle SimData1D.u as Vector{Matrix}
# and use the component selector. For now, it will plot the first component (rho_macro).
#sim_config = create_sim_config_from_csv("numericalExperiments/Euler/ShockTubeSolution(uniform)/figures/euler_shocktube_limiter_uniform_params.csv","none")
#show1DSolutionFig(sim_config; ui_options = :publication) 

### Test

