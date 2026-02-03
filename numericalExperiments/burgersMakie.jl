using Meshfree4ScalarEq
using IPlotPDESols

function grid_velocity(pos::Real, rho::Real, params::Tuple)
    return 1/4
end

# Example SimulationConfig for Burgers
sim_config_burgers = SimulationConfig(
    runScalarSimulation,
    ParamDict(
        "tmax" => 10., "N" => 200, "xmin" => -5.0, "xmax" => 5.0,
        "dt" => .01, "snapshots" => 25, "interp_alpha" => 1.0,
        "interp_range" => 3.5, "remove_ghosts" => false,
        "init_func" => "riemann",
        #"init_params" => (1., 0., 1.),
        #"init_params" => (0.0, 1.0, -2., 2.),#(0., 1., -2.,2.),#(0., 1., -4., -2.), #(0.,1.,-2,2.),
        "init_params" => (1., 0., -2.),
        #"init_params" => (2., 0., -2., .1),
        "randomness_factor" => 0.,
        "SEED" => 10, "sim_function" => (:const, "runScalarSimulation"),
        "bc" => :fixed_dirichlet, "weight_function" => "exponential",
        "order" => 1, "PDE" => "testU3", "PDE_params" => 1., "merge_factor" => .3,
        "grid_mover" => "physical", # "grid_mover_func" => grid_velocity, "grid_mover_params" => Tuple([]),
    ),

    MethodDict(
        "RK2Upwind" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "Upwind",
            "order" => 1,
            "main_flux" => "Rusanov",
            "MOOD" => "none",
        ),
        "RK2Central" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "Central",
            "order" => 2,
            "main_flux" => "Upwind",
            "MOOD" => "none",
        ),
        "RK2MUSCL2Smooth" => ParamDict(
            "timestepper" => "RalstonRK2SmoothSwitch",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U2",
            "switch_tol" => 1e-8,
            "delta_relax" => 0.,
            "order" => 2
        ),
        "RK2MUSCL2(VK)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 2,
            "limiter" => "VK",
            "MOOD" => "none",
        ),
        "RK2MUSCL2(superbee)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 2,
            "limiter" => "superbee",
            "MOOD" => "none",
        ),
        "RK2MUSCL2(minmod)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 2,
            "limiter" => "minmod",
            "MOOD" => "none",
        ),
        "RK2MUSCL2(NoLimiter)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 2,
            "limiter" => "none",
            "MOOD" => "none",
        ),
        "RK2MUSCL2MOOD(U2)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U2",
            "delta_relax" => 0.,
            "order" => 2,
        ),
        "RK2MUSCL2MOOD" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U2",
            "delta_relax" => 0.,
            "order" => 2,
        ),
        "RK2MUSCL2MOOD2" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD2" => "U2",
            "delta_relax" => 0.,
            "order" => 2,
        ),
        "RK2MUSCL2OnlyMOOD2" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD2" => "only",
            "delta_relax" => 0.,
            "order" => 2,
        ),
        "RK2MUSCL2MOOD(U2Relax)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U2",
            "delta_relax" => 1.,
            "order" => 2,
        ),
        "RK2MUSCL2MOOD(LoubertU2)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "LoubertU2",
            "delta_relax" => 1.,
            "order" => 2,
        ),
        "RK2MUSCL2MOOD(U1)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U1",
            "delta_relax" => 0.,
            "order" => 2,
        ),
        "RK2MUSCL2MOOD(U1Relax)" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U1",
            "delta_relax" => 5.,
            "order" => 2,
        ),
        "ARS233MUSCL2(VK)" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "limiter" => "VK",
            "order" => 2,
            "save_relax" => false,
            "relax_velocities" => [[1.,-1.]],
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "none"
        ),
        "ARS233MUSCL2(superbee)" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "limiter" => "superbee",
            "order" => 2,
            "save_relax" => false,
            "relax_velocities" => [[1.,-1.]],
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "none"
        ),
        "ARS233MUSCL2(minmod)" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "limiter" => "minmod",
            "order" => 2,
            "save_relax" => false,
            "relax_velocities" => [[1.,-1.]],
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "none"
        ),
        "ARS233WENO" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "WENO",
            "main_flux" => "Rusanov",
            "order" => 2,
            "save_relax" => false,
            "relax_velocities" => [[1.,-1.]],
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "none"
        ),
        "RK2WENO" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "WENO",
            "main_flux" => "Rusanov",
            "order" => 2,
            "MOOD" => "none"
        ),
        "IMEXRK2MUSCL2" => ParamDict(
            "timestepper" => "IMEXRalstonRK2",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 2,
            "save_relax" => false,
            "relax_velocities" => (1.,-1.),
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "none"
        ),
        "SimpleSplittingMUSCL2Limiter" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "limiter" => "superbee",
            "order" => 2,
            "save_relax" => false,
            "relax_velocities" => (1.,-1.),
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "none"
        ),
        "SSMUSCL2MOOD" => ParamDict(
            "timestepper" => "SimpleSplitting",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 2,
            "save_relax" => false,
            "relax_velocities" => [[1.,-1.]],
            "fallback_gradient" => "Upwind",
            "fallback_flux" => "Rusanov",
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "U2", "delta_relax" => 0.
        ),
        "RK2MUSCL2" => ParamDict(
            "timestepper" => "RalstonRK2",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "MOOD" => "none",
            "order" => 2,
        ),
        "RK4MUSCL2" => ParamDict(
            "timestepper" => "RK4",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "MOOD" => "none",
            "order" => 2,
        ),
        "RK4MUSCL3" => ParamDict(
            "timestepper" => "RK4",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "MOOD" => "none",
            "order" => 3,
        ),
        "RK4MUSCL4" => ParamDict(
            "timestepper" => "RK4",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "MOOD" => "none",
            "order" => 4,
        ),
        "RK4MUSCL5" => ParamDict(
            "timestepper" => "RK4",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 5,
            "MOOD" => "none",
        ),
        "RK4MUSCL2MOOD" => ParamDict(
            "timestepper" => "RK4",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U2",
            "delta_relax" => 0.,
            "order" => 2,
        ),
        "RK4MUSCL3MOOD" => ParamDict(
            "timestepper" => "RK4",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U2",
            "delta_relax" => 0.,
            "order" => 3,
        ),
        "RK4MUSCL4MOOD" => ParamDict(
            "timestepper" => "RK4",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U2",
            "delta_relax" => 0.,
            "order" => 4,
        ),
        "RK4MUSCL5MOOD" => ParamDict(
            "timestepper" => "RK4",
            "main_gradient" => "MUSCL",
            "fallback_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "fallback_flux" => "Rusanov",
            "MOOD" => "U2",
            "delta_relax" => 0.,
            "order" => 5
        ),
        "RK4MUSCL2Limiter" => ParamDict(
            "timestepper" => "RK4",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "MOOD" => "none",
            "order" => 2, "limiter" => "VK"
        ),
        "RK4MUSCL3Limiter" => ParamDict(
            "timestepper" => "RK4",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "MOOD" => "none",
            "order" => 3, "limiter" => "VK"
        ),
        "RK4MUSCL4Limiter" => ParamDict(
            "timestepper" => "RK4",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "MOOD" => "none",
            "order" => 4, "limiter" => "VK"
        ),
        "RK4MUSCL5Limiter" => ParamDict(
            "timestepper" => "RK4",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 5,
            "MOOD" => "none", "limiter" => "VK"
        ),
        "EulerUpwind" => ParamDict(
            "timestepper" => "EulerUpwind",
            "main_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "order" => 1,),
        "EulerUpwindNoMovement" => ParamDict(
            "timestepper" => "EulerUpwind",
            "main_gradient" => "Upwind",
            "main_flux" => "Rusanov",
            "order" => 1, "grid_mover" => "none",
        ),
        "Analytical Solution" => ParamDict(
            "N" => (:const, 1000), 
            "ignore" => ["interp_alpha", "randomness_factor", "SEED", "order"]
             # No randomness_factor needed when regular=true
        ),
        "LLF(uniform grid)" => ParamDict(
            "randomness_factor" => (:const,0.),
            "main_flux" => "Rusanov",
            "order" => 1,
            "timestepper" => "Classic",
            "ignore" => ["interp_alpha", "randomness_factor", "SEED"]
        ),
                "ARS233MUSCL5" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 5,
            "save_relax" => false,
            "relax_velocities" => [[1.,-1.]],
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "none",
        ),
        "ARS233MUSCL5MOOD" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 5,
            "save_relax" => false,
            "relax_velocities" => [[1.,-1.]],
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "U2", "delta_relax" => 0.,
            "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov",
        ),
        "ARS233Upwind" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "Upwind",
            "main_flux" => "Upwind",
            "order" => 1,
            "save_relax" => false,
            "relax_velocities" => [[2.,-2.]],
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "none",
        ),
        "ARS233WENO" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "WENO",
            "main_flux" => "Rusanov",
            "order" => 2,
            "save_relax" => false,
            "relax_velocities" => [[1.,-1.]],
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "none",
        ),
        # "PRSSP3MUSCL5" => ParamDict(
        #     "timestepper" => "PRSSP3",
        #     "main_gradient" => "MUSCL",
        #     "main_flux" => "Rusanov",
        #     "order" => 5,
        #     "relax_method" => true,
        #     "relax_velocities" => (1.,-1.),
        #     "relax_epsilon" => 10. ^ -8,
        #     "MOOD" => "none",
        # ),
        "ARS233MUSCL2" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 2,
            "save_relax" => false,
            "relax_velocities" => [[1.,-1.]],
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "none",
        ),
        "ARS233MUSCL3" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 3,
            "save_relax" => false,
            "relax_velocities" => [[1.,-1.]],
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "none", "save_relax" => false
        ),
        "ARS233MUSCL4" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 4,
            "save_relax" => false,
            "relax_velocities" => [[1.,-1.]],
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "none", "save_relax" => false
        ),
        "ARS233MUSCL5" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 5,
            "save_relax" => false,
            "relax_velocities" => [[1.,-1.]],
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "none", "save_relax" => false
        ),
        "LWMOOD(uniform grid)" => ParamDict(
            "timestepper" => "LW",
            "order" => 2,
            "MOOD" => "U1", "delta_relax" => 0.,
            "ignore" => ["interp_alpha", "randomness_factor", "SEED"]
        ),
        "LW(uniform grid)" => ParamDict(
            "timestepper" => "LW",
            "order" => 2,
            "MOOD" => "none",
            "ignore" => ["interp_alpha", "randomness_factor", "SEED"]
        ),
        "LF(uniform grid)" => ParamDict(
            "timestepper" => "LF",
            "order" => 1,
            "MOOD" => "none",
            "ignore" => ["interp_alpha", "randomness_factor", "SEED"]
        ),
        "ARS233MUSCL2MOOD" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 2,
            "save_relax" => false,
            "relax_velocities" => [[1.,-1.]],
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "U2", "delta_relax" => 0.,
            "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov",
        ),
        "ARS233MUSCL2MOOD2" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 2,
            "save_relax" => false,
            "relax_velocities" => [[1.,-1.]],
            "relax_epsilon" => 10. ^ -8,
            "MOOD2" => "U2", "delta_relax" => 0.,
            #"fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov",
        ),
        "ARS233MUSCL3MOOD" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 3,
            "save_relax" => false,
            "relax_velocities" => [[1.,-1.]],
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "U2", "delta_relax" => 0.,
            "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov",
        ),
        "ARS233MUSCL4MOOD" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 4,
            "save_relax" => false,
            "relax_velocities" => [[1.,-1.]],
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "U2", "delta_relax" => 0.,
            "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov",
        ),
        "ARS233MUSCL5MOOD" => ParamDict(
            "timestepper" => "ARS233",
            "main_gradient" => "MUSCL",
            "main_flux" => "Rusanov",
            "order" => 5,
            "save_relax" => false,
            "relax_velocities" => [[1.,-1.]],
            "relax_epsilon" => 10. ^ -8,
            "MOOD" => "U2", "delta_relax" => 0.,
            "fallback_gradient" => "Upwind", "fallback_flux" => "Rusanov",
        ),
        # "PRSSP3MUSCL2" => ParamDict(
        #     "timestepper" => "PRSSP3",
        #     "main_gradient" => "MUSCL",
        #     "main_flux" => "Rusanov",
        #     "MOOD" => "none",
        #     "order" => 2,
        #     "relax_method" => true,
        #     "relax_velocities" => (1.,-1.),
        #     "relax_epsilon" => 10. ^ -8,
        # )

    ),
    #["ARS233MUSCL2","ARS233MUSCL3","ARS233MUSCL4","ARS233MUSCL5"],
    #["RK4MUSCL2","RK4MUSCL3","RK4MUSCL4","RK4MUSCL5"]
    #["RK4MUSCL2MOOD","RK4MUSCL3MOOD","RK4MUSCL4MOOD","RK4MUSCL5MOOD"]
    #["ARS233MUSCL2MOOD","ARS233MUSCL3MOOD","ARS233MUSCL4MOOD","ARS233MUSCL5MOOD"]
    #"RK4MUSCL5"
    #["RK4MUSCL2Limiter","RK4MUSCL3Limiter","RK4MUSCL4Limiter","RK4MUSCL5Limiter"]
    #["Analytical Solution", "ARS233WENO", "ARS233MUSCL2MOOD", "RK2MUSCL2(Superbee)", "RK2MUSCL2(VKLimiter)"]
    #["SSMUSCL2MOOD", "ARS233MUSCL2", "ARS233MUSCL2MOOD", "ARS233MUSCL2MOOD2"]#,"ARS233MUSCL2","ARS233Upwind"]
    #["Analytical Solution", "RK2MUSCL2MOOD2","RK2MUSCL2MOOD", "RK2MUSCL2", "RK2MUSCL2OnlyMOOD2"]
    #["RK2MUSCL2MOOD","RK2MUSCL2MOOD2", "RK2MUSCL2(superbee)", "ARS233MUSCL2MOOD", "ARS233MUSCL2Limiter","Analytic Solution", "ARS233Upwind"]
    #["ARS233MUSCL2", "ARS233MUSCL2MOOD", "ARS233MUSCL2Limiter", "EulerUpwind","Analytic Solution"]#,"LW", "EulerUpwind"]
    #["LWMOOD","RK2MUSCL2", "LW","ARS233MUSCL2MOOD", "Analytic Solution", "RK2MUSCL2MOOD(U1)", "RK2MUSCL2MOOD(U2)", "RK4MUSCL5MOOD"]
    #["LWMOOD","RK2MUSCL2", "LW","ARS233MUSCL2MOOD", "Analytic Solution", "RK2MUSCL2MOOD(U1)", "RK2MUSCL2MOOD(U2)", "RK2MUSCL2MOOD(U2Relax)", "RK2MUSCL2MOOD(U1Relax)", "RK4MUSCL5MOOD"]#["RK2MUSCL2MOOD", "RK2MUSCL2", "RK4MUSCL5MOOD", "Analytic Solution"] #, "Relax Method 2", "Relax Method 3rd order","Classic","SlopeLimiter","SmoothSwitching","Regular MOOD", "OnlyFallback"]
    #["ARS233MUSCL2MOOD","ARS233MUSCL5MOOD","LWMOOD(uniform grid)","RK2MUSCL2MOOD(U1)","RK2MUSCL2MOOD(U2)","RK4MUSCL5MOOD"]
    #["EulerUpwind","Analytical Solution","RK2MUSCL2(VKLimiter)","RK2MUSCL2(superbee)","RK2MUSCL2", "RK2MUSCL2MOOD", "ARS233MUSCL2MOOD"]
    #["LW(uniform grid)", "ARS233MUSCL2", "ARS233MUSCL5", "EulerUpwind", "LLF(uniform grid)", "RK2MUSCL2", "RK4MUSCL5"]
    #["RK2MUSCL2Smooth", "Analytical Solution"]
    ["EulerUpwind", "Analytical Solution"]
    #["EulerUpwind","Analytical Solution","RK2MUSCL2(VK)","RK2MUSCL2"]
    #["Analytical Solution", "ARS233MUSCL2","ARS233MUSCL2MOOD", "ARS233MUSCL2(VK)", "ARS233Upwind"]
    #"ARS233Upwind"
    #"RK2WENO"
    #["RK2MUSCL2", "RK2Upwind", "RK2MUSCL2MOOD", "RK2MUSCL2(minmod)", "RK2MUSCL2(VK)"]
);
scene_options = Dict{String, Any}()
scene_options = Dict{String, Any}("t" => 10.,"component" => 1, "x_key" => "SEED", "y_key" => "l2error")
# Pass this config to your IPlotPDESols functions
show1DSolutionFig(sim_config_burgers; calc_stats = false, ui_options = :default, scene_options = scene_options);
#showDynamicDependence(sim_config_burgers; ui_options = :publication, scene_options = scene_options)
#showConvergencePlot(sim_config_burgers, "merge_factor", .1:.05:.9 ; calc_stats = false, force_int_param = false)
#showConvergencePlot(sim_config_burgers, "PDE_params", .0:.025:1. ; calc_stats = false, force_int_param = false)
#show2DConvergencePlot(sim_config_burgers, "PDE_params", .1:.01:1. ,"merge_factor", .05:.01:.9; calc_stats = false)
#calculateConvergenceData(sim_config_burgers, "N", 10. .^(1:.25:2); calc_stats = false, force_int_param = true)
#showConvergencePlot(sim_config_burgers, "N", 10. .^(1.5:.125:4.); calc_stats = false, force_int_param = true, initial_calc = true, ui_options = :publication, scene_options = scene_options)
#showConvergencePlot(sim_config_burgers, "delta_relax", (0.:10^-51:10^-50); force_int_param = false, initial_calc = true, ui_options = :publication)
#showConvergencePlot(sim_config_burgers, "switch_tol", 10. .^(-5:.1:-2); force_int_param = false, initial_calc = true, ui_options = :publication)
#showConvergencePlot(sim_config_burgers, "SEED", range(1,10000,500); force_int_param = true, initial_calc = true, ui_options = :publication);
#showConvergencePlot(sim_config_burgers, "relax_velocities", 10. .^(-3:.25:0.); force_int_param = false, initial_calc = true, ui_options = :publication);

# using IPlotPDESols
#test = create_sim_config_from_csv("figures/LA_box_IMEX_direct_comp_params.csv","none")
#show1DSolutionFig(test; calc_stats = false, ui_options = :publication);