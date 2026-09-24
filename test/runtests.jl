using Test
using HyperCloud
import HyperCloud: flux, velocity, max_eigenvalue, prim2cons, cons2prim
using PDEStudioCore
using StaticArrays

include("InitialConditions.jl")
include("analytical_solution.jl")
include("run_simulation.jl")
# Point PDEStudioCore to look inside this test environment
# so it can dynamically resolve run_simulation and analytical_solution
PDEStudioCore.set_target_module!(@__MODULE__)

@testset "HyperCloud Pipeline Tests" begin
    # Setup isolated temp directories and stats preset
    set_stat_preset!("hyperbolic")
    tmp_dir = mktempdir()
    set_save_path!(tmp_dir)

    # Base parameters shared across all spatial schemes
    shared = Dict(
        :PDE => "linear",
        :PDE_params => ((1.0,),),          # Advective speed c = 1.0
        :mins => (-1.0,),
        :maxs => (1.0,),
        :Ns => (100,),
        :periodic => true,
        :domain => "rectangular",
        :init_func => "gauss",
        :init_params => (
            1.,                        # Amplitude a
            (0.,),                        # position
            .1                         # width
        ),
        :tmax => 2.,
        :timestepper => "RK3",             # Upgraded to RK3 for MUSCL/WENO stability
        :interp_range => 2.5,
        :interp_alpha => 1.0,
        :snapshots => 5,
        :SEED => 42,
        :CFL => 0.4
    )

    # Define the different spatial reconstruction schemes
    methods = Dict(
        :upwind => Dict(
            :main_gradient => "Upwind",
            :main_flux => "Upwind",
            :order => 1
        ),
        :muscl => Dict(
            :main_gradient => "MUSCL",
            :main_flux => "Rusanov",
            :limiter => "minmod",
            :order => 2
        ),
        :weno => Dict(
            :main_gradient => "WENO",
            :main_flux => "Rusanov",
            :order => 2
        )
    )

    # Parameter Sweep for Grid Convergence
    varied = Dict(
        :Ns => [(200,), (400,)]
    )

    # Build the SimulationConfig with all three active methods
    config = SimulationConfig(
        :run_simulation,
        shared,
        methods,
        [:upwind, :muscl, :weno];
        varied_params = varied,
        ref_func_name = :analytical_solution
    )

    # Run the complete pipeline (Generation + Stats)
    run_all_simulations(config; force_overwrite=true, calculate_stats=true)

    # Verify each scheme independently
    for scheme in [:upwind, :muscl, :weno]
        @testset "Scheme: $scheme" begin
            # Construct the exact parameter dictionaries expected on disk
            base_params = merge(shared, methods[scheme])
            params_100 = create_param_dict(base_params..., :Ns => (200,))
            params_200 = create_param_dict(base_params..., :Ns => (400,))

            @test does_sim_data_exist(params_100)
            @test does_sim_data_exist(params_200)

            sim_100 = load_sim_data(params_100)
            sim_200 = load_sim_data(params_200)

            # Ensure output structure is Lagrangian data
            @test sim_100 isa LSimData

            @testset "Statistic Generation" begin
                @test haskey(sim_100.stats, :l1error)
                @test haskey(sim_100.stats, :mass)
                @test length(sim_100.stats[:l1error]) == length(sim_100.t)
            end

            @testset "Mass Conservation" begin
                # Integrated mass across time steps should remain constant
                mass_200 = sim_200.stats[:mass]
                initial_mass = mass_200[1][1]
                @test all(m -> isapprox(m[1], initial_mass; rtol=1e-2), mass_200)
            end

            @testset "Grid Convergence" begin
                # Error should strictly reduce as particle density increases
                err_100 = sim_100.stats[:l1error][end][1]
                err_200 = sim_200.stats[:l1error][end][1]
                @test err_200 < err_100
            end
        end
    end
    @testset "MUSCL Higher-Order Convergence (Orders 2-5)" begin
        # 1. Setup a clean, high-precision environment for strict convergence testing
        shared_muscl = copy(shared)
        shared_muscl[:timestepper] = "RK4" # Prevent time-integration from bottlenecking 4th-order spatial error
        
        # Disable limiters to measure the true asymptotic convergence rate of the reconstruction
        methods_muscl = Dict(
            :muscl2 => Dict(:main_gradient => "MUSCL", :main_flux => "Rusanov", :limiter => "none", :order => 2, :mood_criterion => "none"),
            :muscl3 => Dict(:main_gradient => "MUSCL", :main_flux => "Rusanov", :limiter => "none", :order => 3, :mood_criterion => "none"),
            :muscl4 => Dict(:main_gradient => "MUSCL", :main_flux => "Rusanov", :limiter => "none", :order => 4, :mood_criterion => "none"),
            :muscl5 => Dict(:main_gradient => "MUSCL", :main_flux => "Rusanov", :limiter => "none", :order => 5, :mood_criterion => "none")
        )
        
        config_muscl = SimulationConfig(
            :run_simulation,
            shared_muscl,
            methods_muscl,
            [:muscl2, :muscl3, :muscl4, :muscl5];
            varied_params = create_varied_dict(:Ns => [(200,),(400,)]), # Re-uses your Dict(:Ns => [(100,), (200,)])
            ref_func_name = :analytical_solution
        )
        
        run_all_simulations(config_muscl; force_overwrite=true, calculate_stats=true)
        
        # 2. Extract errors and evaluate Empirical Order of Convergence (EOC)
        errors_100 = Dict{Int, Float64}()
        errors_200 = Dict{Int, Float64}()
        
        for order in 2:5
            scheme_sym = Symbol("muscl$order")
            base_p = merge(shared_muscl, methods_muscl[scheme_sym])
            
            sim_100 = load_sim_data(create_param_dict(base_p..., :Ns => (200,)))
            sim_200 = load_sim_data(create_param_dict(base_p..., :Ns => (400,)))
            
            err_100 = sim_100.stats[:relative_l2error][end][1]
            err_200 = sim_200.stats[:relative_l2error][end][1]
            
            errors_100[order] = err_100
            errors_200[order] = err_200
            
            # Calculate Empirical Order of Convergence: log2(E_coarse / E_fine)
            eoc = log2(err_100 / err_200)
            
            @testset "MUSCL Order $order Convergence" begin
                # Assert error reduces with grid refinement
                @test err_200 < err_100
                
                # Test grouped order behaviors (Odd/Even grouping)
                if order == 2 || order == 3
                    # Expecting ~2nd order convergence
                    @test 1.5 < eoc < 2.5
                elseif order == 4 || order == 5
                    # Expecting ~4th order convergence
                    @test 3.5 < eoc < 5.
                end
            end
        end
        
        @testset "Error Hierarchy" begin
            # Ensure the 4th-order group is strictly more accurate than the 2nd-order group
            @test errors_200[4] < errors_200[2]
            @test errors_200[5] < errors_200[3]
            
            # The odd grouped orders (3 and 5) usually yield similar or slightly better 
            # absolute errors than their even counterparts (2 and 4) at the same resolution.
            @test errors_200[3] <= errors_200[2]
            @test errors_200[5] <= errors_200[4]
        end
    end
    
@testset "MUSCL Higher-Order Convergence (Orders 2-5) with MOOD" begin
        # 1. Setup a clean, high-precision environment for strict convergence testing
        shared_muscl = copy(shared)
        shared_muscl[:timestepper] = "RK4" # Prevent time-integration from bottlenecking 4th-order spatial error
        
        # Disable limiters to measure the true asymptotic convergence rate of the reconstruction
        methods_muscl = Dict(
            :muscl2 => Dict(:main_gradient => "MUSCL", :main_flux => "Rusanov", :limiter => "none", :order => 2, :mood_criterion => "U2"),
            :muscl3 => Dict(:main_gradient => "MUSCL", :main_flux => "Rusanov", :limiter => "none", :order => 3, :mood_criterion => "U2"),
            :muscl4 => Dict(:main_gradient => "MUSCL", :main_flux => "Rusanov", :limiter => "none", :order => 4, :mood_criterion => "U2"),
            :muscl5 => Dict(:main_gradient => "MUSCL", :main_flux => "Rusanov", :limiter => "none", :order => 5, :mood_criterion => "U2")
        )
        
        config_muscl = SimulationConfig(
            :run_simulation,
            shared_muscl,
            methods_muscl,
            [:muscl2, :muscl3, :muscl4, :muscl5];
            varied_params = create_varied_dict(:Ns => [(300,),(600,)]), # Re-uses your Dict(:Ns => [(100,), (200,)])
            ref_func_name = :analytical_solution
        )
        
        run_all_simulations(config_muscl; force_overwrite=true, calculate_stats=true)
        
        # 2. Extract errors and evaluate Empirical Order of Convergence (EOC)
        errors_100 = Dict{Int, Float64}()
        errors_200 = Dict{Int, Float64}()
        
        for order in 2:5
            scheme_sym = Symbol("muscl$order")
            base_p = merge(shared_muscl, methods_muscl[scheme_sym])
            
            sim_100 = load_sim_data(create_param_dict(base_p..., :Ns => (300,)))
            sim_200 = load_sim_data(create_param_dict(base_p..., :Ns => (600,)))
            
            err_100 = sim_100.stats[:relative_l2error][end][1]
            err_200 = sim_200.stats[:relative_l2error][end][1]
            
            errors_100[order] = err_100
            errors_200[order] = err_200
            
            # Calculate Empirical Order of Convergence: log2(E_coarse / E_fine)
            eoc = log2(err_100 / err_200)
            
            @testset "MUSCL Order $order Convergence" begin
                # Assert error reduces with grid refinement
                @test err_200 < err_100
                
                # Test grouped order behaviors (Odd/Even grouping)
                if order == 2 || order == 3
                    # Expecting ~2nd order convergence
                    @test 1.5 < eoc < 2.5
                elseif order == 4 || order == 5
                    # Expecting ~4th order convergence
                    @test 3.5 < eoc < 5.
                end
            end
        end
        
        @testset "Error Hierarchy" begin
            # Ensure the 4th-order group is strictly more accurate than the 2nd-order group
            @test errors_200[4] < errors_200[2]
            @test errors_200[5] < errors_200[3]
            
            # The odd grouped orders (3 and 5) usually yield similar or slightly better 
            # absolute errors than their even counterparts (2 and 4) at the same resolution.
            @test errors_200[3] <= errors_200[2]
            @test errors_200[5] <= errors_200[4]
        end
    end
        @testset "MOOD Boundedness vs Unlimited Oscillations (Box IC)" begin
        shared_mood = copy(shared)
        
        # Inject the Box initial condition: (u_bg, u_box, mins, maxs)[span_2](start_span)[span_2](end_span)
        # Background is 0.0, Box is 1.0, located between x = -0.5 and x = 0.5[span_3](start_span)[span_3](end_span)
        shared_mood[:init_func] = "box"
        shared_mood[:init_params] = ((0.0,), (1.0,), (-0.5,), (0.5,)) 
        
        # 1. MOOD-enabled schemes (Should remain bounded)
        methods_mood = Dict(
            :muscl2 => Dict(:main_gradient => "MUSCL", :main_flux => "Rusanov", :limiter => "none", :order => 2, :mood_criterion => "U2"),
            :muscl3 => Dict(:main_gradient => "MUSCL", :main_flux => "Rusanov", :limiter => "none", :order => 3, :mood_criterion => "U2"),
            :muscl4 => Dict(:main_gradient => "MUSCL", :main_flux => "Rusanov", :limiter => "none", :order => 4, :mood_criterion => "U2"),
            :muscl5 => Dict(:main_gradient => "MUSCL", :main_flux => "Rusanov", :limiter => "none", :order => 5, :mood_criterion => "U2")
        )

        # 2. Unbounded schemes (No Limiters, No MOOD)[span_4](start_span)[span_4](end_span)
        methods_nomood = Dict(
            :muscl2_unlimited => Dict(:main_gradient => "MUSCL", :main_flux => "Rusanov", :limiter => "none", :order => 2, :mood_criterion => "none"),
            :muscl3_unlimited => Dict(:main_gradient => "MUSCL", :main_flux => "Rusanov", :limiter => "none", :order => 3, :mood_criterion => "none"),
            :muscl4_unlimited => Dict(:main_gradient => "MUSCL", :main_flux => "Rusanov", :limiter => "none", :order => 4, :mood_criterion => "none"),
            :muscl5_unlimited => Dict(:main_gradient => "MUSCL", :main_flux => "Rusanov", :limiter => "none", :order => 5, :mood_criterion => "none")
        )
        
        # Merge both dictionaries to run them in a single orchestration sweep
        all_methods = merge(methods_mood, methods_nomood)
        method_keys = collect(keys(all_methods))
        
        varied_mood = Dict(:Ns => [(100,)])
        
        config_mood = SimulationConfig(
            :run_simulation,
            shared_mood,
            all_methods,
            method_keys;
            varied_params = varied_mood,
            ref_func_name = :analytical_solution
        )
        
        run_all_simulations(config_mood; force_overwrite=true, calculate_stats=false)
        
        tol = 1e-2 
        
        for order in 2:5
            @testset "Order $order Behaviors" begin
                # --- MOOD Check ---
                base_p_mood = merge(shared_mood, methods_mood[Symbol("muscl$order")])
                sim_mood = load_sim_data(create_param_dict(base_p_mood..., :Ns => (100,)))
                
                min_mood = minimum(val[1] for val in sim_mood.u[end])
                max_mood = maximum(val[1] for val in sim_mood.u[end])
                
                # Assert the MOOD solution stays within the [0, 1] background and box values[span_5](start_span)[span_5](end_span)
                @test min_mood >= 0.0 - tol
                @test max_mood <= 1.0 + tol

                # --- Unlimited Check ---
                base_p_unlim = merge(shared_mood, methods_nomood[Symbol("muscl$(order)_unlimited")])
                sim_unlim = load_sim_data(create_param_dict(base_p_unlim..., :Ns => (100,)))
                
                min_unlim = minimum(val[1] for val in sim_unlim.u[end])
                max_unlim = maximum(val[1] for val in sim_unlim.u[end])
                
                # Assert the Unlimited solution oscillates beyond the constraints
                @test (min_unlim < 0.0 - tol) || (max_unlim > 1.0 + tol)
            end
        end
    end
        @testset "Slope Limiter Boundedness (VK Limiter, Box IC)" begin
        shared_lim = copy(shared)
        
        # Inject the Box initial condition: (u_bg, u_box, mins, maxs)[span_0](start_span)[span_0](end_span)
        # Background is 0.0, Box is 1.0, located between x = -0.5 and x = 0.5
        shared_lim[:init_func] = "box"
        shared_lim[:init_params] = ((0.0,), (1.0,), (-0.5,), (0.5,)) 
        
        # Enable the VK limiter and completely disable MOOD[span_1](start_span)[span_1](end_span)
        methods_vk = Dict(
            :muscl2_vk => Dict(:main_gradient => "MUSCL", :main_flux => "Rusanov", :limiter => "VK", :order => 2, :mood_criterion => "none"),
            :muscl3_vk => Dict(:main_gradient => "MUSCL", :main_flux => "Rusanov", :limiter => "VK", :order => 3, :mood_criterion => "none"),
            :muscl4_vk => Dict(:main_gradient => "MUSCL", :main_flux => "Rusanov", :limiter => "VK", :order => 4, :mood_criterion => "none"),
            :muscl5_vk => Dict(:main_gradient => "MUSCL", :main_flux => "Rusanov", :limiter => "VK", :order => 5, :mood_criterion => "none")
        )
        
        varied_lim = Dict(:Ns => [(100,)])
        
        config_vk = SimulationConfig(
            :run_simulation,
            shared_lim,
            methods_vk,
            collect(keys(methods_vk));
            varied_params = varied_lim,
            ref_func_name = :analytical_solution
        )
        
        # Run the simulations
        run_all_simulations(config_vk; force_overwrite=true, calculate_stats=false)
        
        # Tolerance for minor meshfree scattered interpolation noise
        tol = 3e-2 
        
        for order in 2:5
            @testset "VK Limiter Order $order" begin
                base_p = merge(shared_lim, methods_vk[Symbol("muscl$(order)_vk")])
                sim_vk = load_sim_data(create_param_dict(base_p..., :Ns => (100,)))
                
                # Extract the final snapshot
                min_vk = minimum(val[1] for val in sim_vk.u[end])
                max_vk = maximum(val[1] for val in sim_vk.u[end])
                
                # Assert the VK limiter independently keeps the solution bounded within [0, 1][span_2](start_span)[span_2](end_span)
                @test min_vk >= 0.0 - tol
                @test max_vk <= 1.0 + tol
            end
        end
    end
    @testset "2D Upwind Algorithms (Classic, Tiwari, Praveen Convergence)" begin
        # 1. Base parameters for 2D Advection
        shared_2d_upwind = Dict(
            :PDE => "linear",
            :PDE_params => ((1.0,), (1.0,)), 
            :mins => (-1.0, -1.0),
            :maxs => (1.0, 1.0),
            :periodic => true,
            :domain => "rectangular",
            # 2D Sine IC: (amplitude, period_xy, offset)[span_0](start_span)[span_0](end_span)
            # Smooth initial condition ensures convergence rates aren't degraded by boundary kinks
            :init_func => "gauss",
            :init_params => (1., (0.0, 0.0), .1), 
            :tmax => 0.1,
            :timestepper => "RK2", 
            :interp_range => 2.5,
            :interp_alpha => 1.0,
            :snapshots => 3,
            :SEED => 42,
            :CFL => 0.4,
            :Ns => (20,20)
        )

        # 2. Define the different algorithmic branches of UpwindDivergence
        methods_2d_upwind = Dict(
            :upwind_classic => Dict(:main_gradient => "Upwind", :main_flux => "Rusanov", :upwind_alg_nd => "Classic", :order => 1),
            :upwind_tiwari  => Dict(:main_gradient => "Upwind", :main_flux => "Rusanov", :upwind_alg_nd => "Tiwari",  :order => 1),
            :upwind_praveen => Dict(:main_gradient => "Upwind", :main_flux => "Rusanov", :upwind_alg_nd => "Praveen", :order => 1)
        )

        # Sweep two resolutions to measure 1st-order convergence
        varied_2d = Dict(:Ns => [(50, 50), (100, 100)])

        config_2d_upwind = SimulationConfig(
            :run_simulation,
            shared_2d_upwind,
            methods_2d_upwind,
            [:upwind_classic, :upwind_tiwari, :upwind_praveen];
            varied_params = varied_2d,
            ref_func_name = :analytical_solution
        )

        # Run the complete 2D pipeline
        run_all_simulations(config_2d_upwind; force_overwrite=true, calculate_stats=true)

        # 3. Verification across algorithms
        for alg in [:upwind_classic, :upwind_tiwari, :upwind_praveen]
            @testset "Algorithm: $alg" begin
                base_p = merge(shared_2d_upwind, methods_2d_upwind[alg])
                
                sim_20 = load_sim_data(create_param_dict(base_p..., :Ns => (50, 50)))
                sim_40 = load_sim_data(create_param_dict(base_p..., :Ns => (100, 100)))

                # Ensure output structure is successfully parsed as 2D Lagrangian data
                @test sim_20 isa LSimData
                @test length(sim_20.x[1][1]) == 2 
                
                err_20 = sim_20.stats[:relative_l2error][end][1]
                err_40 = sim_40.stats[:relative_l2error][end][1]
                
                @testset "Error Reduction & EOC" begin
                    # Assert error strictly reduces with grid refinement
                    @test err_40 < err_20
                    
                    # Calculate Empirical Order of Convergence (EOC)
                    eoc = log2(err_20 / err_40)
                    
                    # Assert 1st-order convergence behavior (Meshfree EOC usually hovers between 0.5 and 1.5 for 1st order)
                    @test 0.5 < eoc < 1.5
                end
            end
        end
    end
        @testset "Time Integration Execution (Explicit RK & IMEX)" begin
        # Base parameters for a 1D Advection MUSCL 5 problem
        shared_time = Dict(
            :PDE => "linear",
            :PDE_params => ((1.0,),), 
            :mins => (-1.0,),
            :maxs => (1.0,),
            :periodic => true,
            :domain => "rectangular",
            :init_func => "gauss",
            :init_params => (1.0, (0.0,), 1.0), 
            :tmax => 0.05, # Very short runtime just to verify execution
            :main_gradient => "MUSCL",
            :main_flux => "Rusanov",
            :limiter => "none",
            :order => 5,
            :mood_criterion => "none",
            :interp_range => 2.5,
            :interp_alpha => 1.0,
            :snapshots => 2,
            :SEED => 42,
            :Ns => (100,)
        )
        
        # A lightweight 50-particle grid is enough to check for NaNs/Crashes
        varied_time = Dict(:Ns => [(50,)])
        
        @testset "Explicit Runge-Kutta Methods" begin
            # Note: Explicit Euler requires a very small CFL to remain stable 
            # alongside a 5th-order spatial scheme.
            methods_rk = Dict(
                :ts_euler => Dict(:timestepper => "Euler", :CFL => 0.01),
                :ts_rk2   => Dict(:timestepper => "RK2",   :CFL => 0.1),
                :ts_rk3   => Dict(:timestepper => "RK3",   :CFL => 0.3),
                :ts_rk4   => Dict(:timestepper => "RK4",   :CFL => 0.4)
            )
            
            config_rk = SimulationConfig(
                :run_simulation,
                shared_time,
                methods_rk,
                collect(keys(methods_rk));
                varied_params = varied_time,
                ref_func_name = :analytical_solution
            )
            
            run_all_simulations(config_rk; force_overwrite=true, calculate_stats=true)
            
            for rk in keys(methods_rk)
                @testset "Method: $rk" begin
                    base_p = merge(shared_time, methods_rk[rk])
                    sim = load_sim_data(create_param_dict(base_p..., :Ns => (50,)))
                    
                    @test sim isa LSimData
                    @test haskey(sim.stats, :l1error)
                    
                    # Verify the simulation did not blow up (NaNs or Infs)
                    err = sim.stats[:l1error][end][1]
                    @test isfinite(err)
                end
            end
        end

        @testset "IMEX Relaxation Methods" begin
            shared_imex = copy(shared_time)
            
            # Setup Kinetic Relaxation system for 1D Advection
            # Macro advection speed is 1.0, so kinetic speeds must bound it (e.g., -2.0, 2.0).
            shared_imex[:relax_velocities] = ((-2.0, 2.0),)
            shared_imex[:relax_indices] = [1, 3] # 2 kinetic speeds mapped to 1 macro variable
            shared_imex[:relax_epsilon] = 1e-4
            
            methods_imex = Dict(
                :ts_ars233 => Dict(:timestepper => "ARS233", :CFL => 0.3),
                :ts_prSSP3 => Dict(:timestepper => "PRSSP3", :CFL => 0.3),
                :ts_ars222 => Dict(:timestepper => "ARS222", :CFL => 0.3),
                :ts_ssp332 => Dict(:timestepper => "SSP332", :CFL => 0.3),
                :ts_imexE  => Dict(:timestepper => "IMEXEuler", :CFL => 0.05)
            )
            
            config_imex = SimulationConfig(
                :run_simulation,
                shared_imex,
                methods_imex,
                collect(keys(methods_imex));
                varied_params = varied_time,
                ref_func_name = :analytical_solution
            )
            
            run_all_simulations(config_imex; force_overwrite=true, calculate_stats=true)
            
            for imex in keys(methods_imex)
                @testset "Method: $imex" begin
                    base_p = merge(shared_imex, methods_imex[imex])
                    sim = load_sim_data(create_param_dict(base_p..., :Ns => (50,)))
                    
                    @test sim isa LSimData
                    @test haskey(sim.stats, :l1error)
                    
                    # Verify the implicit solver handled the stiff source term without blowing up
                    err = sim.stats[:l1error][end][1]
                    @test isfinite(err)
                end
            end
        end
    end


end
