using Test
using HyperCloudCore
import HyperCloudCore: flux, velocity, max_eigenvalue, prim2cons, cons2prim, implicit_solve, evaluate_source, pre_solve_update!, math_min, math_max, basis_length, param2vel
using PDEStudioCore
using StaticArrays
using LinearAlgebra
using Random

include("ICs/_main.jl")
include("PDEs/_main.jl")
include("SimulationFunctions/_main.jl")
include("Builder/_main.jl")

include("kinetic_source.jl")
include("time_integration.jl")

# Point PDEStudioCore to look inside this test environment
PDEStudioCore.set_target_module!(@__MODULE__)

@testset "HyperCloud Pipeline Tests" begin
    # Setup isolated temp directories and stats preset
    set_stat_preset!("hyperbolic")
    tmp_dir = mktempdir()
    set_save_path!(tmp_dir)

    # Base parameters shared across all spatial schemes
    shared = Dict{Symbol, Any}(
        :sim_func_name => :run_direct_simulation, # Replaces positional argument
        
        # Global
        :snapshots => 5,
        :tmax => 2.0,
        
        # Grid Namespace
        :Grid_domain => :rectangular,
        :Grid_mins => (-1.0,),
        :Grid_maxs => (1.0,),
        :Grid_Ns => (100,),
        :Grid_periodic => true,
        :Grid_SEED => 42,
        :Grid_randomness_factor => (0.0,),
        
        # Weight Namespace
        :Weight_name => :exponential,
        :Weight_range => 2.5,
        :Weight_alpha => 1.0,
        
        # PDE Namespace
        :PDE_name => :linear,
        :PDE_velocities => ((1.0,),),          
        
        # IC Namespace (Gauss)
        :IC_name => :gauss,
        :IC_a => (1.0,),
        :IC_b => (0.0,),
        :IC_width => 0.1,

        # Time Namespace
        :Time_stepper => :RK3,                 
        :Time_CFL => 0.4,
        
        # Scheme Fallbacks
        :Scheme_MLS_order => 0
    )

    # Define the different spatial reconstruction schemes with strictly required parameters
    methods = Dict(
        :upwind => Dict(
            :Scheme_name => :Upwind,
            :Flux_name => :Rusanov,
            :Scheme_upwind_alg_nd => :Classic, # Strictly Required!
            :Scheme_order => 1
        ),
        :muscl => Dict(
            :Scheme_name => :MUSCL,
            :Flux_name => :Rusanov,
            :Limiter_name => :minmod,
            :Limiter_mode => :hard,            # Strictly Required!
            :Scheme_order => 2
        ),
        :weno => Dict(
            :Scheme_name => :WENO,
            :Flux_name => :Rusanov,
            :Scheme_order => 2
        ),
        :central => Dict(
            :Scheme_name => :Central,
            :Flux_name => :Rusanov,
            :Scheme_order => 2
        )
    )

    # Parameter Sweep for Grid Convergence
    varied = Dict(
        :Grid_Ns => [(200,), (400,)]
    )

    # Build the SimulationConfig (No positional sim_func_name!)
    config = SimulationConfig(
        shared,
        methods,
        [:upwind, :muscl, :weno, :central];
        varied_params = varied,
        ref_func_name = :analytical_solution
    )

    # Run the complete pipeline (Generation + Stats)
    run_all_simulations(config; force_overwrite=true, calculate_stats=true)
    # Minimal concrete types required to instantiate the abstract API hierarchies
    struct DummyPDE <: HyperbolicPDE{1, 1, Float64} end 
    struct DummyExplicitSource <: AbstractExplicitSourceTerm end 
    struct DummyImplicitSource <: AbstractImplicitSourceTerm end 
    struct DummyInterpolator <: DivergenceInterpolator end 
    struct DummyTimeStepper <: TimeStepper end 

    @testset "API Contracts & Fallbacks" begin
        # Setup dummy state variables
        eq = DummyPDE()
        u_state = SVector{1, Float64}(1.5)
        pg_dummy = nothing # Placeholder for ParticleGrid where type is untyped in signatures
        
        @testset "1. PDE API (Physical Equations)" begin
            # Abstract throws
            @test_throws ErrorException flux(eq, u_state)
            @test_throws ErrorException max_eigenvalue(eq, u_state, 1)
            @test_throws ErrorException velocity(eq, u_state, 1)
            
            # Identity mappings
            @test prim2cons(eq, u_state) == u_state 
            @test cons2prim(eq, u_state) == u_state 
            
            # Non-conservative jump for Conservative PDEs
            f_L = SVector{1, SVector{1, Float64}}((SVector{1, Float64}(1.0),))
            f_R = SVector{1, SVector{1, Float64}}((SVector{1, Float64}(2.0),))
            dist = SVector{1, Float64}(0.5)
            
            nc_jump = evaluate_nc_jump(eq, f_L, f_R, dist) 
            @test nc_jump isa SVector{1, SVector{1, Float64}}
            @test nc_jump[1][1] == 0.0 
        end
        
        @testset "2. Source Term API" begin
            exp_st = DummyExplicitSource()
            imp_st = DummyImplicitSource()
            no_exp = NoExplicitSource() #
            no_imp = NoImplicitSource() #
            
            # Explicit sources
            @test_throws ErrorException evaluate_source(exp_st, u_state, 1, pg_dummy, 0.0) 
            @test evaluate_source(no_exp, u_state, 1, pg_dummy, 0.0) == zero(u_state) 
            
            # Implicit sources
            @test_throws ErrorException evaluate_source(imp_st, u_state, 1, pg_dummy, 0.0) 
            @test evaluate_source(no_imp, u_state, 1, pg_dummy, 0.0) == zero(u_state) 
            
            # Global updates
            @test pre_solve_update!(imp_st, nothing, pg_dummy, 0.0) === nothing 
            
            # Implicit solve operations
            @test_throws ErrorException implicit_solve(imp_st, u_state, 0.1, 1, pg_dummy, 0.0) 
            @test implicit_solve(no_imp, u_state, 0.1, 1, pg_dummy, 0.0) == u_state 
        end
        
        @testset "3. Divergence Interpolator API" begin
            interp = DummyInterpolator()
            
            # Missing required implementations
            @test_throws ErrorException update_size!(interp, 100) 
            @test_throws ErrorException _extract_order(interp) 
            
            # Default no-ops
            @test update_content!(interp, 1, u_state, 1:5, pg_dummy, nothing) === nothing 
        end
        
        @testset "4. TimeStepper API" begin
            ts = DummyTimeStepper()
            
            # Missing required implementation
            @test_throws ErrorException update_size!(ts, 100, 50) 
        end
    end
    @testset "Dynamic Grid Resizing (ensure_capacity!)" begin
        geom = get_rectangular_domain(Float64, (0.0,), (1.0,); is_periodic = true)
        pg = ParticleGrid(geom, (0.1,), ExponentialWeightFunction(1.,2.5), 1)
        
        N_initial = length(pg.rhos)
        @test N_initial > 0
        
        req_cap = N_initial + 50
        expected_cap = ceil(Int, req_cap * 1.25)
        
        HyperCloudCore.ensure_capacity!(pg, req_cap)
        
        @testset "Top-Level Grid Arrays" begin
            @test length(pg.rhos) == expected_cap
            @test length(pg.curvatures) == expected_cap
        end
        
        @testset "ParticleGridCore Arrays" begin
            @test length(pg.core.positions) == expected_cap
            @test length(pg.core.is_boundary) == expected_cap
            @test length(pg.core.volumes) == expected_cap
            @test length(pg.core.tags) == expected_cap
            @test length(pg.core.mood_triggered) == expected_cap
            @test length(pg.core.particle_orders) == expected_cap
        end
        
        @testset "SharedBuffers Arrays" begin
            @test length(pg.shared.rho_buffer) == expected_cap
            @test length(pg.shared.pos_buffer) == expected_cap
            @test length(pg.shared.float_buffer) == expected_cap
            @test length(pg.shared.bit_buffer) == expected_cap
            @test length(pg.shared.int_buffer) == expected_cap
        end
        
        @testset "ReorderData Arrays" begin
            @test length(pg.reorder.permutation) == expected_cap
            @test length(pg.reorder.inv_permutation) == expected_cap
            @test length(pg.reorder.new_permutation_buffer) == expected_cap
            @test length(pg.reorder.seen_buffer) == expected_cap
        end
        
        @testset "GlobalBins Arrays" begin
            @test length(pg.bins.next) == expected_cap
        end
        
        @testset "NeighborData Arrays" begin
            N_nb_initial = length(pg.neighbor.indices)
            req_nb_cap = N_nb_initial + 100
            expected_nb_cap = ceil(Int, req_nb_cap * 1.25)
            
            HyperCloudCore.ensure_capacity!(pg.neighbor, req_nb_cap)
            
            @test length(pg.neighbor.indices) == expected_nb_cap
            @test length(pg.neighbor.weights) == expected_nb_cap
            @test length(pg.neighbor.distances) == expected_nb_cap
        end
        
        @testset "Safe No-Op on Shrink" begin
            HyperCloudCore.ensure_capacity!(pg, req_cap - 10)
            @test length(pg.rhos) == expected_cap 
        end
    end

    # Verify each scheme independently
    for scheme in [:upwind, :muscl, :weno, :central]
        @testset "Scheme: $scheme" begin
            base_params = merge(shared, methods[scheme])
            params_100 = create_param_dict(base_params..., :Grid_Ns => (200,))
            params_200 = create_param_dict(base_params..., :Grid_Ns => (400,))

            @test does_sim_data_exist(params_100)
            @test does_sim_data_exist(params_200)

            sim_100 = load_sim_data(params_100)
            sim_200 = load_sim_data(params_200)

            @test sim_100 isa LSimData

            @testset "Statistic Generation" begin
                @test haskey(sim_100.stats, :l1error)
                @test haskey(sim_100.stats, :mass)
                @test length(sim_100.stats[:l1error]) == length(sim_100.t)
            end

            @testset "Mass Conservation" begin
                mass_200 = sim_200.stats[:mass]
                initial_mass = mass_200[1][1]
                @test all(m -> isapprox(m[1], initial_mass; rtol=1e-2), mass_200)
            end

            @testset "Grid Convergence" begin
                err_100 = sim_100.stats[:l1error][end][1]
                err_200 = sim_200.stats[:l1error][end][1]
                @test err_200 < err_100
            end
        end
    end

    @testset "Core Utility Functions (Params & SIMD Math)" begin
        @testset "param2uvec (State Conversions)" begin
            u_scalar = param2uvec(5.0)
            @test u_scalar isa SVector{1, Float64}
            @test u_scalar[1] == 5.0
            
            u_tuple = param2uvec((1.0, 2.0, 3.0))
            @test u_tuple isa SVector{3, Float64}
            @test u_tuple == SVector(1.0, 2.0, 3.0)
            
            u_arr = param2uvec([4.0, 5.0])
            @test u_arr isa SVector{2, Float64}
            @test u_arr == SVector(4.0, 5.0)

            # Added to cover param2uvec(::SVector)
            u_svec = param2uvec(SVector(6.0, 7.0))
            @test u_svec isa SVector{2, Float64}
            @test u_svec == SVector(6.0, 7.0)
        end
        
        @testset "param2xvec (Space Conversions)" begin
            x_scalar = param2xvec(-1.5)
            @test x_scalar isa SVector{1, Float64}
            
            x_tuple = param2xvec((0.0, 1.0))
            @test x_tuple isa SVector{2, Float64}

            # Added to cover param2xvec(::AbstractVector) and param2xvec(::SVector)
            x_arr = param2xvec([2.0, 3.0])
            @test x_arr isa SVector{2, Float64}
            
            x_svec = param2xvec(SVector(4.0, 5.0))
            @test x_svec isa SVector{2, Float64}
        end
        
        @testset "param2fvec (Flux Conversions)" begin
            f_scalar = param2fvec(2.5)
            @test f_scalar isa SVector{1, SVector{1, Float64}}
            @test f_scalar[1][1] == 2.5
            
            f_vec_tuple = param2fvec(([1.0, 2.0], [3.0, 4.0]))
            @test f_vec_tuple isa SVector{2, SVector{2, Float64}}
            @test f_vec_tuple[1] == SVector(1.0, 2.0)
            @test f_vec_tuple[2] == SVector(3.0, 4.0)
            
            f_vec_arr = param2fvec([[5.0], [6.0], [7.0]])
            @test f_vec_arr isa SVector{3, SVector{1, Float64}}
            @test f_vec_arr[3][1] == 7.0

            # Added to cover param2fvec(::NTuple{D, NTuple{M, T}})
            f_tuple_tuple = param2fvec(((1.0, 2.0), (3.0, 4.0)))
            @test length(f_tuple_tuple) == 2

            # Added to cover param2fvec(::Vector{T})
            f_flat_arr = param2fvec([8.0, 9.0])
            @test length(f_flat_arr) == 1
            @test length(f_flat_arr[1]) == 2
        end
        
        @testset "param2svec (Nested State Conversions)" begin
            s_tuple = param2svec((1.5, 2.5))
            @test s_tuple isa SVector{2, SVector{1, Float64}}
            @test s_tuple[1][1] == 1.5
            @test s_tuple[2][1] == 2.5

            # Added to cover param2svec(::Real)
            s_scalar = param2svec(3.14)
            @test s_scalar isa SVector{1, SVector{1, Float64}}
            @test s_scalar[1][1] == 3.14
            
            # Added to cover param2svec(::Flux) / identity fallthrough
            f_val = param2fvec(2.5) 
            s_flux = param2svec(f_val)
            @test s_flux == f_val
        end

        # Added entirely missing testset to cover param2vel (lines 78, 81, 84)
        @testset "param2vel (Velocity Casting)" begin
            # 1. Single scalar
            vel_scalar = param2vel(2.5)
            @test vel_scalar isa SVector{1, SMatrix{1, 1, Float64, 1}}
            @test vel_scalar[1][1, 1] == 2.5
            
            # 2. Tuple of scalars
            vel_tuple = param2vel((1.0, 2.0))
            @test vel_tuple isa SVector{2, SMatrix{1, 1, Float64, 1}}
            
            # 3. Tuple of Tuples (Diagonals)
            vel_diag = param2vel(((1.0, 2.0), (3.0, 4.0)))
            @test length(vel_diag) == 2
        end
        
        @testset "Branchless SIMD Math (math_max / math_min)" begin
            @test math_max(10.0, 5.0) == 10.0
            @test math_max(-2.0, 3.0) == 3.0
            
            @test math_min(10.0, 5.0) == 5.0
            @test math_min(-2.0, 3.0) == -2.0
            
            v1 = [1.0, 5.0, 3.0]
            v2 = [2.0, 4.0, 3.0]
            
            @test math_max(v1, v2) == [2.0, 5.0, 3.0]
            @test math_min(v1, v2) == [1.0, 4.0, 3.0]
        end
    end
    
    @testset "MUSCL Higher-Order Convergence (Orders 2-5)" begin
        shared_muscl = copy(shared)
        shared_muscl[:Time_stepper] = :RK4
        
        methods_muscl = Dict(
            :muscl2 => Dict(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Limiter_name => :none, :Scheme_order => 2),
            :muscl3 => Dict(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Limiter_name => :none, :Scheme_order => 3),
            :muscl4 => Dict(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Limiter_name => :none, :Scheme_order => 4),
            :muscl5 => Dict(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Limiter_name => :none, :Scheme_order => 5)
        )
        
        config_muscl = SimulationConfig(
            shared_muscl,
            methods_muscl,
            [:muscl2, :muscl3, :muscl4, :muscl5];
            varied_params = create_varied_dict(:Grid_Ns => [(200,),(400,)]),
            ref_func_name = :analytical_solution
        )
        
        run_all_simulations(config_muscl; force_overwrite=true, calculate_stats=true)
        
        errors_100 = Dict{Int, Float64}()
        errors_200 = Dict{Int, Float64}()
        
        for order in 2:5
            scheme_sym = Symbol("muscl$order")
            base_p = merge(shared_muscl, methods_muscl[scheme_sym])
            
            sim_100 = load_sim_data(create_param_dict(base_p..., :Grid_Ns => (200,)))
            sim_200 = load_sim_data(create_param_dict(base_p..., :Grid_Ns => (400,)))
            
            err_100 = sim_100.stats[:relative_l2error][end][1]
            err_200 = sim_200.stats[:relative_l2error][end][1]
            
            errors_100[order] = err_100
            errors_200[order] = err_200
            
            eoc = log2(err_100 / err_200)
            
            @testset "MUSCL Order $order Convergence" begin
                @test err_200 < err_100
                if order == 2 || order == 3
                    @test 1.5 < eoc < 2.5
                elseif order == 4 || order == 5
                    @test 3.5 < eoc < 5.0
                end
            end
        end
        
        @testset "Error Hierarchy" begin
            @test errors_200[4] < errors_200[2]
            @test errors_200[5] < errors_200[3]
            @test errors_200[3] <= errors_200[2]
            @test errors_200[5] <= errors_200[4]
        end
    end
    
    @testset "MUSCL Higher-Order Convergence (Orders 2-5) with MOOD" begin
        shared_muscl = copy(shared)
        shared_muscl[:Time_stepper] = :RK4 
        
        # MOOD Strictly Requires Strategy and Delta Relax
        methods_muscl = Dict(
            :muscl2 => Dict(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Limiter_name => :none, :Scheme_order => 2, :MOOD_criterion => :U2, :MOOD_strategy => :EPD1, :MOOD_delta_relax => 1e-4),
            :muscl3 => Dict(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Limiter_name => :none, :Scheme_order => 3, :MOOD_criterion => :U2, :MOOD_strategy => :EPD1, :MOOD_delta_relax => 1e-4),
            :muscl4 => Dict(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Limiter_name => :none, :Scheme_order => 4, :MOOD_criterion => :U2, :MOOD_strategy => :EPD1, :MOOD_delta_relax => 1e-4),
            :muscl5 => Dict(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Limiter_name => :none, :Scheme_order => 5, :MOOD_criterion => :U2, :MOOD_strategy => :EPD1, :MOOD_delta_relax => 1e-4)
        )
        
        config_muscl = SimulationConfig(
            shared_muscl,
            methods_muscl,
            [:muscl2, :muscl3, :muscl4, :muscl5];
            varied_params = create_varied_dict(:Grid_Ns => [(300,),(600,)]),
            ref_func_name = :analytical_solution
        )
        
        run_all_simulations(config_muscl; force_overwrite=true, calculate_stats=true)
        
        for order in 2:5
            scheme_sym = Symbol("muscl$order")
            base_p = merge(shared_muscl, methods_muscl[scheme_sym])
            
            sim_100 = load_sim_data(create_param_dict(base_p..., :Grid_Ns => (300,)))
            sim_200 = load_sim_data(create_param_dict(base_p..., :Grid_Ns => (600,)))
            
            err_100 = sim_100.stats[:relative_l2error][end][1]
            err_200 = sim_200.stats[:relative_l2error][end][1]
            eoc = log2(err_100 / err_200)
            
            @testset "MUSCL Order $order Convergence" begin
                @test err_200 < err_100
                if order == 2 || order == 3
                    @test 1.5 < eoc < 2.5
                elseif order == 4 || order == 5
                    @test 3.5 < eoc < 5.0
                end
            end
        end
    end

    @testset "MOOD Boundedness vs Unlimited Oscillations (Box IC)" begin
        shared_mood = copy(shared)
        shared_mood[:IC_name] = :box
        shared_mood[:IC_u_bg] = (0.0,)
        shared_mood[:IC_u_box] = (1.0,)
        shared_mood[:IC_mins] = (-0.5,)
        shared_mood[:IC_maxs] = (0.5,)
        
        methods_mood = Dict(
            :muscl2 => Dict(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Limiter_name => :none, :Scheme_order => 2, :MOOD_criterion => :U2, :MOOD_strategy => :EPD1, :MOOD_delta_relax => 1e-4),
            :muscl3 => Dict(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Limiter_name => :none, :Scheme_order => 3, :MOOD_criterion => :U2, :MOOD_strategy => :EPD1, :MOOD_delta_relax => 1e-4),
            :muscl4 => Dict(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Limiter_name => :none, :Scheme_order => 4, :MOOD_criterion => :U2, :MOOD_strategy => :EPD1, :MOOD_delta_relax => 1e-4),
            :muscl5 => Dict(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Limiter_name => :none, :Scheme_order => 5, :MOOD_criterion => :U2, :MOOD_strategy => :EPD1, :MOOD_delta_relax => 1e-4)
        )

        methods_nomood = Dict(
            :muscl2_unlimited => Dict(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Limiter_name => :none, :Scheme_order => 2),
            :muscl3_unlimited => Dict(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Limiter_name => :none, :Scheme_order => 3),
            :muscl4_unlimited => Dict(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Limiter_name => :none, :Scheme_order => 4),
            :muscl5_unlimited => Dict(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Limiter_name => :none, :Scheme_order => 5)
        )
        
        all_methods = merge(methods_mood, methods_nomood)
        varied_mood = Dict(:Grid_Ns => [(100,)])
        
        config_mood = SimulationConfig(
            shared_mood,
            all_methods,
            collect(keys(all_methods));
            varied_params = varied_mood,
            ref_func_name = :analytical_solution
        )
        
        run_all_simulations(config_mood; force_overwrite=true, calculate_stats=false)
        
        tol = 1e-2 
        
        for order in 2:5
            @testset "Order $order Behaviors" begin
                base_p_mood = merge(shared_mood, methods_mood[Symbol("muscl$order")])
                sim_mood = load_sim_data(create_param_dict(base_p_mood..., :Grid_Ns => (100,)))
                
                min_mood = minimum(val[1] for val in sim_mood.u[end])
                max_mood = maximum(val[1] for val in sim_mood.u[end])
                
                @test min_mood >= 0.0 - tol
                @test max_mood <= 1.0 + tol

                base_p_unlim = merge(shared_mood, methods_nomood[Symbol("muscl$(order)_unlimited")])
                sim_unlim = load_sim_data(create_param_dict(base_p_unlim..., :Grid_Ns => (100,)))
                
                min_unlim = minimum(val[1] for val in sim_unlim.u[end])
                max_unlim = maximum(val[1] for val in sim_unlim.u[end])
                
                @test (min_unlim < 0.0 - tol) || (max_unlim > 1.0 + tol)
            end
        end
    end

    @testset "Slope Limiter Boundedness (VK Limiter, Box IC)" begin
        shared_lim = copy(shared)
        shared_lim[:IC_name] = :box
        shared_lim[:IC_u_bg] = (0.0,)
        shared_lim[:IC_u_box] = (1.0,)
        shared_lim[:IC_mins] = (-0.5,)
        shared_lim[:IC_maxs] = (0.5,)
        
        # Limiter requires mode
        methods_vk = Dict(
            :muscl2_vk => Dict(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Limiter_name => :VK, :Limiter_mode => :hard, :Scheme_order => 2),
            :muscl3_vk => Dict(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Limiter_name => :VK, :Limiter_mode => :hard, :Scheme_order => 3),
            :muscl4_vk => Dict(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Limiter_name => :VK, :Limiter_mode => :hard, :Scheme_order => 4),
            :muscl5_vk => Dict(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Limiter_name => :VK, :Limiter_mode => :hard, :Scheme_order => 5)
        )
        
        varied_lim = Dict(:Grid_Ns => [(100,)])
        
        config_vk = SimulationConfig(
            shared_lim,
            methods_vk,
            collect(keys(methods_vk));
            varied_params = varied_lim,
            ref_func_name = :analytical_solution
        )
        
        run_all_simulations(config_vk; force_overwrite=true, calculate_stats=false)
        
        tol = 3e-2 
        
        for order in 2:5
            @testset "VK Limiter Order $order" begin
                base_p = merge(shared_lim, methods_vk[Symbol("muscl$(order)_vk")])
                sim_vk = load_sim_data(create_param_dict(base_p..., :Grid_Ns => (100,)))
                
                min_vk = minimum(val[1] for val in sim_vk.u[end])
                max_vk = maximum(val[1] for val in sim_vk.u[end])
                
                @test min_vk >= 0.0 - tol
                @test max_vk <= 1.0 + tol
            end
        end
    end

    @testset "MOOD Strategies & Criteria Execution" begin
        shared_mood_exec = Dict{Symbol, Any}(
            :sim_func_name => :run_direct_simulation,
            :PDE_name => :linear,
            :PDE_velocities => ((1.0,),), 
            :Grid_mins => (-1.0,),
            :Grid_maxs => (1.0,),
            :Grid_periodic => true,
            :Grid_domain => :rectangular,
            :Grid_randomness_factor => (0.0,),
            :IC_name => :box,
            :IC_u_bg => (0.0,),
            :IC_u_box => (1.0,),
            :IC_mins => (-0.5,),
            :IC_maxs => (0.5,), 
            :tmax => 0.05, 
            :Scheme_name => :MUSCL,
            :Flux_name => :Rusanov,
            :Limiter_name => :none,
            :Scheme_order => 2,
            :Scheme_MLS_order => 0,
            :Weight_name => :exponential,
            :Weight_range => 2.5,
            :Weight_alpha => 1.0,
            :snapshots => 2,
            :Grid_SEED => 42,
            :Time_CFL => 0.3,
            :Grid_Ns => (50,),
            :Time_stepper => :RK2,
            :MOOD_delta_relax => 1e-4 # Required when MOOD is active
        )
        
        methods_mood_exec = Dict{Symbol, Any}()
        strategies = [:EPD0, :SEPD0, :EPD1, :EPD2]
        criteria = [:U1, :U2]
        
        for strat in strategies
            for crit in criteria
                sym = Symbol("mood_$(lowercase(string(strat)))_$(lowercase(string(crit)))")
                methods_mood_exec[sym] = Dict(
                    :MOOD_strategy => strat,
                    :MOOD_criterion => crit
                )
            end
        end
        
        config_mood_exec = SimulationConfig(
            shared_mood_exec,
            methods_mood_exec,
            collect(keys(methods_mood_exec));
            ref_func_name = :analytical_solution
        )
        
        run_all_simulations(config_mood_exec; force_overwrite=true, calculate_stats=true)
        
        for strat in strategies
            for crit in criteria
                sym = Symbol("mood_$(lowercase(string(strat)))_$(lowercase(string(crit)))")
                @testset "Strategy: $strat | Criterion: $crit" begin
                    base_p = merge(shared_mood_exec, methods_mood_exec[sym])
                    sim = load_sim_data(create_param_dict(base_p..., :Grid_Ns => (50,)))
                    
                    @test sim isa LSimData
                    @test haskey(sim.stats, :l1error)
                    @test isfinite(sim.stats[:l1error][end][1])
                end
            end
        end
    end

    @testset "2D Upwind Algorithms (Classic, Tiwari, Praveen Convergence)" begin
        shared_2d_upwind = Dict{Symbol, Any}(
            :sim_func_name => :run_direct_simulation,
            :PDE_name => :linear,
            :PDE_velocities => ((1.0,), (1.0,)), 
            :Grid_mins => (-1.0, -1.0),
            :Grid_maxs => (1.0, 1.0),
            :Grid_periodic => true,
            :Grid_domain => :rectangular,
            :Grid_randomness_factor => (0.0,0.0),
            :IC_name => :gauss,
            :IC_a => (1.0,),
            :IC_b => (0.0, 0.0),
            :IC_width => 0.1, 
            :tmax => 0.1,
            :Time_stepper => :RK2, 
            :Weight_name => :exponential,
            :Weight_range => 2.5,
            :Weight_alpha => 1.0,
            :snapshots => 3,
            :Grid_SEED => 42,
            :Time_CFL => 0.4,
            :Grid_Ns => (20,20)
        )

        methods_2d_upwind = Dict(
            :upwind_classic => Dict(:Scheme_name => :Upwind, :Flux_name => :Rusanov, :Scheme_upwind_alg_nd => :Classic, :Scheme_order => 1),
            :upwind_tiwari  => Dict(:Scheme_name => :Upwind, :Flux_name => :Rusanov, :Scheme_upwind_alg_nd => :Tiwari,  :Scheme_order => 1),
            :upwind_praveen => Dict(:Scheme_name => :Upwind, :Flux_name => :Rusanov, :Scheme_upwind_alg_nd => :Praveen, :Scheme_order => 1)
        )

        varied_2d = Dict(:Grid_Ns => [(50, 50), (100, 100)])

        config_2d_upwind = SimulationConfig(
            shared_2d_upwind,
            methods_2d_upwind,
            [:upwind_classic, :upwind_tiwari, :upwind_praveen];
            varied_params = varied_2d,
            ref_func_name = :analytical_solution
        )

        run_all_simulations(config_2d_upwind; force_overwrite=true, calculate_stats=true)

        for alg in [:upwind_classic, :upwind_tiwari, :upwind_praveen]
            @testset "Algorithm: $alg" begin
                base_p = merge(shared_2d_upwind, methods_2d_upwind[alg])
                
                sim_20 = load_sim_data(create_param_dict(base_p..., :Grid_Ns => (50, 50)))
                sim_40 = load_sim_data(create_param_dict(base_p..., :Grid_Ns => (100, 100)))

                @test sim_20 isa LSimData
                @test length(sim_20.x[1][1]) == 2 
                
                err_20 = sim_20.stats[:relative_l2error][end][1]
                err_40 = sim_40.stats[:relative_l2error][end][1]
                
                @testset "Error Reduction & EOC" begin
                    @test err_40 < err_20
                    eoc = log2(err_20 / err_40)
                    @test 0.5 < eoc < 1.5
                end
            end
        end
    end

    @testset "Time Integration Execution (Explicit RK & IMEX)" begin
        shared_time = Dict{Symbol, Any}(
            :PDE_name => :linear,
            :PDE_velocities => ((1.0,),), 
            :Grid_mins => (-1.0,),
            :Grid_maxs => (1.0,),
            :Grid_periodic => true,
            :Grid_domain => :rectangular,
            :Grid_randomness_factor => (0.0,),
            :IC_name => :gauss,
            :IC_a => (1.0,),
            :IC_b => (0.0,),
            :IC_width => 1.0, 
            :tmax => 0.05,
            :Scheme_name => :MUSCL,
            :Flux_name => :Rusanov,
            :Limiter_name => :none,
            :Scheme_order => 5,
            :Scheme_MLS_order => 0,
            :Weight_name => :exponential,
            :Weight_range => 2.5,
            :Weight_alpha => 1.0,
            :snapshots => 2,
            :Grid_SEED => 42,
            :Grid_Ns => (100,)
        )
        
        varied_time = Dict(:Grid_Ns => [(50,)])
        
        @testset "Explicit Runge-Kutta Methods" begin
            methods_rk = Dict(
                :ts_euler => Dict(:Time_stepper => :Euler, :Time_CFL => 0.01, :sim_func_name => :run_direct_simulation),
                :ts_rk2   => Dict(:Time_stepper => :RK2,   :Time_CFL => 0.1,  :sim_func_name => :run_direct_simulation),
                :ts_rk3   => Dict(:Time_stepper => :RK3,   :Time_CFL => 0.3,  :sim_func_name => :run_direct_simulation),
                :ts_rk4   => Dict(:Time_stepper => :RK4,   :Time_CFL => 0.4,  :sim_func_name => :run_direct_simulation)
            )
            
            config_rk = SimulationConfig(
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
                    sim = load_sim_data(create_param_dict(base_p..., :Grid_Ns => (50,)))
                    
                    @test sim isa LSimData
                    @test haskey(sim.stats, :l1error)
                    @test isfinite(sim.stats[:l1error][end][1])
                end
            end
        end

        @testset "IMEX Relaxation Methods" begin
            shared_imex = copy(shared_time)
            
            # Setup Kinetic Relaxation system 
            shared_imex[:sim_func_name] = :run_kinetic_simulation
            shared_imex[:Kinetic_velocities] = ((-2.0, 2.0),)
            shared_imex[:Kinetic_indices] = [1, 3] 
            shared_imex[:Kinetic_epsilon] = 1e-4
            
            methods_imex = Dict(
                :ts_ars233 => Dict(:Time_stepper => :ARS233, :Time_CFL => 0.3),
                :ts_ars233_mood => Dict(:Time_stepper => :ARS233, :Time_CFL => 0.3, :MOOD_criterion => :U2, :MOOD_strategy => :EPD1, :MOOD_delta_relax => 1e-4),
                :ts_prSSP3 => Dict(:Time_stepper => :PRSSP3, :Time_CFL => 0.3),
                :ts_ars222 => Dict(:Time_stepper => :ARS222, :Time_CFL => 0.3),
                :ts_ssp332 => Dict(:Time_stepper => :SSP332, :Time_CFL => 0.3),
                :ts_imexE  => Dict(:Time_stepper => :IMEXEuler, :Time_CFL => 0.05)
            )
            
            config_imex = SimulationConfig(
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
                    sim = load_sim_data(create_param_dict(base_p..., :Grid_Ns => (50,)))
                    
                    @test sim isa LSimData
                    @test haskey(sim.stats, :l1error)
                    @test isfinite(sim.stats[:l1error][end][1])
                end
            end
        end
    end

    @testset "Spherical Domain & Boundary Conditions" begin
        shared_bc = Dict{Symbol, Any}(
            :sim_func_name => :run_direct_simulation,
            :PDE_name => :linear,
            :PDE_velocities => ((1.0,), (1.0,)), 
            :Grid_mins => (-1.0, -1.0),
            :Grid_maxs => (1.0, 1.0),
            :Grid_periodic => false,
            :Grid_domain => :spherical, 
            :Grid_randomness_factor => (.2,.2),
            :IC_name => :gauss,
            :IC_a => (1.0,),
            :IC_b => (0.0, 0.0),
            :IC_width => 0.5, 
            :tmax => 0.1,
            :Scheme_name => :Upwind,
            :Flux_name => :Upwind,
            :Scheme_upwind_alg_nd => :Classic, # Required for Upwind
            :Scheme_order => 1,
            :Time_stepper => :RK2,
            :Weight_name => :exponential,
            :Weight_range => 2.5,
            :Weight_alpha => 1.0,
            :snapshots => 2,
            :Grid_SEED => 42,
            :Time_CFL => 0.4,
            :Grid_Ns => (30,30)
        )
        
        methods_bc = Dict(
            :bc_fixed   => Dict(:Grid_bc => Dict(1 => :fixed_dirichlet)),
            :bc_outflow => Dict(:Grid_bc => Dict(1 => :outflow))
        )
        
        config_bc = SimulationConfig(
            shared_bc,
            methods_bc,
            [:bc_fixed, :bc_outflow];
            ref_func_name = :analytical_solution
        )
        
        run_all_simulations(config_bc; force_overwrite=true, calculate_stats=false)
        
        @testset "Fixed Dirichlet Pipeline" begin
            base_p = merge(shared_bc, methods_bc[:bc_fixed])
            sim = load_sim_data(create_param_dict(base_p..., :Grid_Ns => (30, 30)))
            
            @test sim isa LSimData
            @test length(sim.x[1][1]) == 2
            @test isfinite(sim.u[end][1][1])
        end
        
        @testset "Outflow BC Pipeline" begin
            base_p = merge(shared_bc, methods_bc[:bc_outflow])
            sim = load_sim_data(create_param_dict(base_p..., :Grid_Ns => (30, 30)))
            
            @test sim isa LSimData
            @test length(sim.x[1][1]) == 2
            @test isfinite(sim.u[end][1][1])
        end
    end
end