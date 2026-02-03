using Random

"""
    runScalarSimulation(params::ParamDictType) -> Union{AbstractSimData, Nothing}

Runs a 1D or 2D scalar conservation law simulation.
This is the general-purpose runner for equations like Linear Advection and Burgers.
"""
function runScalarSimulation(params::ParamDictType)::Union{AbstractSimData, Nothing}
    @info "\n--- Running Scalar Simulation ---"
    run_params = copy(params)

    try
        # --- 1. Load Core Parameters ---
        tmax::Float64 = run_params["tmax"]
        xmin::Float64, xmax::Float64 = run_params["xmin"], run_params["xmax"]
        bc::Symbol = run_params["bc"]
        eq_name::String = run_params["PDE"]
        initFunc_name::String = run_params["init_func"]
        init_params = get(run_params, "init_params", nothing)
        timestepper_name = get(run_params, "timestepper", nothing)
        grid_mover_name = get(run_params, "grid_mover", nothing)
        snapshots::Int = run_params["snapshots"]
        
        # --- 2. Determine Dimension and PDE Physics ---
        local dimension::Int
        local eq::ScalarHyperbolicPDE

        # Use the PDE name to determine dimension and create the equation object
        if eq_name == "linear"
            pde_params = run_params["PDE_params"]
            if pde_params isa Real
                dimension = 1
                eq = LinearAdvection(pde_params)
            else
                dimension = 2
                eq = LinearAdvection(Tuple(pde_params))
            end
        elseif eq_name == "burgers"
            dimension = 1
            a = get(run_params,"PDE_params", 0.)
            eq = BurgersEquation(a)
        elseif eq_name == "burgers2d"
            dimension = 2
            eq = BurgersEquation2D()
        elseif eq_name == "testU3"
            a = get(run_params,"PDE_params", 0.)
            dimension = 1
            eq = TestU3Equation(a)
        else
            error("Scalar PDE '$eq_name' is not implemented.")
        end

        IC = getInitialCondition(initFunc_name, init_params)

        if grid_mover_name == "physical"
            grid_mover = PhysicalGridMover(eq)
        elseif grid_mover_name == "custom"
            func = run_params["grid_mover_func"]
            ps = run_params["grid_mover_params"]
            grid_mover = CustomGridMover(func,ps)
        elseif isnothing(grid_mover_name) || (grid_mover_name == "none")
            grid_mover = NoGridMover()
        else
            error("Only physical, custom or none grid movers supported!")
        end
        
        # --- 3. REFACTORED: Handle Analytic Solution Case Early ---
        if isnothing(timestepper_name) || timestepper_name == "Analytic"
            @info "  Computing analytical solution for a D=$dimension scalar PDE..."
            
            # Setup a temporary grid to sample the solution
            local grid_analytic, xs
            ts = tmax > 0 ? collect(0.0:(tmax/snapshots):tmax) : [0.0]

            if dimension == 1
                Nx = run_params["N"]
                grid_analytic = ParticleGrid1D(xmin, xmax, Nx , bc, 0.; rng = MersenneTwister(1))
                xs = grid_analytic.positions
                us = [[IC(x, t, eq, grid_analytic) for x in xs] for t in ts]
            else # dimension == 2
                local Nx,Ny
                if haskey(run_params, "N")
                    Nx, Ny = run_params["N"], run_params["N"]
                else
                    Nx, Ny = run_params["Nx"], run_params["Ny"]
                end
                ymin, ymax = run_params["ymin"], run_params["ymax"]
                grid_analytic = ParticleGrid2D(xmin, xmax, ymin, ymax, Nx, Ny, bc, 0.)
                xs = grid_analytic.positions
                us = [[IC(p, t, eq, grid_analytic) for p in xs] for t in ts]
            end
            sim_data = createSimData([xs for _ in ts], us, ts, run_params)
            calculateAllStats!(sim_data, (x,t) -> IC(x,t,eq,grid_analytic);quad_tol = 10e-9, dierckx_k = 3)
            sim_data.stats["time"] = 0.0
            #calculateAllStats!(sim_data, (x,t) -> IC(x,t,eq,grid_analytic); discontinuity_points_func = t -> get_discontinuity_points(IC, eq, t, grid_analytic), quad_tol = 10e-9, dierckx_k = 4)
            return sim_data
        end
        
        # --- 4. Load Remaining Numerical Parameters ---
        cfl = get(run_params, "CFL", nothing)
        dt = get(run_params, "dt", nothing)
        order = run_params["order"]
        interp_alpha = get(run_params, "interp_alpha", 1.0)
        interp_range_factor = get(run_params, "interp_range", 1.5)
        randomness_factor = get(run_params, "randomness_factor", 0.0)
        mood_name = get(run_params, "MOOD", nothing)
        mood_name2 = get(run_params, "MOOD2", nothing)
        delta_relax_factor = get(run_params, "delta_relax", 0)
        main_grad_name = get(run_params,"main_gradient",nothing)
        fallback_grad_name = get(run_params, "fallback_gradient", nothing)
        main_flux_name = get(run_params, "main_flux", nothing)
        fallback_flux_name = get(run_params, "fallback_flux", nothing)
        seed_val = get(run_params, "SEED", nothing)
        relax_velocities_config = get(run_params, "relax_velocities", nothing)
        weight_func_name = get(run_params, "weight_function", nothing)
        lim = get(run_params, "limiter", nothing)
        remove_ghosts = get(run_params, "remove_ghosts", true)
        merge_factor = get(run_params, "merge_factor", 0.)
        

        #@assert (isnothing(lim) || order == 2 || lim == "none") "Only 2nd order supported with limiter!"

        # --- 5. Grid Creation (Dimension-Aware) ---
        rng = MersenneTwister(seed_val)

        if dimension == 1
            Nx = run_params["N"]
            dx_nominal = (xmax - xmin) / Nx
            randomness = randomness_factor * dx_nominal
            
            interp_range = interp_range_factor * dx_nominal
            delta_relax = dx_nominal * delta_relax_factor
            upwind_alg_2d = "Classic"
        else # dimension == 2
            local Nx,Ny
            if haskey(run_params, "N")
                Nx, Ny = run_params["N"], run_params["N"]
            else
                Nx, Ny = run_params["Nx"], run_params["Ny"]
            end
            ymin, ymax = run_params["ymin"], run_params["ymax"]
            dx_nominal = (xmax - xmin) / Nx
            dy_nominal = (ymax - ymin) / Ny
            randomness = (randomness_factor[1] * dx_nominal, randomness_factor[2] * dy_nominal)
            interp_range = interp_range_factor * max(dx_nominal, dy_nominal)
            delta_relax = dx_nominal * dy_nominal * delta_relax_factor         
            upwind_alg_2d = main_grad_name == "Upwind" || fallback_grad_name == "Upwind" ? run_params["upwind_alg_2d"] : nothing
        end
        weight_func = if weight_func_name == "exponential"; exponentialWeightFunction(interp_alpha, interp_range)
                      elseif !isnothing(weight_func_name) error("Weight function not implemented yet!") end
        local particleGrid
        if dimension == 1
            particleGrid = ParticleGrid1D(xmin, xmax, Nx, bc, interp_range_factor; rng=rng, randomness=randomness, merge_factor = merge_factor, weight_func = weight_func)
        else
            particleGrid = ParticleGrid2D(xmin, xmax, ymin, ymax, Nx, Ny, bc, interp_range_factor; weight_func = weight_func, rng=rng, randomness=randomness)
        end
        N_total_particles = particleGrid.N
        #determineVolumes!(particleGrid)
        setInitialConditions!(particleGrid, IC)
        
        # --- 6. Time Step and Settings ---
        if !isnothing(cfl)
            # For non-linear, use a dummy linear equation with max characteristic speed
            eq_for_dt = eq isa LinearAdvection ? eq : (dimension == 1 ? LinearAdvection(1.0) : LinearAdvection((1.,1.))) # Adjust max speed for Burgers if needed
            dt = cfl * getTimeStep(particleGrid, eq_for_dt)
        elseif isnothing(dt)
            error("Either 'dt' or 'CFL' must be provided.")
        end

        save_freq = max(1, round(Int, (tmax / snapshots) / dt))
        settings = SimSetting(tmax, dt, interp_range, interp_alpha, save_freq)

        limiter = if lim == "minmod"; MinmodLimiter()
                  elseif lim == "superbee"; SuperbeeLimiter()
                  elseif lim == "VK"; VenkatakrishnanLimiter()
                  elseif lim == "BJ"; BarthJespersenLimiter()
                  elseif lim == "none" || isnothing(lim); NoLimiter()
                  else error("Limiter '$lim' not recognized") end

        # --- Build Method Components ---
        mood_fun =   if mood_name == "U2"; MOODu2(deltaRelax=delta_relax)
                     elseif mood_name == "LoubertU2"; mood_fun = MOODLoubertU2(deltaRelax = delta_relax)
                     elseif mood_name == "U1"; MOODu1(deltaRelax = delta_relax)
                     elseif mood_name == "only"; OnlyMOOD()
                     elseif mood_name == "none" || isnothing(mood_name); NoMOOD()
                     else error("MOOD '$mood_name' not recognized for 2D.")
                     end
        mood_fun2 =   if mood_name2 == "U2"; MOODu2(deltaRelax=delta_relax)
                     elseif mood_name2 == "LoubertU2"; mood_fun = MOODLoubertU2(deltaRelax = delta_relax)
                     elseif mood_name2 == "U1"; MOODu1(deltaRelax = delta_relax)
                     elseif mood_name2 == "only"; OnlyMOOD()
                     elseif mood_name2 == "none" || isnothing(mood_name2); NoMOOD()
                     else error("MOOD '$mood_name2' not recognized for 2D.")
                     end

        MainFlux = if main_flux_name == "Rusanov"; RusanovFlux()
                     elseif main_flux_name == "Upwind"; UpwindFlux()
                     elseif main_flux_name == "LW"; MainFlux = LaxWendroffFlux()
                     elseif main_grad_name!="WENO" error("Flux $main_flux_name NYI");
                     end

        FallbackFlux = if fallback_flux_name == "Rusanov"; RusanovFlux()
                         elseif fallback_flux_name == "Upwind"; UpwindFlux()
                         elseif !isnothing(fallback_flux_name); error("Fallback Flux '$fallback_flux_name' not implemented.")
                         end
        
        is_classic = timestepper_name == "LW" || timestepper_name == "Classic" || timestepper_name == "LF"
        MainGrad = if main_grad_name == "MUSCL"; MUSCL(order-1, dimension;numericalFlux = MainFlux, limiter = limiter, mood = mood_fun2)
                     elseif main_grad_name == "Upwind"; UpwindGradient(order, dimension; numericalFlux=MainFlux, algType=upwind_alg_2d)
                     elseif main_grad_name == "Central"; CentralGradient(order, dimension;)
                     elseif main_grad_name == "WENO"; WENO(order, dimension;)
                     elseif main_grad_name == "DumbserWENO"; @assert dimension == 2 "DumbserWENO can only be used for 2D, for 1D use WENO instead!"; DumbserWENO(order)
                     elseif !is_classic; error("Main Gradient '$main_grad_name' not implemented for 2D.")
                     end

        FallbackGrad = if isnothing(fallback_grad_name); NoFallbackGrad()
                         elseif fallback_grad_name == "Upwind"; UpwindGradient(1, dimension; numericalFlux=FallbackFlux, algType=upwind_alg_2d)
                         elseif !isnothing(fallback_grad_name); error("Fallback Gradient '$fallback_grad_name' not implemented for 2D.")
                         end
        
        local xs, us, ts, elapsed_time, save_relax
        if isnothing(relax_velocities_config)
            method = if timestepper_name == "RalstonRK2"; RalstonRK2(MainGrad, FallbackGrad, mood_fun, grid_mover)
            elseif timestepper_name == "EulerUpwind"; method = EulerUpwind(MainGrad, grid_mover) # Assumes EulerUpwind ignores fallback/mood args if passed
            elseif timestepper_name == "RK3"; method = RK3(MainGrad, FallbackGrad, mood_fun)
            elseif timestepper_name == "RK4"; method = RK4(MainGrad, FallbackGrad, mood_fun)
            elseif timestepper_name == "LF"; method = LaxFriedrich()
            elseif timestepper_name == "LW"; method = ClassicalRichtmyerLWMOOD(; mood = mood_fun)
            elseif timestepper_name == "Classic"; method = ClassicalTimeStepper(MainFlux)
            elseif timestepper_name == "Upwind"; method = Upwind(N_total_particles)
            elseif timestepper_name == "RalstonRK2SmoothSwitch"; method = RalstonRK2SmoothSwitch(MainGrad, FallbackGrad, mood_fun; tol =  run_params["switch_tol"])
            else error("Unknown Timestepper!") end

            save_relax = false
            # --- 8. Run Simulation ---
            elapsed_time, xs, us, ts = mainTimeIntegrator!(method, eq, particleGrid, settings; snapshots = snapshots, remove_ghosts = remove_ghosts)
            #@profview mainTimeIntegrator!(method, eq, particleGrid, settings; snapshots = snapshots)
        else
            relax_eps = run_params["relax_epsilon"]
            save_relax = run_params["save_relax"]
            N_macro_vars = 1
            N_kinetic = length(relax_velocities_config)
            # --- 4. Construct Kinetic System (Dimension-Aware) ---
            num_kinetic_per_macro::Vector{Int} = [length(v) for v in relax_velocities_config]
            N_total_kinetic = sum(num_kinetic_per_macro)
            
            kinetic_eqs_vec = Vector{LinearAdvection{dimension}}(undef, N_total_kinetic)
            SE = typeof(eq)
            M_funcs_vec = Vector{MaxwellianFunctor{dimension,N_macro_vars,SE}}(undef, N_total_kinetic)
            
            kinetic_to_macro_map = [collect(1:N_total_kinetic)]
            
            # Set Maxwellian parameters based on dimension
            coeff, int_factor = dimension == 1 ? (0.5, 1.0) : (0.25, 2.)

            global_k_idx = 1
            for i_macro in 1:N_macro_vars
                for speed in relax_velocities_config[i_macro]
                    kinetic_eqs_vec[global_k_idx] = LinearAdvection(speed)
                    
                    local i_dim::Int, relax_speed::Float64
                    if dimension == 1
                        i_dim = 1
                        relax_speed = speed
                    else # dimension == 2
                        i_dim = abs(speed[1]) > 1e-12 ? 1 : 2
                        relax_speed = speed[i_dim]
                    end

                    M_funcs_vec[global_k_idx] = MaxwellianFunctor(eq, i_macro, i_dim, relax_speed, coeff, int_factor)
                    global_k_idx += 1
                end
            end
            source_term = RelaxationSourceTerm(M_funcs_vec, relax_eps, kinetic_to_macro_map)

            # BUG FIX 2: Correctly initialize the kinetic particle grids to be in equilibrium.
            pgs_vec = [deepcopy(particleGrid) for _ in 1:N_total_kinetic]
            for k in 1:N_total_kinetic
                for p_idx in 1:pgs_vec[k].N
                    # Get the macroscopic IC at this point
                    macro_ic_at_p = particleGrid.rhos[p_idx]
                    # Set the kinetic IC to be the Maxwellian evaluated at the macro IC
                    pgs_vec[k].rhos[p_idx] = M_funcs_vec[k]((macro_ic_at_p,))
                end
            end
            pgs = ParticleGridSystem(Tuple(pgs_vec),collect(1:N_total_kinetic))
            kinetic_eqs = Tuple(kinetic_eqs_vec)

            implicit_solver = LinearizedRelaxationImplicitSolver()
            system_method = if timestepper_name == "ARS233"; ARS233(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, grid_mover)
                            elseif timestepper_name == "PRSSP3"; PareschiRussoIMEXSSP3(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, grid_mover)
                            elseif timestepper_name == "ARS222"; ARS222(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, grid_mover)
                            elseif timestepper_name == "ARS232"; ARS232(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, grid_mover)
                            elseif timestepper_name == "SimpleSplitting"; SimpleSplitting(EulerUpwind(MainGrad; fallbackInterpolator=FallbackGrad, mood=mood_fun), source_term)
                            else error("Unknown TimeStepper name for system: '$timestepper_name'") 
                            end
            elapsed_time, sys_xs, sys_us, ts = mainTimeIntegrator!(system_method, kinetic_eqs, pgs, settings; snapshots = snapshots, remove_ghosts = remove_ghosts)
            #@profview mainTimeIntegrator!(system_method, kinetic_eqs, pgs, settings; snapshots = snapshots)
            @info "System integration (D=$dimension) finished in $(round(elapsed_time, digits=2)) seconds."

            local us_final
            if save_relax
                us_final = sys_us
            else
                m = length(ts)
                # Final storage: Vector of Matrices. 
                # Structure: [TimeStep] -> [ParticleIndex, MacroVarIndex]
                us_final = Vector{Matrix{Float64}}(undef, m)
                
                for t_idx in eachindex(ts)
                    kinetic_data_at_t = sys_us[t_idx]
                    
                    # 1. FIX: Get N from the *current* snapshot, not the first one
                    N_curr = size(kinetic_data_at_t, 1)
                    
                    # 2. FIX: Allocate a Matrix (N_curr x N_macro_vars)
                    macro_data_at_t = Matrix{Float64}(undef, N_curr, N_macro_vars)
                    
                    for i_macro in 1:N_macro_vars
                        indices = kinetic_to_macro_map[i_macro]
                        
                        # 3. FIX: Sum only the columns corresponding to 'indices'
                        # view(...) creates a zero-cost slice of the specific columns
                        # sum(..., dims=2) sums across those columns
                        # vec(...) converts the resulting Nx1 matrix to a vector for assignment
                        macro_data_at_t[:, i_macro] .= vec(sum(view(kinetic_data_at_t, :, indices), dims=2))
                    end
                    us_final[t_idx] = macro_data_at_t
                end
            end

            xs = sys_xs
            us = us_final
        end
        @info "Scalar D=$dimension simulation finished in $(round(elapsed_time, digits=2)) seconds."

        sim_data_result = createSimData(xs, us, ts, run_params)
        if !save_relax
            if dimension == 1
                calculateAllStats!(sim_data_result, (x,t) -> IC(x,t,eq,particleGrid); discontinuity_points_func = t -> get_discontinuity_points(IC, eq, t, particleGrid), quad_tol = 10e-9, dierckx_k = 4)
            elseif dimension == 2
                calculateAllStats!(sim_data_result, (x,t) -> IC(x,t,eq,particleGrid); discontinuity_points_func = t -> get_discontinuity_points(IC, eq, t, particleGrid), quad_tol = 10e-9, dierckx_k = 4)
            end
        end         
        
        sim_data_result.stats["time"] = elapsed_time
        return sim_data_result

    catch e
        @error "Error during Scalar simulation!" params=params exception=(e, catch_backtrace())
        return nothing
    end
end
