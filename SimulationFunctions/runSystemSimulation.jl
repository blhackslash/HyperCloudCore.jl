using StaticArrays
using Random
using LinearAlgebra

function create_kinetic_map(num_kinetic_per_macro::Vector{<:Integer})
    cumulative_counts = [0; cumsum(num_kinetic_per_macro)]
    kinetic_to_macro_map = [
        collect((cumulative_counts[i] + 1) : cumulative_counts[i+1])
        for i in 1:length(num_kinetic_per_macro)
    ]
    return kinetic_to_macro_map
end

@inline function create_kinetic_num(relax_vel::Vector)
    return [length(v) for v in relax_vel]
end

function stable_vel_config(relax_vel::Union{Vector{Vector{Float64}}, Vector{Vector{Tuple{Float64,Float64}}}})
    return relax_vel
end

"""
    runSystemSimulation(params::ParamDictType) -> Union{AbstractSimData, Nothing}

Runs a 1D or 2D system of conservation laws using a kinetic relaxation method.
"""
function runSystemSimulation(params::ParamDictType)::Union{AbstractSimData, Nothing}
    @info "\n--- Running System Simulation (Relaxation Method) ---"
    run_params = copy(params)

    try
        # --- 1. Load Core Parameters ---
        tmax::Float64 = run_params["tmax"]
        xmin::Float64, xmax::Float64 = run_params["xmin"], run_params["xmax"]
        bc::Symbol = run_params["bc"]
        system_name::String = run_params["PDE"]
        initFunc_name::String = run_params["init_func"]
        init_params = run_params["init_params"]
        timestepper_name::String = run_params["timestepper"]
        snapshots::Int = run_params["snapshots"]
        grid_mover_name = get(run_params, "grid_mover", nothing)
        
        # --- 2. Determine Dimension and System Physics ---
        local dimension::Int
        local N_macro_vars::Int
        local system_eq::HyperbolicPDESystem
        lagrange = false
        
        if system_name == "euler1d"
            dimension = 1
            system_eq = Euler1D()
            N_macro_vars = 3 
            vel_var = 2
        elseif system_name == "leuler1d"
            @assert !isnothing(grid_mover_name) "Lagrangian Euler simulation needs grid movement!"
            dimension = 1
            system_eq = LEuler1D()
            N_macro_vars = 3 
            vel_var = 2
            lagrange = true
        elseif system_name == "euler2d"
            dimension = 2
            system_eq = Euler2D()
            N_macro_vars = 4 
            vel_var = (2,3)
        else
            error("System '$system_name' is not implemented.")
        end
        
        IC = getInitialCondition(initFunc_name, init_params)
        
        if grid_mover_name == "physical"
            # Pass the kinetic indices and the spatial dimension
            grid_mover = PhysicalGridMover(system_eq, Interpolator{dimension,1,1}(), km[vel_var], Val(dimension))
        elseif grid_mover_name == "custom"
            func = run_params["grid_mover_func"]
            ps = run_params["grid_mover_params"]
            grid_mover = CustomGridMover(func, ps)
        elseif isnothing(grid_mover_name) || (grid_mover_name == "none")
            grid_mover = NoGridMover()
        else
            error("Only physical, custom or none grid movers supported!")
        end
        
        # --- 3. Load Remaining Numerical Parameters ---
        relax_velocities_config = stable_vel_config(params["relax_velocities"])
        relax_eps::Float64 = run_params["relax_epsilon"]
        main_grad_name::String = run_params["main_gradient"]
        fallback_grad_name = get(run_params,"fallback_gradient",nothing)
        order::Int = run_params["order"]
        main_flux_name = get(run_params,"main_flux",nothing)
        fallback_flux_name = get(run_params,"fallback_flux",nothing)
        lim = get(run_params, "limiter", nothing)
        mood_name::String = run_params["MOOD"]
        delta_relax_factor = get(run_params,"delta_relax",0)
        cfl = get(run_params, "CFL", nothing)
        dt_val = get(run_params, "dt", nothing)
        interp_alpha::Float64 = run_params["interp_alpha"]
        interp_range_factor::Float64 = run_params["interp_range"]
        randomness_factor = run_params["randomness_factor"]
        seed_val = run_params["SEED"]
        weight_func_name = run_params["weight_function"]
        save_relax = run_params["save_relax"]
        remove_ghosts = get(run_params,"remove_ghosts",true)
        merge_factor = get(run_params, "merge_factor", 0.)

        @assert (isnothing(lim) || order == 2 || lim == "none") "Only 2nd order supported with limiter!"

        # --- 4. Construct Kinetic System (Dimension-Aware) ---
        num_kinetic_per_macro::Vector{Int} = [length(v) for v in relax_velocities_config]
        N_total_kinetic = sum(num_kinetic_per_macro)
        
        kinetic_eqs_vec = Vector{LinearAdvection{dimension}}(undef, N_total_kinetic)
        kinetic_to_macro_map = create_kinetic_map(num_kinetic_per_macro)
        
        int_factor = dimension == 1 ? 1.0 : 2.0
        
        # Setup Kin2Macro mapper
        summation = 1
        edges = Vector{Int}(undef, N_macro_vars + 1)
        edges[1] = 1
        for (k, kk) in enumerate(num_kinetic_per_macro)
            summation += kk
            edges[k+1] = summation 
        end 
        km = Kin2Macro(edges)
        
        local source_term
        
        if lagrange
            relax_speeds = Vector{Float64}(undef, N_total_kinetic)
            global_k_idx = 1
            for i_macro in 1:N_macro_vars
                for speed in relax_velocities_config[i_macro] 
                    kinetic_eqs_vec[global_k_idx] = LinearAdvection(speed)
                    
                    if dimension == 1
                        relax_speed = speed
                    else 
                        relax_speed = abs(speed[1]) > abs(speed[2]) ? speed[1] : speed[2]
                        if abs(relax_speed) < 1e-14
                            error("Relaxation speed for 2D velocity $speed is zero.")
                        end
                    end
                    relax_speeds[global_k_idx] = relax_speed
                    global_k_idx += 1
                end
            end

            coeffs = Tuple(map(x -> 1/x, num_kinetic_per_macro))
            source_term = NonLocalRelaxationSourceTerm(
                system_eq, relax_eps, km, coeffs, Tuple(relax_speeds), int_factor
            )
        else
            # Pre-allocate parameter buffers for Local Relaxation
            coeffs_k = Float64[]
            speeds_k = Float64[]
            dims_k   = Int[]
            ints_k   = Float64[]
            
            global_k_idx = 1
            for i_macro in 1:N_macro_vars
                coeff = 1.0 / num_kinetic_per_macro[i_macro]
                for speed in relax_velocities_config[i_macro]
                    kinetic_eqs_vec[global_k_idx] = LinearAdvection(speed)
                    
                    local i_dim::Int, relax_speed::Float64
                    if dimension == 1
                        i_dim = 1
                        relax_speed = speed
                    else 
                        if abs(speed[1]) > abs(speed[2])
                            i_dim = 1
                            relax_speed = speed[1]
                        else
                            i_dim = 2
                            relax_speed = speed[2]
                        end
                        if abs(relax_speed) < 1e-14
                            error("Relaxation speed for 2D velocity $speed is zero.")
                        end
                    end

                    push!(coeffs_k, coeff)
                    push!(speeds_k, relax_speed)
                    push!(dims_k, i_dim)
                    push!(ints_k, int_factor)
                    
                    global_k_idx += 1
                end
            end
            
            # Construct the new flat RelaxationSourceTerm
            source_term = RelaxationSourceTerm(
                system_eq, relax_eps, km, 
                Tuple(coeffs_k), Tuple(speeds_k), Tuple(ints_k), Tuple(dims_k)
            )
        end
        
        # --- 5. Grid & Initial Condition Setup ---
        rng = MersenneTwister(seed_val)
        
        if dimension == 1
            Nx = run_params["N"]
            dx_nominal = (xmax - xmin) / Nx
            randomness = randomness_factor * dx_nominal
            interp_range = interp_range_factor * dx_nominal
            delta_relax = dx_nominal * delta_relax_factor
            upwind_alg_2d = "Classic"
        else 
            Nx, Ny = run_params["Nx"], run_params["Ny"]
            ymin, ymax = run_params["ymin"], run_params["ymax"]
            dx_nominal = (xmax - xmin) / Nx
            dy_nominal = (ymax - ymin) / Ny
            randomness = (randomness_factor[1] * dx_nominal, randomness_factor[2] * dy_nominal)
            interp_range = interp_range_factor * max(dx_nominal, dy_nominal)
            delta_relax = dx_nominal * dy_nominal * delta_relax_factor         
            upwind_alg_2d = main_grad_name == "Upwind" || fallback_grad_name == "Upwind" ? run_params["upwind_alg_2d"] : nothing
        end
        
        weight_func = if weight_func_name == "exponential"; exponentialWeightFunction(interp_alpha, interp_range)
                      else error("Weight function not implemented yet!") end
        
        # Replaced ParticleGridSystem loop with a single createParticleGrid Call
        local pg
        if dimension == 1
            pg = createParticleGrid(Val(1), xmin, xmax, Nx, bc, interp_range_factor; M=N_total_kinetic, rng=rng, merge_factor=merge_factor, randomness=randomness, weight_func=weight_func)
        else
            pg = createParticleGrid(Val(2), xmin, xmax, ymin, ymax, Nx, Ny, bc, interp_range_factor; M=N_total_kinetic, weight_func=weight_func, rng=rng, randomness=randomness)
        end        

        # --- 6. Time Step Calculation ---
        local dt::Float64
        if !isnothing(cfl)
            max_abs_speed = 0.0
            for group in relax_velocities_config
                for s in group
                    max_abs_speed = max(max_abs_speed, norm(s))
                end
            end
            if max_abs_speed < 1e-9; max_abs_speed = 1.0; end
            
            temp_eq_for_dt = dimension == 1 ? LinearAdvection(max_abs_speed) : LinearAdvection((max_abs_speed, max_abs_speed))
            dt = cfl * getTimeStep(pg, temp_eq_for_dt)
        else
            dt = dt_val
        end
        
        save_freq = max(1, round(Int, (tmax / snapshots) / dt))
        settings = SimSetting(tmax, dt, interp_range, interp_alpha, save_freq)
        
        # --- 7. Build Numerical Method & Run Simulation ---
        local mood_fun
        if mood_name == "U1"; mood_fun = MOODu1(deltaRelax = delta_relax)
        elseif mood_name == "U2"; mood_fun = MOODu2(deltaRelax = delta_relax)
        elseif mood_name == "LoubertU2"; mood_fun = MOODLoubertU2(deltaRelax = delta_relax)
        elseif mood_name == "none" || isnothing(mood_name); mood_fun = NoMOOD()
        elseif mood_name == "only"; mood_fun = OnlyMOOD()
        else error("MOOD '$mood_name' not recognized") end

        local limiter
        if !isnothing(lim)
            @assert (main_grad_name == "MUSCL") "Slope limiter only supported for MUSCL-schemes!"
            @assert (order == 2) "Only linear reconstruction supported at the moment!"
        end 
        
        limiter = if lim == "minmod"; MinmodLimiter()
                  elseif lim == "superbee"; SuperbeeLimiter()
                  elseif lim == "VK"; VenkatakrishnanLimiter()
                  elseif lim == "BJ"; BarthJespersenLimiter()
                  elseif lim == "none" || isnothing(lim); NoLimiter()
                  else error("Limiter '$lim' not recognized") end

        MainFlux = if main_flux_name == "Rusanov"; RusanovFlux() 
                   elseif main_flux_name == "Upwind"; UpwindFlux()
                   elseif main_grad_name != "WENO" error("Flux $main_flux_name NYI") end
            
        FallbackFlux = if fallback_flux_name == "Rusanov"; RusanovFlux() 
                       elseif fallback_flux_name == "Upwind"; UpwindFlux()
                       elseif !isnothing(fallback_flux_name); error("Flux $fallback_flux_name NYI") end
                       
        local upwind_alg_2d
        if main_grad_name == "Upwind" || !isa(mood_fun, NoMOOD) || fallback_grad_name == "Upwind"
            upwind_alg_2d = dimension == 2 ? run_params["upwind_alg_2d"] : "Classic"
        else
            upwind_alg_2d = "nothing"
        end
        
        MainGrad = if main_grad_name == "MUSCL"; MUSCL(order-1, dimension; numericalFlux = MainFlux, limiter = limiter)
                   elseif main_grad_name == "WENO"; WENO(order, dimension)
                   elseif main_grad_name == "Upwind"; UpwindGradient(order, dimension; numericalFlux=MainFlux, algType=upwind_alg_2d)
                   else error("Unknown MainGrad: $main_grad_name") end
                   
        FallbackGrad = if fallback_grad_name == "Upwind"; UpwindGradient(1, dimension; numericalFlux=FallbackFlux, algType=upwind_alg_2d)
                       elseif isnothing(fallback_grad_name); NoFallbackGrad()
                       else error("Only Upwind implemented as Fallback!") end
                       
        implicit_solver = LinearizedRelaxationImplicitSolver()
        
        system_method = if timestepper_name == "ARS233"; ARS233(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, grid_mover)
                        elseif timestepper_name == "PRSSP3"; PareschiRussoIMEXSSP3(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, grid_mover)
                        elseif timestepper_name == "ARS222"; ARS222(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, grid_mover)
                        elseif timestepper_name == "ARS232"; ARS232(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, grid_mover)
                        else error("Unknown TimeStepper name for system: '$timestepper_name'") end
        
        kinetic_eqs = Tuple(kinetic_eqs_vec)

        # Note: You will need to update setInitialConditions! to accept a single ParticleGrid
        setInitialConditions!(pg, source_term, IC)
        
        elapsed_time, xs_data, sys_us_kinetic, ts = mainTimeIntegrator!(system_method, kinetic_eqs, pg, settings; snapshots = snapshots, remove_ghosts = remove_ghosts)
        
        @info "System integration (D=$dimension) finished in $(round(elapsed_time, digits=2)) seconds."
        
        # --- 8. Post-process & Return ---
        local us_final
        if save_relax
            us_final = sys_us_kinetic
        else
            m = length(ts)
            us_final = Vector{Matrix{Float64}}(undef, m)
            
            for t_idx in eachindex(ts)
                kinetic_data_at_t = sys_us_kinetic[t_idx]
                macro_data_at_t = Matrix{Float64}(undef, size(kinetic_data_at_t, 1), N_macro_vars)
                
                for i_macro in 1:N_macro_vars
                    indices = kinetic_to_macro_map[i_macro]
                    # Direct column summation
                    macro_data_at_t[:, i_macro] .= 0.0
                    for idx in indices
                        macro_data_at_t[:, i_macro] .+= kinetic_data_at_t[:, idx]
                    end
                end
                us_final[t_idx] = macro_data_at_t
            end
        end

        sim_data_result = createSimData(xs_data, us_final, ts, run_params)
        sim_data_result.stats["time"] = elapsed_time
        return sim_data_result

    catch e
        @error "Error during System simulation!" params=params exception=(e, catch_backtrace())
        return nothing
    end
end