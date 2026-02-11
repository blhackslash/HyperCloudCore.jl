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
This is the general-purpose runner for systems like Euler equations.
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

        if system_name == "euler1d"
            dimension = 1
            system_eq = Euler1D()
            N_macro_vars = 3 # rho, m, E
            vel_var = 2
        elseif system_name == "euler2d"
            dimension = 2
            system_eq = Euler2D()
            N_macro_vars = 4 # rho, mx, my, E
            vel_var = (2,3)
        else
            error("System '$system_name' is not implemented.")
        end
        
        # --- REFACTORED: Handle Analytic Solution Case Early ---
        IC = getInitialCondition(initFunc_name, init_params)
        if grid_mover_name == "physical"
            grid_mover = PhysicalGridMover(system_eq,Interpolator{dimension,1,1}())
        elseif grid_mover_name == "custom"
            func = run_params["grid_mover_func"]
            ps = run_params["grid_mover_params"]
            grid_mover = CustomGridMover(func,ps)
        elseif isnothing(grid_mover_name) || (grid_mover_name == "none")
            grid_mover = NoGridMover()
        else
            error("Only physical, custom or none grid movers supported!")
        end
        if timestepper_name == "Analytic"
            @info "  Computing analytical solution..."
            # Setup a temporary grid to sample the solution
            grid_analytic = if dimension == 1
                ParticleGrid1D(xmin, xmax, run_params["N"], bc, rng=MersenneTwister(1), bc != :periodic)
            else
                ParticleGrid2D(xmin, xmax, run_params["ymin"], run_params["ymax"], run_params["Nx"], run_params["Ny"], bc, rng=rng=MersenneTwister(1), bc != :periodic)
            end
            
            dt_analytic = tmax / snapshots
            ts = collect(0:dt_analytic:tmax)
            if ts[end] != tmax; push!(ts,tmax) end
            xs = [grid_analytic.positions for _ in ts]
            us = Vector{Matrix{Float64}}(undef, length(ts))

            for (i, t) in enumerate(ts)
                us[i] = hcat([collect(IC(x, t, system_eq, grid_analytic)) for x in xs[i]]...)'
            end
            return createSimData(xs, us, ts, run_params)
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
        SE = typeof(system_eq)
        M_funcs_vec = Vector{MaxwellianFunctor{dimension,N_macro_vars,SE}}(undef, N_total_kinetic)
        
        kinetic_to_macro_map = create_kinetic_map(num_kinetic_per_macro)
        
        # Set Maxwellian parameters based on dimension
        int_factor = dimension == 1 ? 1.0 :  2.

        global_k_idx = 1
        for i_macro in 1:N_macro_vars
            coeff = 1 / num_kinetic_per_macro[i_macro]
            for speed in relax_velocities_config[i_macro]
                kinetic_eqs_vec[global_k_idx] = LinearAdvection(speed)
                
                # In runSystemSimulation.jl and runScalarSimulation.jl
                local i_dim::Int, relax_speed::Float64

                if dimension == 1
                    i_dim = 1
                    relax_speed = speed
                else # dimension == 2
                    abs_vx = abs(speed[1])
                    abs_vy = abs(speed[2])

                    if abs_vx > abs_vy
                        i_dim = 1
                        relax_speed = speed[1]
                    else
                        i_dim = 2
                        relax_speed = speed[2]
                    end
                    
                    # Critical check to prevent division by zero
                    if abs(relax_speed) < 1e-14
                        # If speed is (0,0), relax_speed is 0. We must handle this.
                        # Option 1: Error out
                        error("Relaxation speed for 2D velocity $speed is zero.")
                    end
                end

                M_funcs_vec[global_k_idx] = MaxwellianFunctor(system_eq, i_macro, i_dim, relax_speed, coeff, int_factor)
                global_k_idx += 1
            end
        end
        source_term = RelaxationSourceTerm(M_funcs_vec, relax_eps, kinetic_to_macro_map)
        
        # --- 5. Grid & Initial Condition Setup (Dimension-Aware) ---
        N_ghost = bc == :periodic ? 0 : ceil(Int, interp_range_factor) + 1
        rng = MersenneTwister(seed_val)
        
        if dimension == 1
            Nx = run_params["N"]
            dx_nominal = (xmax - xmin) / Nx
            randomness = randomness_factor * dx_nominal
            
            interp_range = interp_range_factor * dx_nominal
            delta_relax = dx_nominal * delta_relax_factor
            upwind_alg_2d = "Classic"
        else # dimension == 2
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
                      elseif !isnothing(weight_func_name) error("Weight function not implemented yet!") end
        local particleGrid_template
        if dimension == 1
            particleGrid_template = ParticleGrid1D(xmin, xmax, Nx, bc, interp_range_factor; rng=rng, merge_factor = merge_factor, randomness=randomness, weight_func = weight_func)
        else
            particleGrid_template = ParticleGrid2D(xmin, xmax, ymin, ymax, Nx, Ny, bc, interp_range_factor; weight_func = weight_func, rng=rng, randomness=randomness)
        end        
        # --- REFACTORED: Initial Condition Setup for System ---

        # 1. Calculate the macroscopic initial condition at all particle positions (including ghosts).
        #    This creates a vector of tuples, e.g., [(rho,m,E)_1, (rho,m,E)_2, ...].
        #macro_ic_at_points = [IC(pos...) for pos in particleGrid_template.positions]

        # 2. Create the tuple of particle grids for each kinetic component.
        PG = typeof(particleGrid_template)
        particleGrids_vec::Vector{PG} = [deepcopy(particleGrid_template) for _ in 1:N_total_kinetic]

        setInitialConditions!(particleGrids_vec, M_funcs_vec, IC)
        
        # --- 6. Time Step Calculation (Dimension-Aware) ---
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
            dt = cfl * getTimeStep(particleGrid_template, temp_eq_for_dt)
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

        local weight_func
        weight_func = if weight_func_name == "exponential"; exponentialWeightFunction(interp_alpha, interp_range)
                      else error("Weight function not implemented yet!") end

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
            elseif main_flux_name == "Upwind"; MainFlux = UpwindFlux()
            elseif main_grad_name!="WENO" error("Flux $main_flux_name NYI");
            end
            
            
        FallbackFlux = if fallback_flux_name == "Rusanov"; RusanovFlux() 
                       elseif fallback_flux_name == "Upwind"; FallbackFlux = UpwindFlux()
                       elseif !isnothing(fallback_flux_name); error("Flux $fallback_flux_name NYI"); end
        local upwind_alg_2d
        if main_grad_name == "Upwind" || !isa(mood_fun, NoMOOD) || fallback_grad_name == "Upwind"
            upwind_alg_2d = dimension == 2 ? run_params["upwind_alg_2d"] : "Classic"
        else
            upwind_alg_2d = "nothing"
        end
        MainGrad = if main_grad_name == "MUSCL"
                    MUSCL(order-1, dimension; numericalFlux = MainFlux, limiter = limiter)
                    elseif main_grad_name == "WENO"
                        WENO(order, dimension;)
                elseif main_grad_name == "Upwind"
                    UpwindGradient(order, dimension; numericalFlux=MainFlux, algType=upwind_alg_2d)
                else error("Unknown MainGrad: $main_grad_name"); end
        FallbackGrad = if fallback_grad_name == "Upwind" UpwindGradient(1, dimension; numericalFlux=FallbackFlux, algType=upwind_alg_2d)
                        elseif isnothing(fallback_grad_name) NoFallbackGrad()
                       else error("Only Upwind implemented as Fallback!") end
        implicit_solver = LinearizedRelaxationImplicitSolver()
        
        system_method = if timestepper_name == "ARS233"; ARS233(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, grid_mover)
                        elseif timestepper_name == "PRSSP3"; PareschiRussoIMEXSSP3(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, grid_mover)
                        elseif timestepper_name == "ARS222"; ARS222(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, grid_mover)
                        elseif timestepper_name == "ARS232"; ARS232(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, grid_mover)
                        elseif timestepper_name == "SimpleSplitting"; SimpleSplitting(RalstonRK2(MainGrad; fallbackInterpolator=FallbackGrad, mood=mood_fun), source_term)
                        else error("Unknown TimeStepper name for system: '$timestepper_name'") end
        
        # Convert to tuples for performance before passing to the integrator
        kinetic_eqs = Tuple(kinetic_eqs_vec)
        particleGrids = ParticleGridSystem(Tuple(particleGrids_vec),kinetic_to_macro_map)

        elapsed_time, xs_data, sys_us_kinetic, ts = mainTimeIntegrator!(system_method, kinetic_eqs, particleGrids, settings; snapshots = snapshots, remove_ghosts = remove_ghosts)
        @info "System integration (D=$dimension) finished in $(round(elapsed_time, digits=2)) seconds."

        # --- 8. Post-process & Return ---
        
        local us_final
        if save_relax
            us_final = sys_us_kinetic
        else
            m = length(ts)
            # Pre-allocate the final macroscopic solution array
            us_final = Vector{Matrix{Float64}}(undef, m)
            
            # Use an efficient loop instead of `map`
            for t_idx in eachindex(ts)
                kinetic_data_at_t = sys_us_kinetic[t_idx]
                # Pre-allocate the matrix for this time step
                macro_data_at_t = similar(kinetic_data_at_t, size(kinetic_data_at_t, 1), N_macro_vars)
                
                for i_macro in 1:N_macro_vars
                    indices = kinetic_to_macro_map[i_macro]
                    # Sum the relevant columns directly into the output matrix without intermediate allocations
                    sum!(@view(macro_data_at_t[:, i_macro]), @view(kinetic_data_at_t[:, indices]))
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
