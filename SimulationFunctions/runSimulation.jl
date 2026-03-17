using Random
using LinearAlgebra
using StaticArrays

# ==============================================================================
# HELPER UTILITIES
# ==============================================================================

function create_kinetic_map(num_kinetic_per_macro::Vector{<:Integer})
    cumulative_counts = [0; cumsum(num_kinetic_per_macro)]
    return [collect((cumulative_counts[i] + 1) : cumulative_counts[i+1]) for i in 1:length(num_kinetic_per_macro)]
end

@inline function create_kinetic_num(relax_vel::Vector)
    return [length(v) for v in relax_vel]
end

function stable_vel_config(relax_vel::Union{Vector{Vector{Float64}}, Vector{Vector{Tuple{Float64,Float64}}}})
    return relax_vel
end

# ==============================================================================
# HELPER FUNCTION BARRIERS
# These @noinline functions break the Type-Inference loop. 
# Inside them, all types (method, eq, pg) are strictly known, ensuring C-speed.
# ==============================================================================

@noinline function _execute_analytic_sim(IC, eq, grid_analytic, tmax, snapshots, run_params)
    ts = tmax > 0 ? collect(0.0:(tmax/snapshots):tmax) : [0.0]
    xs = grid_analytic.positions
    us = [[IC(p, t, eq, grid_analytic) for p in xs] for t in ts]

    sim_data = createSimData([xs for _ in ts], us, ts, run_params)
    sim_data.stats["time"] = 0.0
    return sim_data
end

@noinline function _execute_explicit_sim!(method, eq, pg, settings, run_params, dimension, snapshots, remove_ghosts, M_components)
    elapsed_time, xs, us_svector, ts, k_step = mainTimeIntegrator!(method, eq, pg, settings; snapshots = snapshots, remove_ghosts = remove_ghosts)
    @info "Explicit Simulation (D=$dimension) finished in $(round(elapsed_time, digits=2)) seconds."

    # --- Transform SVector arrays back into Matrices for plotting/saving ---
    m = length(ts)
    us_final = Vector{Matrix{Float64}}(undef, m)
    
    for t_idx in 1:m
        N_particles = length(us_svector[t_idx])
        mat = Matrix{Float64}(undef, N_particles, M_components)
        
        # Fast column-major extraction
        for c in 1:M_components
            for i in 1:N_particles
                mat[i, c] = us_svector[t_idx][i][c]
            end
        end
        us_final[t_idx] = mat
    end

    sim_data_result = createSimData(xs, us_final, ts, run_params)
    sim_data_result.stats["time"] = elapsed_time
    sim_data_result.stats["k_step"] = k_step
    return sim_data_result
end

@noinline function _execute_kinetic_sim!(system_method, kinetic_eqs, pg, settings, run_params, dimension, snapshots, remove_ghosts, save_relax, N_macro_vars, kinetic_to_macro_map)
    elapsed_time, xs_data, sys_us_kinetic, ts = mainTimeIntegrator!(system_method, kinetic_eqs, pg, settings; snapshots = snapshots, remove_ghosts = remove_ghosts)
    @info "Kinetic Relaxation Simulation (D=$dimension) finished in $(round(elapsed_time, digits=2)) seconds."

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
end

# ==============================================================================
# MAIN SIMULATION BUILDER
# ==============================================================================

"""
    runSimulation(params::ParamDictType) -> Union{AbstractSimData, Nothing}

Unified function to run 1D/2D Scalar and System conservation laws.
"""
function runSimulation(params::ParamDictType)::Union{AbstractSimData, Nothing}
    @info "\n--- Running General Simulation ---"
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
        
        relax_velocities_config = get(run_params, "relax_velocities", nothing)
        is_kinetic = !isnothing(relax_velocities_config)
        
        # --- 2. Determine Dimension and PDE Physics ---
        local dimension::Int
        local N_macro_vars::Int
        local eq
        local vel_var = nothing
        lagrange = false

        if eq_name == "linear"
            pde_params = run_params["PDE_params"]
            if pde_params isa Real
                dimension = 1
                eq = LinearAdvection(pde_params)
            else
                dimension = 2
                eq = LinearAdvection(Tuple(pde_params))
            end
            N_macro_vars = 1
            vel_var = 1
        elseif eq_name == "burgers"
            dimension = 1
            eq = BurgersEquation(get(run_params,"PDE_params", 0.))
            N_macro_vars = 1
            vel_var = 1
        elseif eq_name == "burgers2d"
            dimension = 2
            eq = BurgersEquation2D()
            N_macro_vars = 1
            vel_var = 1
        elseif eq_name == "testU3"
            dimension = 1
            eq = TestU3Equation(get(run_params,"PDE_params", 0.))
            N_macro_vars = 1
            vel_var = 1
        elseif eq_name == "euler1d"
            dimension = 1
            eq = Euler1D()
            N_macro_vars = 3 
            vel_var = 2
        elseif eq_name == "leuler1d"
            @assert !isnothing(grid_mover_name) "Lagrangian Euler simulation needs grid movement!"
            dimension = 1
            eq = LEuler1D()
            N_macro_vars = 3 
            vel_var = 2
            lagrange = true
        elseif eq_name == "euler2d"
            dimension = 2
            eq = Euler2D()
            N_macro_vars = 4 
            vel_var = (2,3)
        else
            error("PDE '$eq_name' is not implemented.")
        end

        IC = getInitialCondition(initFunc_name, init_params)

        # --- 3. Set Up Kinetic Components & Source Term (If applicable) ---
        local M_components::Int
        local km = nothing
        local kinetic_to_macro_map = nothing
        local kinetic_eqs_vec = nothing
        local source_term = nothing
        
        if is_kinetic
            relax_eps::Float64 = run_params["relax_epsilon"]
            relax_velocities_config = stable_vel_config(relax_velocities_config)
            num_kinetic_per_macro::Vector{Int} = [length(v) for v in relax_velocities_config]
            M_components = sum(num_kinetic_per_macro)
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
            kinetic_eqs_vec = Vector{LinearAdvection{dimension}}(undef, M_components)
            
            if lagrange
                relax_speeds = Vector{Float64}(undef, M_components)
                global_k_idx = 1
                for i_macro in 1:N_macro_vars
                    for speed in relax_velocities_config[i_macro] 
                        kinetic_eqs_vec[global_k_idx] = LinearAdvection(speed)
                        relax_speed = dimension == 1 ? speed : (abs(speed[1]) > abs(speed[2]) ? speed[1] : speed[2])
                        if abs(relax_speed) < 1e-14; error("Relaxation speed is zero."); end
                        relax_speeds[global_k_idx] = relax_speed
                        global_k_idx += 1
                    end
                end
                coeffs = Tuple(map(x -> 1/x, num_kinetic_per_macro))
                source_term = NonLocalRelaxationSourceTerm(eq, relax_eps, km, coeffs, Tuple(relax_speeds), int_factor)
            else
                coeffs_k, speeds_k, dims_k, ints_k = Float64[], Float64[], Int[], Float64[]
                global_k_idx = 1
                for i_macro in 1:N_macro_vars
                    coeff = 1.0 / num_kinetic_per_macro[i_macro]
                    for speed in relax_velocities_config[i_macro]
                        kinetic_eqs_vec[global_k_idx] = LinearAdvection(speed)
                        local i_dim::Int, relax_speed::Float64
                        if dimension == 1
                            i_dim, relax_speed = 1, speed
                        else 
                            i_dim = abs(speed[1]) > abs(speed[2]) ? 1 : 2
                            relax_speed = speed[i_dim]
                        end
                        push!(coeffs_k, coeff); push!(speeds_k, relax_speed); push!(dims_k, i_dim); push!(ints_k, int_factor)
                        global_k_idx += 1
                    end
                end
                source_term = RelaxationSourceTerm(eq, relax_eps, km, Tuple(coeffs_k), Tuple(speeds_k), Tuple(ints_k), Tuple(dims_k))
            end
        else
            km = Kin2Macro(1:N_macro_vars)
            M_components = N_macro_vars
        end

        # --- 4. Setup Grid Mover ---
        if grid_mover_name == "physical"
            if is_kinetic && !isnothing(vel_var)
                grid_mover = PhysicalGridMover(eq, Interpolator{dimension,1,1}(), km[vel_var], Val(dimension))
            else
                grid_mover = PhysicalGridMover(eq, Interpolator{dimension,1,1}())
            end
        elseif grid_mover_name == "custom"
            grid_mover = CustomGridMover(run_params["grid_mover_func"], run_params["grid_mover_params"])
        elseif isnothing(grid_mover_name) || (grid_mover_name == "none")
            grid_mover = NoGridMover()
        else
            error("Only physical, custom or none grid movers supported!")
        end

        # --- 5. Handle Analytic Case (Standard explicitly only) ---
        if !is_kinetic && (isnothing(timestepper_name) || timestepper_name == "Analytic")
            @info "  Computing analytical solution for a D=$dimension PDE..."
            local grid_analytic
            if dimension == 1
                grid_analytic = createParticleGrid(Val(1), xmin, xmax, run_params["N"] , bc, 0.; rng = MersenneTwister(1))
            else 
                Nx, Ny = haskey(run_params, "N") ? (run_params["N"], run_params["N"]) : (run_params["Nx"], run_params["Ny"])
                grid_analytic = createParticleGrid(Val(1), xmin, xmax, run_params["ymin"], run_params["ymax"], Nx, Ny, bc, 0.)
            end
            return _execute_analytic_sim(IC, eq, grid_analytic, tmax, snapshots, run_params)
        end

        # --- 6. Load Remaining Parameters & Create Grid ---
        cfl = get(run_params, "CFL", nothing)
        dt_val = get(run_params, "dt", nothing)
        order = run_params["order"]
        interp_alpha = get(run_params, "interp_alpha", 1.0)
        interp_range_factor = get(run_params, "interp_range", 1.5)
        randomness_factor = get(run_params, "randomness_factor", 0.0)
        mood_name = get(run_params, "MOOD", nothing)
        mood_name2 = get(run_params, "MOOD2", nothing)
        delta_relax_factor = get(run_params, "delta_relax", 0.0)
        main_grad_name = get(run_params, "main_gradient", nothing)
        fallback_grad_name = get(run_params, "fallback_gradient", nothing)
        main_flux_name = get(run_params, "main_flux", nothing)
        fallback_flux_name = get(run_params, "fallback_flux", nothing)
        seed_val = get(run_params, "SEED", nothing)
        weight_func_name = get(run_params, "weight_function", nothing)
        lim = get(run_params, "limiter", nothing)
        remove_ghosts = get(run_params, "remove_ghosts", true)
        merge_factor = get(run_params, "merge_factor", 0.0)
        
        rng = MersenneTwister(seed_val)
        local pg
        local delta_relax, interp_range, upwind_alg_2d

        if dimension == 1
            Nx = run_params["N"]
            dx_nom = (xmax - xmin) / Nx
            interp_range = interp_range_factor * dx_nom
            delta_relax = dx_nom * delta_relax_factor
            upwind_alg_2d = "Classic"
            weight_func = exponentialWeightFunction(interp_alpha, interp_range)
            
            pg = createParticleGrid(Val(1), xmin, xmax, Nx, bc, interp_range_factor; M=M_components, rng=rng, randomness=(randomness_factor * dx_nom), merge_factor=merge_factor, weight_func=weight_func, km = km, mover = grid_mover)
        else
            Nx, Ny = haskey(run_params, "N") ? (run_params["N"], run_params["N"]) : (run_params["Nx"], run_params["Ny"])
            ymin, ymax = run_params["ymin"], run_params["ymax"]
            dx_nom, dy_nom = (xmax - xmin)/Nx, (ymax - ymin)/Ny
            interp_range = interp_range_factor * max(dx_nom, dy_nom)
            delta_relax = dx_nom * dy_nom * delta_relax_factor         
            upwind_alg_2d = (main_grad_name == "Upwind" || fallback_grad_name == "Upwind") ? run_params["upwind_alg_2d"] : nothing
            weight_func = exponentialWeightFunction(interp_alpha, interp_range)
            
            pg = createParticleGrid(Val(2), xmin, xmax, ymin, ymax, Nx, Ny, bc, interp_range_factor; M=M_components, weight_func=weight_func, rng=rng, randomness=(randomness_factor[1]*dx_nom, randomness_factor[2]*dy_nom), km = km, mover = grid_mover)
        end

        # --- 7. Time Step Calculation ---
        local dt::Float64
        if !isnothing(cfl)
            if is_kinetic
                max_abs_speed = maximum([norm(s) for group in relax_velocities_config for s in group])
                if max_abs_speed < 1e-9; max_abs_speed = 1.0; end
                temp_eq_dt = dimension == 1 ? LinearAdvection(max_abs_speed) : LinearAdvection((max_abs_speed, max_abs_speed))
                dt = cfl * getTimeStep(pg, temp_eq_dt)
            else
                eq_for_dt = eq isa LinearAdvection ? eq : (dimension == 1 ? LinearAdvection(1.0) : LinearAdvection((1.,1.))) 
                dt = cfl * getTimeStep(pg, eq_for_dt)
            end
        else
            dt = dt_val
        end
        save_freq = max(1, round(Int, (tmax / snapshots) / dt))
        settings = SimSetting(tmax, dt, interp_range, interp_alpha, save_freq)

        # --- 8. Build Interpolators and Limiters ---
        limiter = if lim == "minmod"
            MinmodLimiter()
        elseif lim == "superbee"
            SuperbeeLimiter()
        elseif lim == "VK"
            VenkatakrishnanLimiter()
        elseif lim == "BJ"
            BarthJespersenLimiter()
        elseif lim == "none" || isnothing(lim)
            NoLimiter()
        else
            error("Limiter '$lim' not recognized")
        end
        
        mood_fun = if mood_name == "U2"
            MOODu2(deltaRelax=delta_relax)
        elseif mood_name == "LoubertU2"
            MOODLoubertU2(deltaRelax=delta_relax)
        elseif mood_name == "U1"
            MOODu1(deltaRelax=delta_relax)
        elseif mood_name == "only"
            OnlyMOOD()
        elseif mood_name == "none" || isnothing(mood_name)
            NoMOOD()
        else
            error("MOOD not recognized.")
        end

        mood_fun2 = if mood_name2 == "U2"
            MOODu2(deltaRelax=delta_relax)
        elseif mood_name2 == "LoubertU2"
            MOODLoubertU2(deltaRelax=delta_relax)
        elseif mood_name2 == "U1"
            MOODu1(deltaRelax=delta_relax)
        elseif mood_name2 == "only"
            OnlyMOOD()
        elseif mood_name2 == "none" || isnothing(mood_name2)
            NoMOOD()
        else
            error("MOOD2 not recognized.")
        end

        MainFlux = if main_flux_name == "Rusanov"
            RusanovFlux()
        elseif main_flux_name == "Upwind"
            UpwindFlux()
        elseif main_flux_name == "LW"
            LaxWendroffFlux()
        elseif main_grad_name != "WENO" 
            error("Flux $main_flux_name NYI")
        end

        FallbackFlux = if fallback_flux_name == "Rusanov"
            RusanovFlux()
        elseif fallback_flux_name == "Upwind"
            UpwindFlux()
        elseif !isnothing(fallback_flux_name)
            error("Fallback Flux NYI")
        end
        
        MainGrad = if main_grad_name == "MUSCL"
            MUSCL(order-1, dimension; numericalFlux = MainFlux, limiter = limiter, mood = mood_fun2)
        elseif main_grad_name == "Upwind"
            UpwindGradient(order, dimension; numericalFlux=MainFlux, algType=upwind_alg_2d)
        elseif main_grad_name == "Central"
            CentralGradient(order, dimension)
        elseif main_grad_name == "WENO"
            WENO(order, dimension)
        elseif main_grad_name == "DumbserWENO"
            DumbserWENO(order)
        elseif !(timestepper_name in ["LW", "Classic", "LF"])
            error("Main Gradient NYI")
        end

        FallbackGrad = if isnothing(fallback_grad_name)
            NoFallbackGrad()
        elseif fallback_grad_name == "Upwind"
            UpwindGradient(1, dimension; numericalFlux=FallbackFlux, algType=upwind_alg_2d)
        else
            error("Fallback Gradient NYI")
        end

        # --- 9. Final Execution Dispatch ---
if !is_kinetic
            # We determine the exact state type based on the number of macro variables
            state_type = SVector{M_components, Float64}
            
            if timestepper_name == "EulerUpwind"
                method = EulerUpwind(eq, MainGrad, FallbackGrad, mood_fun, state_type)
            elseif timestepper_name == "RalstonRK2"
                method = RalstonRK2(eq, MainGrad, FallbackGrad, mood_fun, state_type)
            elseif timestepper_name == "RK3"
                method = RK3(eq, MainGrad, FallbackGrad, mood_fun, state_type)
            elseif timestepper_name == "RK4"
                method = RK4(eq, MainGrad, FallbackGrad, mood_fun, state_type)
            else
                error("Unknown direct TimeStepper: '$timestepper_name'")
            end

            # Note: Ensure your setInitialConditions! is updated to write SVectors!
            setInitialConditions!(pg, eq, IC)
            
            return _execute_explicit_sim!(method, eq, pg, settings, run_params, dimension, snapshots, remove_ghosts, M_components)
        else
            error("Kinetic / IMEX simulations are temporarily disabled for refactoring.")
        end
        # else
        #     implicit_solver = LinearizedRelaxationImplicitSolver()
            
        #     system_method = if timestepper_name == "ARS233"
        #         ARS233(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, grid_mover)
        #     elseif timestepper_name == "PRSSP3"
        #         PareschiRussoIMEXSSP3(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, grid_mover)
        #     elseif timestepper_name == "ARS222"
        #         ARS222(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, grid_mover)
        #     elseif timestepper_name == "ARS232"
        #         ARS232(MainGrad, FallbackGrad, mood_fun, implicit_solver, source_term, grid_mover)
        #     elseif timestepper_name == "SimpleSplitting"
        #         SimpleSplitting(EulerUpwind(MainGrad; fallbackInterpolator=FallbackGrad, mood=mood_fun), source_term)
        #     else
        #         error("Unknown IMEX TimeStepper: '$timestepper_name'")
        #     end
            
        #     kinetic_eqs = Tuple(kinetic_eqs_vec)
        #     setInitialConditions!(pg, source_term, IC)
        #     save_relax = get(run_params, "save_relax", false)

        #     return _execute_kinetic_sim!(system_method, kinetic_eqs, pg, settings, run_params, dimension, snapshots, remove_ghosts, save_relax, N_macro_vars, kinetic_to_macro_map)
        # end

    catch e
        @error "Error during Simulation!" params=params exception=(e, catch_backtrace())
        return nothing
    end
end