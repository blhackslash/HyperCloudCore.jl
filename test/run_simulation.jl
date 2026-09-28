using Random
using LinearAlgebra
using StaticArrays

# ==============================================================================
# EXECUTION BARRIERS
# ==============================================================================

@inline _unwrap(v::SVector{1, T}) where {T} = v[1]
@inline _unwrap(v) = v

@noinline function _execute_explicit_sim!(method, eq, pg, dt, is_cfl, run_params, dimension, snapshots, remove_ghosts, M_components, xmins, xmaxs, ::Type{T}) where {T}
    xs_svector, us_svector, ts_full, k_step, elapsed_time = solve_equation(method, eq, pg, run_params[:tmax], dt; is_cfl = is_cfl, snapshots = snapshots, remove_ghosts = remove_ghosts, show_progress = true, progress_interval = .1)
    @info "Explicit Simulation (D=$dimension) finished in $(round(elapsed_time, digits=2)) seconds."

    valid_indices = findall(i -> isassigned(us_svector, i), 1:length(us_svector))
    ts = ts_full[valid_indices]
    
    xs_final = xs_svector[valid_indices]
    us_final = us_svector[valid_indices]
    tmin, tmax = zero(T), T(run_params[:tmax])
    
    sim_data_result = create_sim_data(xs_final, us_final, ts, run_params; xmins = xmins, xmaxs = xmaxs, tmin = tmin, tmax = tmax)
    add_stat!(sim_data_result, :Runtime, elapsed_time, :none)
    add_stat!(sim_data_result, :k_step, Float64(k_step), :none) 
    return sim_data_result
end

@noinline function _execute_kinetic_sim!(system_method, eq_kin, pg, dt, is_cfl, run_params, dimension, snapshots, remove_ghosts, save_relax, km, xmins, xmaxs, ::Type{T}) where {T}
    xs_svector, us_svector, ts_full, k_step, elapsed_time = solve_equation(system_method, eq_kin, pg, run_params[:tmax], dt; is_cfl = is_cfl, snapshots = snapshots, remove_ghosts = remove_ghosts, show_progress = true, progress_interval = .1)
    @info "Kinetic Relaxation Simulation (D=$dimension) finished in $(round(elapsed_time, digits=2)) seconds."

    valid_indices = findall(i -> isassigned(us_svector, i), 1:length(us_svector))
    ts = ts_full[valid_indices]
    xs_final = xs_svector[valid_indices]

    if save_relax
        us_final = us_svector[valid_indices]
    else
        us_final = [[km(u) for u in snap] for snap in us_svector[valid_indices]]
    end
    tmin, tmax = zero(T), T(run_params[:tmax])
    
    sim_data_result = create_sim_data(xs_final, us_final, ts, run_params; xmins = xmins, xmaxs = xmaxs, tmin = tmin, tmax = tmax)
    add_stat!(sim_data_result, :Runtime, elapsed_time, :none)
    add_stat!(sim_data_result, :k_step, Float64(k_step), :none) 
    return sim_data_result
end

function build_equation(params::ParamDict, ::Type{T}) where {T}
    # =========================================================================
    # SMART BYPASS: Direct Struct Instantiation
    # =========================================================================
    if haskey(params, :PDE_params) && params[:PDE_params] isa Tuple && !isempty(params[:PDE_params]) && params[:PDE_params][1] isa HyperbolicPDE
        eq = params[:PDE_params][1]
        
        D = typeof(eq).parameters[1]
        NM = typeof(eq).parameters[2]
        vel_var = eq isa EulerEquation ? Tuple(2:D+1) : (1,)
        
        return eq, D, NM, vel_var
    end

    # =========================================================================
    # STANDARD FACTORY: String-based Instantiation
    # =========================================================================
    eq_name = lowercase(string(params[:PDE]))
    
    path_str = lowercase(string(get(params, :PDE_path, "mapped")))
    path_obj = if path_str == "line"
        LinePath()
    elseif path_str == "mapped"
        MappedPath()
    elseif path_str == "naive" || path_str == "naiveaverage"
        NaiveAveragePath()
    else
        error("Unknown PDE path: $path_str")
    end
    
    rep_str = lowercase(string(get(params, :PDE_representation, "conservative")))
    rep = if rep_str == "conservative"
        Conservative()
    elseif rep_str == "primitive"
        Primitive(path_obj)
    elseif rep_str == "lagrangian" || rep_str == "lagrange"
        Lagrangian(path_obj)
    else
        error("Unknown PDE representation: $rep_str")
    end

    D = haskey(params, :Ns) ? length(params[:Ns]) : 1
    
    local eq, NM, vel_var

    if eq_name == "linear"
        # Linear Advection automatically deduces T from the provided velocity matrix in its constructor
        eq = LinearAdvection(params[:PDE_params]; rep=rep) 
        NM = length(eq.vel[1])
        vel_var = (1,)
        
    elseif eq_name == "burgers"
        NM = 1
        vel_var = (1,)
        eq = BurgersEquation(Val(D), T, rep)
        
    elseif eq_name == "euler"
        NM = D + 2
        vel_var = Tuple(2:D+1)
        eq = EulerEquation(Val(D), T, T(GAS_GAMMA_EULER), rep)        
        
    else
        error("PDE '$eq_name' is not implemented.")
    end
    
    return eq, D, NM, vel_var
end

function build_kinetic_system(params::ParamDict, D::Int, NM::Int, eq::HyperbolicPDE, ::Type{T}) where {T}
    relax_config = get(params, :relax_velocities, nothing)
    
    if isnothing(relax_config)
        km = Kin2Macro(collect(1:(NM + 1)))
        return km, NM, nothing, NoSourceTerm()
    end
    
    relax_eps::T = T(params[:relax_epsilon])
    relax_indices = params[:relax_indices]
    km = Kin2Macro(relax_indices)
    
    NK = length(relax_config[1])
    eq_kin = LinearAdvection(relax_config)
    
    coeffs = State{NM, T}(ntuple(m -> one(T) / T(relax_indices[m+1] - relax_indices[m]), Val(NM)))
    int_factor = T(D) 
    
    source_term = eq.rep isa Conservative ? 
        RelaxationSourceTerm(km, relax_eps, coeffs, eq, eq_kin, int_factor) :
        NonLocalRelaxationSourceTerm(km, relax_eps, eq, coeffs, eq_kin, int_factor)
        
    return km, NK, eq_kin, source_term
end

function build_spatial_schemes(params::ParamDict, D::Int, M_comps::Int, delta_relax::T, ::Type{T}) where {T}
    order     = params[:order]
    main_grad = string(params[:main_gradient])
    main_flux = string(params[:main_flux])
    
    lim_name  = string(get(params, :limiter, "none"))
    lim_mode  = get(params, :limiter_mode, :soft)
    mood_crit = string(get(params, :mood_criterion, "none"))
    mood_strat= string(get(params, :mood_strategy, "EPD1"))
    mls_order = get(params, :MLS_order, 0)
    
    limiter = if lim_name == "minmod"; MinmodLimiter(lim_mode)
              elseif lim_name == "superbee"; SuperbeeLimiter(lim_mode)
              elseif lim_name == "VK"; VenkatakrishnanLimiter(lim_mode)
              elseif lim_name == "BJ"; BarthJespersenLimiter(lim_mode)
              else; NoLimiter() end

    mood_criterion = if mood_crit == "U2"; MOODu2(delta_relax)
               elseif mood_crit == "U1"; MOODu1(delta_relax)
               elseif mood_crit == "only"; OnlyMOOD()
               else; NoMOOD() end
               
    mood_strategy = if mood_strat == "EPD0"; EPD0()
                    elseif mood_strat == "SEPD0"; StrictEPD0()
                    elseif mood_strat == "EPD1"; EPD1()
                    elseif mood_strat == "EPD2"; EPD2()
                    end
    mood_fun = MOOD(mood_strategy, mood_criterion) 

    MainFlux = main_flux == "Rusanov" ? RusanovFlux() : (main_flux == "Upwind" ? UpwindFlux() : error("Flux NYI"))
    MainGrad = if main_grad == "MUSCL"
        MUSCL(T, D, M_comps, order; div_order = mls_order, flux = MainFlux, limiter = limiter, mood = mood_fun)
    elseif main_grad == "Upwind"
        UpwindDivergence(T, D, M_comps, order; flux=MainFlux, algType=get(params, :upwind_alg_nd, "Classic"))
    elseif main_grad == "Central"
        CentralDivergence(T, D, M_comps, order)
    elseif main_grad == "WENO"
        WENO(T, D, M_comps, order)
    else
        error("Main Gradient NYI") 
    end

    return MainGrad
end
# ==============================================================================
# MAIN SIMULATION ORCHESTRATOR
# ==============================================================================
function run_simulation(params::ParamDict)::Union{AbstractSimData, Nothing}
    @info "--- Running General N-Dimensional Simulation ---"
    
    T = get(params, :real_type, Float64)
    
    try
        if !haskey(params, :timestepper)
            @warn "Skipping run: 'timestepper' missing."
            return nothing
        end
        
        eq_macro, D, NM, vel_var = build_equation(params, T)
        IC = getInitialCondition(params[:init_func], get(params, :init_params, nothing))
        
        km, M_comps, eq_kin, source_term = build_kinetic_system(params, D, NM, eq_macro, T)
        is_kinetic = !isnothing(eq_kin)
        
        # 1. Parse Parameters & Catch Serialized Strings from the Database
        raw_bc = get(params, :bc, Dict{Int, AbstractBoundaryCondition}())
        
        # If the runner loaded the dictionary as a string, evaluate it back into code
        if raw_bc isa String
            raw_bc = eval(Meta.parse(raw_bc))
        end
        
        # 2. Universal Boundary Condition Translation
        bc_map = Dict{Int, AbstractBoundaryCondition}()
        
        for (tag, bc_obj) in raw_bc
            if bc_obj isa AbstractBoundaryCondition
                # Structs passed directly (New Method)
                bc_map[tag] = bc_obj
            else
                # Symbols or Strings passed (Backwards Compatibility)
                bc_sym = Symbol(bc_obj)
                if bc_sym === :outflow || bc_sym === :OutflowBC
                    bc_map[tag] = OutflowBC()
                elseif bc_sym === :fixed_dirichlet || bc_sym === :FixedDirichlet
                    bc_map[tag] = FixedDirichlet()
                else
                    # Dynamic Custom Struct Fallback
                    if isdefined(Main, bc_sym)
                        bc_map[tag] = getfield(Main, bc_sym)()
                    elseif isdefined(@__MODULE__, bc_sym)
                        bc_map[tag] = getfield(@__MODULE__, bc_sym)()
                    else
                        error("Boundary Condition '$bc_sym' could not be found.")
                    end
                end
            end
        end
        
        # 2. Geometry Resolution (String Dispatch OR Custom Struct)
        domain_input = get(params, :domain, "rectangular")
        local geom
        
        if typeof(domain_input) <: String || typeof(domain_input) <: Symbol
            shape = lowercase(string(domain_input))
            # Only require :mins and :maxs if building a default shape from scratch
            req_mins = T.(params[:mins]::Tuple)
            req_maxs = T.(params[:maxs]::Tuple)
            
            if shape == "rectangular"
                geom = get_rectangular_domain(T, req_mins, req_maxs; bc_map = bc_map)
            elseif shape == "spherical"
                center = ntuple(d -> (req_mins[d] + req_maxs[d]) / 2.0, Val(D))
                radius = (req_maxs[1] - req_mins[1]) / 2.0
                geom = get_spherical_domain(T, center, radius; bc_map = bc_map)
            else
                error("Unknown built-in domain shape: $shape")
            end
        else
            # The user provided a fully instantiated AbstractGeometricDomain directly!
            geom = domain_input
        end
        
        # 3. Dynamically extract bounds from the resolved geometry
        geom_mins = Tuple(geom.mins)
        geom_maxs = Tuple(geom.maxs)
        
        Ns = params[:Ns]::Tuple
        rf = get(params, :randomness_factor, ntuple(_ -> 0.0, D))
        
        # Calculate numerical grid properties based on the geometry's physical bounds
        dxs = ntuple(d -> (T(geom_maxs[d]) - T(geom_mins[d])) / Int(Ns[d]), Val(D))
        max_dx = maximum(dxs)
        vol_dx = prod(dxs)
        randomness = ntuple(d -> T(rf[d]) * dxs[d], Val(D))
        nominal_dx = dxs
        
        # 4. Enforce mandatory numerical parameters
        if !haskey(params, :interp_range)
            error("The ':interp_range' parameter is required for meshfree interpolation.")
        end
        interp_range_factor = params[:interp_range]
        interp_alpha = get(params, :interp_alpha, T(1.0))
        rng = MersenneTwister(params[:SEED])
        
        interp_range = T(interp_range_factor) * max_dx
        delta_relax = vol_dx * T(get(params, :delta_relax, 0.0))
        weight_func = ExponentialWeightFunction(T(interp_alpha), interp_range)
        
        mover_name = string(get(params, :grid_mover, "none"))
        grid_mover = if mover_name == "physical"; PhysicalGridMover{D}(vel_var)
                     elseif mover_name == "custom"; CustomGridMover(params[:grid_mover_func], params[:grid_mover_params])
                     else; NoGridMover() end
                     
        is_per_input = get(params, :periodic, false)

        # 5. Construct the Particle Grid
        pg = ParticleGrid(
            geom, nominal_dx, interp_range_factor;
            is_periodic = is_per_input,
            randomness = randomness,
            rng = rng,
            M = M_comps,
            weight_func = weight_func,
            mover = grid_mover
        )

        MainGrad = build_spatial_schemes(params, D, M_comps, delta_relax, T)

        cfl = get(params, :CFL, nothing)
        is_cfl = !isnothing(cfl)
        dt = is_cfl ? T(cfl) : T(params[:dt])
        ts_name = string(params[:timestepper])

        if !is_kinetic
            method = if ts_name == "Euler"; GeneralRKTimeStepper(eq_macro, MainGrad, RK1_Euler_Tableau(T))
                     elseif ts_name == "RK2"; GeneralRKTimeStepper(eq_macro, MainGrad, RK2_Ralston_Tableau(T))
                     elseif ts_name == "RK3"; GeneralRKTimeStepper(eq_macro, MainGrad, RK3_SSP_Tableau(T))
                     elseif ts_name == "RK4"; GeneralRKTimeStepper(eq_macro, MainGrad, RK4_Classical_Tableau(T))
                     else; error("Unknown Explicit TimeStepper: '$ts_name'") end
            
            setInitialConditions!(pg, eq_macro, IC)
            return _execute_explicit_sim!(method, eq_macro, pg, dt, is_cfl, params, D, params[:snapshots], get(params, :remove_ghosts, true), M_comps, geom_mins, geom_maxs, T)
            
        else
            implicit_solver = LinearizedRelaxationImplicitSolver()
            method = if ts_name == "ARS233"; GeneralIMEXTimeStepper(eq_macro, MainGrad, implicit_solver, source_term, IMEX_ARS233_Tableau(T))
                     elseif ts_name == "PRSSP3"; GeneralIMEXTimeStepper(eq_macro, MainGrad, implicit_solver, source_term, IMEX_PRSSP3_Tableau(T))
                     elseif ts_name == "ARS222"; GeneralIMEXTimeStepper(eq_macro, MainGrad, implicit_solver, source_term, IMEX_ARS222_Tableau(T))
                     elseif ts_name == "SSP332"; GeneralIMEXTimeStepper(eq_macro, MainGrad, implicit_solver, source_term, IMEX_SSP2332_Tableau(T))
                     elseif ts_name == "IMEXEuler"; GeneralIMEXTimeStepper(eq_macro, MainGrad, implicit_solver, source_term, IMEX_Euler_Tableau(T))
                     else; error("Unknown IMEX TimeStepper: '$ts_name'") end
            
            setInitialConditions!(pg, source_term, IC, eq_macro)
            return _execute_kinetic_sim!(method, eq_kin, pg, dt, is_cfl, params, D, params[:snapshots], get(params, :remove_ghosts, true), get(params, :save_relax, false), km, geom_mins, geom_maxs, T)
        end

    catch e
        @error "Error during Simulation!" exception=(e, catch_backtrace())
        return nothing
    end
end