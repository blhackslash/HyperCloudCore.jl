
export run_direct_simulation

function run_direct_simulation(params::ParamDict)::Union{AbstractSimData, Nothing}
    T = get(params, :real_type, Float64) 
    
    try
        @info "--- Running Direct N-Dimensional Simulation ---"
        context = Dict{Symbol, Any}()
        context[:Type] = T
        
        # 1. Extract Config Namespaces
        pde_conf    = extract_namespace(params, :PDE)
        domain_conf = extract_namespace(params, :Grid) 
        grid_conf   = extract_namespace(params, :Grid)
        weight_conf = extract_namespace(params, :Weight)
        flux_conf   = extract_namespace(params, :Flux)
        lim_conf    = extract_namespace(params, :Limiter)
        mood_conf   = extract_namespace(params, :MOOD)
        scheme_conf = extract_namespace(params, :Scheme)
        ic_conf     = extract_namespace(params, :IC)
        time_conf   = extract_namespace(params, :Time)
        
        # 2. Base Equation & Dimensions
        eq = build_equation(pde_conf, context)
        D = get_D(eq)
        
        context[:Equation] = eq
        context[:D] = D
        context[:M] = get_M(eq)
        context[:ExplicitSources] = ()
        context[:ImplicitSources] = ()
        
        # 3. Geometry & Grid
        geom = build_domain(domain_conf, context)
        context[:Domain] = geom
        context[:WeightConf] = weight_conf
        
        pg = build_particle_grid(grid_conf, context)
        context[:Grid] = pg
        
        # 4. Spatial Schemes
        context[:Flux]    = build_flux(flux_conf, context)
        context[:Limiter] = build_limiter(lim_conf, context)
        context[:MOOD]    = build_mood(mood_conf, context)
        context[:Scheme]  = build_scheme(scheme_conf, context)
        
        # 5. Time Stepper
        context[:Tableau] = build_tableau(time_conf, context)
        method = build_timestepper(context)
        
        # 6. Initialization
        IC = build_ic(ic_conf, context)
        set_initial_conditions!(pg, eq, IC)
        
        # 7. Time Settings
        is_cfl = haskey(time_conf, :CFL)
        dt = is_cfl ? T(time_conf[:CFL]) : T(time_conf[:dt])
        tmax = T(get(time_conf, :tmax, params[:tmax]))
        
        # 8. Execution
        xs_svector, us_svector, ts_full, k_step, elapsed_time = solve_equation(
            method, eq, pg, tmax, dt; 
            is_cfl = is_cfl, snapshots = params[:snapshots], 
            remove_ghosts = get(params, :remove_ghosts, true), 
        )
        
        @info "Direct Simulation (D=$D) finished in$(round(elapsed_time, digits=2)) seconds."

        # 9. Data Packaging
        valid_indices = findall(i -> isassigned(us_svector, i), 1:length(us_svector))
        sim_data = create_sim_data(xs_svector[valid_indices], us_svector[valid_indices], ts_full[valid_indices], params; xmins=Tuple(geom.mins), xmaxs=Tuple(geom.maxs), tmin=zero(T), tmax=tmax)
        add_stat!(sim_data, :Runtime, elapsed_time, :none)
        add_stat!(sim_data, :k_step, Float64(k_step), :none) 
        
        return sim_data

    catch e
        @error "Error during Direct Simulation!" exception=(e, catch_backtrace())
        return nothing
    end
end