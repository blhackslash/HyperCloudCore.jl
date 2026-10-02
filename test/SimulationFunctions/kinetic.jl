
export  run_kinetic_simulation

function run_kinetic_simulation(params::ParamDict)::Union{AbstractSimData, Nothing}
    T = get(params, :real_type, Float64) 
    
    try
        @info "--- Running Kinetic N-Dimensional Simulation ---"
        context = Dict{Symbol, Any}()
        context[:Type] = T
        
        # 1. Extract Config Namespaces
        pde_conf    = extract_namespace(params, :PDE)
        kin_conf    = extract_namespace(params, :Kinetic)
        domain_conf = extract_namespace(params, :Grid) 
        grid_conf   = extract_namespace(params, :Grid)
        weight_conf = extract_namespace(params, :Weight)
        flux_conf   = extract_namespace(params, :Flux)
        lim_conf    = extract_namespace(params, :Limiter)
        mood_conf   = extract_namespace(params, :MOOD)
        scheme_conf = extract_namespace(params, :Scheme)
        ic_conf     = extract_namespace(params, :IC)
        time_conf   = extract_namespace(params, :Time)
        
        # 2. Base Macroscopic Equation
        eq_macro = build_equation(pde_conf, context)
        D = get_D(eq_macro)
        
        context[:Equation] = eq_macro 
        context[:D] = D
        context[:M] = get_M(eq_macro)

        IC = build_ic(ic_conf, context)
        
        # 3. Build Kinetic System
        eq_kin, source_term = build_kinetic_system(kin_conf, context)
        
        # Overwrite Context for downstream components to scale to NK dimensions
        context[:Equation] = eq_kin
        context[:M] = get_M(eq_kin)
        context[:ExplicitSources] = ()
        context[:ImplicitSources] = (source_term,)
        
        # 4. Geometry & Grid
        geom = build_domain(domain_conf, context)
        context[:Domain] = geom
        context[:WeightConf] = weight_conf
        
        pg = build_particle_grid(grid_conf, context)
        context[:Grid] = pg
        
        # 5. Spatial Schemes
        context[:Flux]    = build_flux(flux_conf, context)
        context[:Limiter] = build_limiter(lim_conf, context)
        context[:MOOD]    = build_mood(mood_conf, context)
        context[:Scheme]  = build_scheme(scheme_conf, context)
        
        # 6. Time Stepper
        context[:Tableau] = build_tableau(time_conf, context)
        method = build_timestepper(context)
        
        # 7. Initialization
        set_initial_conditions!(pg, source_term, IC, eq_macro)
        
        # 8. Time Settings
        is_cfl = haskey(time_conf, :CFL)
        dt = is_cfl ? T(time_conf[:CFL]) : T(time_conf[:dt])
        tmax = T(get(time_conf, :tmax, params[:tmax]))
        
        # 9. Execution
        xs_svector, us_svector, ts_full, k_step, elapsed_time = solve_equation(
            method, eq_kin, pg, tmax, dt; 
            is_cfl = is_cfl, snapshots = params[:snapshots], 
            remove_ghosts = get(params, :remove_ghosts, true), 
        )
        
        @info "Kinetic Simulation (D=$D) finished in$(round(elapsed_time, digits=2)) seconds."

        # 10. Data Packaging & Macroscopic Collapse
        valid_indices = findall(i -> isassigned(us_svector, i), 1:length(us_svector))
        
        if get(params, :save_relax, false) || get(kin_conf, :save_relax, false)
            us_final = us_svector[valid_indices]
        else
            # Extract km directly from the returned source_term
            km = source_term.km
            us_final = [[km(u) for u in snap] for snap in us_svector[valid_indices]]
        end
        
        sim_data = create_sim_data(xs_svector[valid_indices], us_final, ts_full[valid_indices], params; xmins=Tuple(geom.mins), xmaxs=Tuple(geom.maxs), tmin=zero(T), tmax=tmax)
        add_stat!(sim_data, :Runtime, elapsed_time, :none)
        add_stat!(sim_data, :k_step, Float64(k_step), :none) 
        
        return sim_data

    catch e
        @error "Error during Kinetic Simulation!" exception=(e, catch_backtrace())
        return nothing
    end
end