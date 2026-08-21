export mainTimeIntegrator!

function (method::TimeStepper)(kwargs...)
    error("Each `TimeStepper` must override the ()-operator.")
end

function update_size!(ib::InteractionBuffer, num_interactions::Int)
    ensure_capacity!(ib.f, num_interactions)
    ensure_capacity!(ib.df, num_interactions)
    ensure_capacity!(ib.df_flux, num_interactions)
    ensure_capacity!(ib.df_scratch, num_interactions)
    ensure_capacity!(ib.mask, num_interactions)
    return nothing
end

function update_size!(ts::GeneralIMEXTimeStepper, N_particles::Int, M_neighbors::Int)
    ensure_capacity!(ts.rho_n, N_particles)
    
    for i in 1:ts.num_stages
        ensure_capacity!(ts.Y_stages[i], N_particles)
        ensure_capacity!(ts.K_E_stages[i], N_particles)
        ensure_capacity!(ts.K_I_stages[i], N_particles)
    end
    
    update_size!(ts.int_buffer, M_neighbors)
    return nothing
end

function update_size!(ts::GeneralRKTimeStepper, N_particles::Int, M_neighbors::Int)
    ensure_capacity!(ts.rho_n, N_particles)
    ensure_capacity!(ts.rho_stage, N_particles)
    
    for i in 1:length(ts.K_stages)
        ensure_capacity!(ts.K_stages[i], N_particles)
    end
    
    update_size!(ts.int_buffer, M_neighbors)
    return nothing
end

@inline function update_content!(
    ib::InteractionBuffer{D, M, T},
    nb_indices::AbstractVector{Int}, 
    f_i::State{M, T}, 
    nb_slice::UnitRange{Int}, 
    fVec::AbstractVector{State{M, T}}
) where {D, M, T}
    
    @inbounds for k in nb_slice
        j = nb_indices[k]
        f_j = fVec[j] 
        
        ib.f[k]  = f_j
        ib.df[k] = f_j - f_i 
    end
    return nothing
end

function saveData!(
    xs_storage::AbstractVector, 
    us_storage::AbstractVector, 
    ts_storage::AbstractVector, 
    snap_idx::Int, 
    pg::ParticleGrid{D, M, T}, 
    current_t::Real, 
    remove_ghosts::Bool
) where {D, M, T}
    
    ts_storage[snap_idx] = T(current_t)
    N_active = pg.meta.N 
    
    pos_array = get_positions(pg)
    rho_array = pg.rhos
    
    if remove_ghosts
        active_boundary_view = @view pg.core.is_boundary[1:N_active]
        indices = findall(.!active_boundary_view)
        N_save = length(indices)
        
        xs_storage[snap_idx] = Vector{Space{D, T}}(undef, N_save)
        us_storage[snap_idx] = Vector{State{M, T}}(undef, N_save)
        
        copyto!(xs_storage[snap_idx], view(pos_array, indices))
        copyto!(us_storage[snap_idx], view(rho_array, indices))
    else
        N_save = N_active
        
        xs_storage[snap_idx] = Vector{Space{D, T}}(undef, N_save)
        us_storage[snap_idx] = Vector{State{M, T}}(undef, N_save)
        
        copyto!(xs_storage[snap_idx], view(pos_array, 1:N_save))
        copyto!(us_storage[snap_idx], view(rho_array, 1:N_save))
    end
end

function mainTimeIntegrator!(
    timestepper::TimeStepper, 
    eq, 
    pg::ParticleGrid{D, M, T},
    tmax::Real,
    dt::Real;
    is_cfl::Bool = false,
    snapshots::Integer = 10,
    remove_ghosts::Bool = false,
    show_progress::Bool = true,
    progress_interval::Real = 1.0
) where {D, M, T}
    
    xs = Vector{Vector{Space{D, T}}}(undef, snapshots + 1)
    us = Vector{Vector{State{M, T}}}(undef, snapshots + 1)
    ts = Vector{T}(undef, snapshots + 1)

    tmax_val = T(tmax)
    dt_inp = T(dt)
    t_snap = range(zero(T), tmax_val, length=snapshots+1)
    snap_counter = 1
    t = zero(T)
    k_step = 0
    
    div_interp = hasproperty(timestepper, :divergence_interpolator) ? timestepper.divergence_interpolator : NoFallbackGrad()

    saveData!(xs, us, ts, snap_counter, pg, t, remove_ghosts)
    snap_counter += 1 
    
    @info "Using $(_USE_THREADS[] ? "@threads" : "@batch") for parallel runs!"

    p = Progress(10000, desc="Running Simulation...", dt=progress_interval, enabled=show_progress)
    
    # Initialize trackers for the interval-based ETA
    last_log_time = time()
    last_sim_time = t

    elapsed_time = @elapsed while t < tmax_val
        dt_val = is_cfl ? dt_inp * getTimeStep(pg, eq, div_interp) : dt_inp
        dt_val = min(dt_val, tmax_val - t)
        if dt_val <= T(1e-12); break; end

        timestepper(eq, pg, t, dt_val)
        
        t += dt_val
        k_step += 1
        
        while snap_counter <= snapshots && t >= t_snap[snap_counter]
            saveData!(xs, us, ts, snap_counter, pg, t, remove_ghosts)
            snap_counter += 1
        end

        if show_progress
            current_progress = ceil(Int, (t / tmax_val) * 10000)
            update!(p, min(current_progress, 10000))
        else
            current_time = time()
            wall_dt = current_time - last_log_time
            
            if wall_dt > progress_interval
                sim_dt = t - last_sim_time
                pct = round((t / tmax_val) * 100, digits=1)
                
                # Calculate ETA only if simulation has advanced
                if sim_dt > 0
                    eta_seconds = (tmax_val - t) * (wall_dt / sim_dt)
                    eta_secs_int = round(Int, eta_seconds)
                    
                    # Format as HH:MM:SS
                    h = eta_secs_int ÷ 3600
                    m = (eta_secs_int % 3600) ÷ 60
                    s = eta_secs_int % 60
                    eta_str = string(lpad(h, 2, '0'), ":", lpad(m, 2, '0'), ":", lpad(s, 2, '0'))
                    
                    @info "Simulation Progress: $pct% | ETA: $eta_str"
                else
                    @info "Simulation Progress: $pct% | ETA: Calculating..."
                end
                
                # Reset interval trackers
                last_log_time = current_time
                last_sim_time = t
            end
        end
    end
    
    show_progress && finish!(p)

    if snap_counter <= snapshots + 1
        saveData!(xs, us, ts, snapshots + 1, pg, t, remove_ghosts)
    end

    return xs, us, ts, k_step, elapsed_time
end

include("MeshfreeRKTimeSteppers.jl")
include("ButcherTableaus.jl")
include("SourceTerms.jl")
include("ImplicitSolvers.jl")
include("MeshfreeIMEXTimeSteppers.jl")