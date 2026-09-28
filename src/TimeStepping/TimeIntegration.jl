export solve_equation

function (method::TimeStepper)(kwargs...)
    error("Each `TimeStepper` must override the ()-operator.")
end

"""
    update_size!(ib::InteractionBuffer, num_interactions)
    update_size!(ts::GeneralIMEXTimeStepper, N_particles, M_neighbors)
    update_size!(ts::GeneralRKTimeStepper, N_particles, M_neighbors)

Dynamically resizes internal buffers and state arrays to accommodate the current number of particles and neighbor interactions.

# Details
- For `InteractionBuffer`, it ensures sufficient capacity for fields like `f`, `df`, `df_flux`, and `mask`.
- For time steppers, it resizes the target RK/IMEX stage arrays (e.g., `Y_stages`, `K_stages`) and automatically cascades the update to the internal neighbor `InteractionBuffer`.
"""
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

"""
    update_content!(ib::InteractionBuffer, nb_indices, f_i, nb_slice, fVec)

Populates the interaction buffer for a given target particle.

# Details
- Retrieves neighbor states from `fVec` using the provided `nb_indices`.
- Directly stores the neighbor state into `ib.f` and computes the raw difference (`f_j - f_i`) into `ib.df` for immediate access during flux evaluation.
"""
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

"""
    saveData!(xs_storage, us_storage, ts_storage, snap_idx, pg, current_t, remove_ghosts)

Extracts and archives the simulation state at a specific snapshot index.

# Details
- Records the current time into `ts_storage`.
- If `remove_ghosts` is true, it filters out boundary particles and saves only the active domain core. Otherwise, it copies the entire grid.
- Allocates new state and position vectors for the targeted snapshot and copies the corresponding views from the particle grid.
"""
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

"""
    solve_equation(timestepper, eq, pg, tmax, dt; kwargs...)

The primary simulation orchestrator governing the main time-stepping loop. 

# Arguments
- `timestepper::TimeStepper`: The selected Runge-Kutta or IMEX time integrator.
- `eq::HyperbolicPDE`: The physical equation system.
- `pg::ParticleGrid`: The active mesh-free domain configuration.
- `tmax::Real`: The final simulation time.
- `dt::Real`: The baseline time step.

# Keyword Arguments
- `is_cfl::Bool`: If true, `dt` is treated as a CFL number, and the physical time step is dynamically computed at each iteration using the grid and interpolator properties.
- `snapshots::Integer`: The number of discrete data dumps to record evenly across the simulation timeline.
- `remove_ghosts::Bool`: Strips boundary/ghost particles from the returned snapshot data if true.
- `show_progress::Bool`: Toggles visual progress tracking.
- `progress_interval::Real`: Sets the refresh rate (in seconds) for logging the simulation's progress and calculating the ETA.

# Returns
- A tuple containing: `(position_history, state_history, time_history, total_steps, elapsed_wall_time)`.
"""
function solve_equation(
    timestepper::TimeStepper, 
    eq::HyperbolicPDE{D, M, T, R}, 
    pg::ParticleGrid{D, M, T},
    tmax::Real,
    dt::Real;
    is_cfl::Bool = false,
    snapshots::Integer = 10,
    remove_ghosts::Bool = false,
    show_progress::Bool = true,
    progress_interval::Real = 1.0
) where {D, M, T, R}
    
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
        dt_val = is_cfl ? dt_inp * get_time_step(pg, eq, div_interp) : dt_inp
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
include("MeshfreeIMEXTimeSteppers.jl")