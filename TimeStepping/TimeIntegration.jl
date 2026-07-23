function (method::TimeStepper)(kwargs...)
    error("Each `TimeStepper' must override the ()-operator.")
end

function update_size!(ib::InteractionBuffer, num_interactions::Int)
    ensure_capacity!(ib.f, num_interactions)
    ensure_capacity!(ib.df, num_interactions)
    ensure_capacity!(ib.dfFlux, num_interactions)
    ensure_capacity!(ib.df_scratch, num_interactions)
    ensure_capacity!(ib.mask, num_interactions)
    return nothing
end
function update_size!(ts::GeneralIMEXTimeStepper, N_particles::Int, M_neighbors::Int)
    ensure_capacity!(ts.U_n, N_particles)
    
    for i in 1:ts.num_stages
        ensure_capacity!(ts.Y_stages[i], N_particles)
        ensure_capacity!(ts.K_E_stages[i], N_particles)
        ensure_capacity!(ts.K_I_stages[i], N_particles)
    end
    
    # Cascade down to the interaction buffer!
    update_size!(ts.int_buffer, M_neighbors)
    return nothing
end

function update_size!(ts::GeneralRKTimeStepper, N_particles::Int, M_neighbors::Int)
    ensure_capacity!(ts.rho_n, N_particles)
    ensure_capacity!(ts.rho_stage, N_particles)
    
    for i in 1:length(ts.K_stages)
        ensure_capacity!(ts.K_stages[i], N_particles)
    end
    
    # Cascade down to the interaction buffer!
    update_size!(ts.int_buffer, M_neighbors)
    return nothing
end

"""
Parallel "pre-gather" loop to fill the interaction buffer.
"""
@inline function update_content!(
    ib::InteractionBuffer{D, M},
    nb_indices::AbstractVector{Int}, 
    f_i::State{M}, 
    nb_slice::UnitRange{Int}, 
    fVec::AbstractVector{State{M}}
) where {D, M}
    
    # ivdep tells the compiler it is safe to ignore perceived memory dependencies
    @inbounds for k in nb_slice
        j = nb_indices[k]
        f_j = fVec[j] 
        
        ib.f[k]  = f_j
        ib.df[k] = f_j - f_i 
    end
    return nothing
end

include("MeshfreeRKTimeSteppers.jl")
include("ButcherTableaus.jl")
include("SourceTerms.jl")
include("ImplicitSolvers.jl")
include("MeshfreeIMEXTimeSteppers.jl")


"""
    saveData!(...)

Saves data from a unified ParticleGrid into pre-allocated storage slots.
Works natively for both 1D and Multi-D SVectors.
"""
function saveData!(
    xs_storage::AbstractVector, 
    us_storage::AbstractVector, 
    ts_storage::AbstractVector, 
    snap_idx::Int, 
    pg::ParticleGrid{D, M}, 
    current_t::Real, 
    remove_ghosts::Bool
) where {D, M}
    
    ts_storage[snap_idx] = current_t
    N_active = pg.meta.N 
    
    # Extract native arrays (These are Vector{SVector})
    pos_array = get_positions(pg)
    rho_array = pg.rhos
    
    if remove_ghosts
        # Create a view of active particles and find interior indices
        active_boundary_view = @view pg.core.is_boundary[1:N_active]
        indices = findall(.!active_boundary_view)
        N_save = length(indices)
        
        # Pre-allocate SVector output arrays for this snapshot
        xs_storage[snap_idx] = Vector{Space{D}}(undef, N_save)
        us_storage[snap_idx] = Vector{State{M}}(undef, N_save)
        
        # Perform fast vector copy based on the filtered indices
        copyto!(xs_storage[snap_idx], view(pos_array, indices))
        copyto!(us_storage[snap_idx], view(rho_array, indices))
    else
        N_save = N_active
        
        # Pre-allocate SVector output arrays for this snapshot
        xs_storage[snap_idx] = Vector{Space{D}}(undef, N_save)
        us_storage[snap_idx] = Vector{State{M}}(undef, N_save)
        
        # Perform fast contiguous memory copy for active particles
        copyto!(xs_storage[snap_idx], view(pos_array, 1:N_save))
        copyto!(us_storage[snap_idx], view(rho_array, 1:N_save))
    end
end

"""
    mainTimeIntegrator!(...)

Unified time integration loop for both scalar and system equations.
"""
function mainTimeIntegrator!(
    timestepper::TimeStepper, 
    eq, # Can be ScalarHyperbolicPDE or DiagonalHyperbolicSystem
    pg::ParticleGrid{D, M},
    tmax::Real,
    dt::Real;
    is_cfl::Bool = false,
    snapshots::Integer = 10,
    remove_ghosts::Bool = false
) where {D, M}
    
    # Output arrays hold Vectors of SVectors!
    xs = Vector{Vector{Space{D}}}(undef, snapshots + 1)
    us = Vector{Vector{State{M}}}(undef, snapshots + 1)
    ts = Vector{Float64}(undef, snapshots + 1)

    tmax = Float64(tmax)
    dt_inp = Float64(dt)
    t_snap = range(0.0, tmax, length=snapshots+1)
    snap_counter = 1
    t = 0.0
    k_step = 0
    grad_interp = hasproperty(timestepper, :gradientInterpolator) ? timestepper.gradientInterpolator : NoFallbackGrad()

    # 1. Save initial condition (t=0)
    saveData!(xs, us, ts, snap_counter, pg, t, remove_ghosts)
    snap_counter += 1 
    threshold = calculate_thread_threshold(pg,timestepper.gradientInterpolator)
    _set_threads!(threshold)
    println(_use_threads())

    p = Progress(10000, desc="Running Simulation...")

    elapsed_time = @elapsed while t < tmax
        
        dt = is_cfl ? dt_inp * getTimeStep(pg, eq, grad_interp) : dt_inp
        dt = min(dt, tmax - t)
        if dt <= 1e-12; break; end

        timestepper(eq, pg, t, dt)
        
        t += dt
        k_step += 1
        # 2. Save intermediate snapshots (stop before the final slot)
        while snap_counter <= snapshots && t >= t_snap[snap_counter]
            saveData!(xs, us, ts, snap_counter, pg, t, remove_ghosts)
            snap_counter += 1
        end

# Calculate the actual fraction of time completed (t / tmax)
        current_progress = ceil(Int, (t / tmax) * 10000)
        
        # Safely cap it at 10000 to prevent bounds warnings near the end
        update!(p, min(current_progress, 10000))
    end
    finish!(p)

    # 3. Always force the final snapshot exactly at the end
    if snap_counter <= snapshots + 1
        saveData!(xs, us, ts, snapshots + 1, pg, t, remove_ghosts)
    end

    return xs, us, ts, k_step, elapsed_time
end