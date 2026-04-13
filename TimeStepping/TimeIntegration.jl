function (method::TimeStepper)(eq, pg, settings, time, dt)
    error("Each `TimeStepper' must override the ()-operator.")
end

"""
    initFs!(neighbor_fs, neighbor_dfs, nb_indices, i, f_i, nb_slice, fVec)

Parallel "pre-gather" loop to fill the `neighbor_fs` and `neighbor_dfs` 
buffers using data from `fVec`. Works natively with SVector states.
"""
@inline function initFs!(
    neighbor_fs::AbstractVector{State{M}}, 
    neighbor_dfs::AbstractVector{State{M}}, 
    nb_indices::AbstractVector{Int}, 
    f_i::State{M}, 
    nb_slice::UnitRange{Int}, 
    fVec::AbstractVector{State{M}}
) where {M}
    
    # ivdep tells the compiler it is safe to ignore perceived memory dependencies
    @inbounds for k in nb_slice
        j = nb_indices[k]
        f_j = fVec[j] 
        
        neighbor_fs[k]  = f_j
        neighbor_dfs[k] = f_j - f_i 
    end
end

function initTS!(ts::MeshfreeTimeStepper, pg::ParticleGrid)
    updateNeighbors!(pg)
end

function initTSBuffer!(ts::MeshfreeTimeStepper, pg::ParticleGrid)
    # `num_interactions` is the total length of the flat neighbor lists (M)
    num_interactions = length(pg.neighbor.indices) 
    
    # --- 3. Resize Per-Interaction Buffers (Size M) ---
    _ensure_capacity!(ts.neighbor_fs, num_interactions)
    _ensure_capacity!(ts.neighbor_dfs, num_interactions)
    initAddTSBuffer!(ts, pg)
    
    return nothing
end

include("MeshfreeRKTimeSteppers.jl")
include("FixedGridTimeSteppers.jl")
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
    eqs, # Can be ScalarHyperbolicPDE or DiagonalHyperbolicSystem
    pg::ParticleGrid{D, M}, 
    settings::SimSetting;
    snapshots::Integer = 10,
    remove_ghosts::Bool = false
) where {D, M}
    
    # Output arrays hold Vectors of SVectors!
    xs = Vector{Vector{Space{D}}}(undef, snapshots + 1)
    us = Vector{Vector{State{M}}}(undef, snapshots + 1)
    ts = Vector{Float64}(undef, snapshots + 1)

    t_snap = range(0.0, settings.tmax, length=snapshots+1)
    snap_counter = 1
    t = 0.0
    k_step = 0

    # 1. Save initial condition (t=0)
    saveData!(xs, us, ts, snap_counter, pg, t, remove_ghosts)
    snap_counter += 1 

    p = Progress(convert(Int, ceil(settings.tmax / settings.dt)), desc="Running Simulation...")

    elapsed_time = @elapsed while t < settings.tmax
        dt = min(settings.dt, settings.tmax - t)
        if dt <= 1e-12; break; end

        timestepper(eqs, pg, settings, t, dt)
        
        t += dt
        k_step += 1

        # 2. Save intermediate snapshots (stop before the final slot)
        while snap_counter <= snapshots && t >= t_snap[snap_counter]
            saveData!(xs, us, ts, snap_counter, pg, t, remove_ghosts)
            snap_counter += 1
        end

        next!(p)
    end
    finish!(p)

    # 3. Always force the final snapshot exactly at the end
    if snap_counter <= snapshots + 1
        saveData!(xs, us, ts, snapshots + 1, pg, t, remove_ghosts)
    end

    return xs, us, ts, k_step, elapsed_time
end