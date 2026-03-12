function (method::TimeStepper)(eq, pg, settings, time, dt)
    error("Each `TimeStepper' must override the ()-operator.")
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

function _ensure_capacity!(v::AbstractVector, n::Int)
    if length(v) < n
        resize!(v, n)
    end
    return nothing
end

include("MeshfreeTimeSteppers.jl")
include("FixedGridTimeSteppers.jl")


include("ButcherTableaus.jl")
include("SourceTerms.jl")
include("ImplicitSolvers.jl")
include("MeshfreeSystemTimeSteppers.jl")

# --- Low-Level `saveData!` Helpers ---

function _copy_positions!(dest::Vector{Float64}, src::AbstractVector)
    copyto!(dest, src)
end

function _copy_positions!(dest::Vector{Tuple{Float64,Float64}}, src::AbstractVector{SVector{2, Float64}})
    @inbounds for i in eachindex(dest, src)
        dest[i] = Tuple(src[i])
    end
end

"""
    saveData!(...)

Saves data from a unified ParticleGrid into pre-allocated storage slots.
Works for both scalar (M=1) and system (M>1) equations natively.
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
    
    if remove_ghosts
        active_boundary_view = @view pg.core.is_boundary[1:N_active]
        indices = findall(.!active_boundary_view)
        
        N_save = length(indices)
        pos_type = D == 1 ? Float64 : Tuple{Float64,Float64}
        
        xs_storage[snap_idx] = Vector{pos_type}(undef, N_save)
        us_storage[snap_idx] = Matrix{Float64}(undef, N_save, M)
        
        # Copy positions
        _copy_positions!(xs_storage[snap_idx], view(get_positions(pg), indices))
        
        # Copy rhos using a direct matrix slice
        copyto!(us_storage[snap_idx], view(pg.rhos, indices, :))
    else
        N_save = N_active
        pos_type = D == 1 ? Float64 : Tuple{Float64,Float64}
        
        xs_storage[snap_idx] = Vector{pos_type}(undef, N_save)
        us_storage[snap_idx] = Matrix{Float64}(undef, N_save, M)
        
        # Copy strictly 1:N_active 
        _copy_positions!(xs_storage[snap_idx], view(get_positions(pg), 1:N_save))
        
        # Copy rhos using a direct matrix slice
        copyto!(us_storage[snap_idx], view(pg.rhos, 1:N_save, :))
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

    pos_type = D == 1 ? Float64 : Tuple{Float64,Float64}
    
    xs = Vector{Vector{pos_type}}(undef, snapshots + 1)
    us = Vector{Matrix{Float64}}(undef, snapshots + 1)
    ts = Vector{Float64}(undef, snapshots + 1)

    t_snap = range(0.0, settings.tmax, length=snapshots+1)
    snap_counter = 1
    t = 0.0
    k_step = 0

    saveData!(xs, us, ts, snap_counter, pg, t, remove_ghosts)
    snap_counter += 1 

    p = Progress(convert(Int, ceil(settings.tmax / settings.dt)), desc="Running Simulation...")

    elapsed_time = @elapsed while t < settings.tmax && snap_counter <= (snapshots + 1)
        dt = min(settings.dt, settings.tmax - t)
        if dt <= 1e-12; break; end

        timestepper(eqs, pg, settings, t, dt)
        
        t += dt
        k_step += 1

        while snap_counter <= (snapshots + 1) && t >= t_snap[snap_counter]
            saveData!(xs, us, ts, snap_counter, pg, t, remove_ghosts)
            snap_counter += 1
        end
        
        ProgressMeter.next!(p)
    end

    if snap_counter <= (snapshots + 1)
        saveData!(xs, us, ts, snap_counter, pg, t, remove_ghosts)
    end

    # Return trimmed arrays in case early exit occurred
    num_saved = snap_counter - 1
    return elapsed_time, xs[1:num_saved], us[1:num_saved], ts[1:num_saved]
end
