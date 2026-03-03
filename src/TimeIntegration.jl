module TimeIntegration

using LinearAlgebra
using ProgressMeter
using StaticArrays
using Base.Threads
using ..CoreUtils
using ..ParticleGrids
using ..SimSettings
using ..HyperbolicPDEs
using ..Interpolations
using ..SourceTerms
using ..ImplicitSolvers
using ..MOOD
using ..MLSWeightFunctions
using ..GridMovement

export mainTimeIntegrator!, mainTimeIntegratorNew!

"""
    TimeStepper

Time integration methods are split into two groups: MeshfreeTimeStepper and FixedGridTimeStepper, the former working for all types of grids, 
the latter only working for fixed grid with uniform discretisation parameter. Time integration methods are implemented as structs that override
the ()-operator. See EulerUpwind in "MeshfreeTimeSteppers.jl" as an example. 
"""
abstract type TimeStepper end
abstract type MeshfreeTimeStepper <: TimeStepper end
abstract type FixedGridTimeStepper <: TimeStepper end
abstract type MeshfreeSystemTimeStepper <: MeshfreeTimeStepper end

function (method::TimeStepper)(eq::ScalarHyperbolicPDE, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    error("Each `TimeStepper' must override the ()-operator.")
end
function initTS!(ts::MeshfreeTimeStepper, pg::ParticleGrid)
    updateNeighbors!(pg)
end

function initTSBuffer!(ts::MeshfreeTimeStepper, pg::ParticleGrid)
    # `num_interactions` is the total length of the flat neighbor lists (M)
    num_interactions = length(pg.neighbor_indices) 
    # --- 3. Resize Per-Interaction Buffers (Size M) ---
    _ensure_capacity!(ts.neighbor_fs, num_interactions)
    _ensure_capacity!(ts.neighbor_dfs, num_interactions)
    initAddTSBuffer!(ts, pg)
    return nothing
end
function initTSBuffer!(ts::MeshfreeSystemTimeStepper, pgs::ParticleGridSystem{D,N}) where {D,N}
    # `num_interactions` is the total length of the flat neighbor lists (M)
    num_interactions = length(pgs[1].neighbor_indices)
    num_eqs = length(pgs)
    if size(ts.all_neighbor_dfs,1) < num_interactions
        n = num_interactions + num_interactions ÷ 4
        ts.all_neighbor_fs = Matrix{Float64}(undef, n, num_eqs)
        ts.all_neighbor_dfs = Matrix{Float64}(undef, n, num_eqs)
    end
    initAddTSBuffer!(ts, pgs)
    return nothing
end

"""
Ensures a vector `v` has at least capacity `n`.
Resizes if `length(v) < n`.
"""
function _ensure_capacity!(v::AbstractVector, n::Int)
    if length(v) < n
        n = n #+ n ÷ 4
        resize!(v, n)
    end
    return nothing
end


include("MeshfreeTimeSteppers.jl")
include("FixedGridTimeSteppers.jl")
include("MeshfreeSystemTimeSteppers.jl")

using StaticArrays

# --- Low-Level `saveData!` Helpers ---

"""
Helper for 1D positions (Vector{Float64}). Simply copies the data.
"""
function _copy_positions!(dest::Vector{Float64}, src::AbstractVector{Float64})
    copyto!(dest, src)
end

"""
Helper for 2D positions (Vector{SVector} -> Vector{Tuple}).
Converts SVectors to Tuples for plotting compatibility while copying.
"""
function _copy_positions!(dest::Vector{Tuple{Float64,Float64}}, src::AbstractVector{SVector{2, Float64}})
    @inbounds for i in eachindex(dest, src)
        dest[i] = Tuple(src[i])
    end
end

"""
Saves data from a SCALAR particle grid into pre-allocated storage slots.
"""
function saveData!(xs_storage::AbstractVector{X}, us_storage::AbstractVector{U}, ts_storage, snap_idx::Int, 
                   pg::ParticleGrid, current_t, remove_ghosts::Bool) where {X,U}
    
    # Save the current time
    ts_storage[snap_idx] = current_t
    N = pg.N

    
    if remove_ghosts
        indices = collect(1:N)[view(.!pg.is_boundary,1:N)]
        N_save = length(indices)
        xs_storage[snap_idx] = Vector{X}(undef,N_save)
        us_storage[snap_idx] = Vector{U}(undef,N_save)
        # Get the destination vectors for this snapshot
        dest_pos = xs_storage[snap_idx]
        dest_rho = us_storage[snap_idx]
        # Get logical indices of non-ghost particles
        _copy_positions!(dest_pos, view(pg.positions, indices))
        copyto!(dest_rho, view(pg.rhos, indices))
    else
        xs_storage[snap_idx] = Vector{X}(undef,N)
        us_storage[snap_idx] = Vector{U}(undef,N)
        # Get the destination vectors for this snapshot
        dest_pos = xs_storage[snap_idx]
        dest_rho = us_storage[snap_idx]
        # Copy all data
        _copy_positions!(dest_pos, view(pg.positions,1:N))
        copyto!(dest_rho, view(pg.rhos,1:N))
    end
end

# In TimeIntegration.jl

# In TimeIntegration.jl

# In TimeIntegration.jl

function saveData!(
    xs_storage::AbstractVector, 
    us_storage::AbstractVector, 
    ts_storage::AbstractVector, 
    snap_idx::Int, 
    system_pgs::ParticleGridSystem{N_grids, D_dim}, 
    current_t::Real, 
    remove_ghosts::Bool
) where {N_grids, D_dim}
    
    ts_storage[snap_idx] = current_t
    first_pg = system_pgs[1]
    N_active = first_pg.N # <--- Valid particles are only 1:N
    
    if remove_ghosts
        # Only look for boundaries within the ACTIVE range (1:N)
        # Otherwise we might pick up garbage 'false' flags from the buffer zone
        active_boundary_view = @view first_pg.is_boundary[1:N_active]
        indices = findall(.!active_boundary_view)
        
        N_save = length(indices)
        pos_type = D_dim == 1 ? Float64 : Tuple{Float64,Float64}
        
        xs_storage[snap_idx] = Vector{pos_type}(undef, N_save)
        us_storage[snap_idx] = Matrix{Float64}(undef, N_save, N_grids)
        
        # Copy positions using the safe indices
        _copy_positions!(xs_storage[snap_idx], view(first_pg.positions, indices))
        
        # Copy rhos
        dest_matrix = us_storage[snap_idx]
        for (k, pg) in enumerate(system_pgs)
            copyto!(view(dest_matrix, :, k), view(pg.rhos, indices))
        end
    else
        N_save = N_active
        pos_type = D_dim == 1 ? Float64 : Tuple{Float64,Float64}
        
        xs_storage[snap_idx] = Vector{pos_type}(undef, N_save)
        us_storage[snap_idx] = Matrix{Float64}(undef, N_save, N_grids)
        
        # Copy strictly 1:N_active (Ignore the buffer tail!)
        _copy_positions!(xs_storage[snap_idx], view(first_pg.positions, 1:N_save))
        
        dest_matrix = us_storage[snap_idx]
        for (k, pg) in enumerate(system_pgs)
            copyto!(view(dest_matrix, :, k), view(pg.rhos, 1:N_save))
        end
    end
end


"""
Optimized main time integrator for a SCALAR equation.
"""
function mainTimeIntegrator!(
    timeStepper::TimeStepper, 
    eq::ScalarHyperbolicPDE{D}, 
    particleGrid::ParticleGrid, 
    settings::SimSetting;
    snapshots::Integer = 10, 
    remove_ghosts::Bool = false
) where {D}
    # --- Pre-allocate Storage ---
    # Determine the number of particles to save
    N_save = remove_ghosts ? (particleGrid.N - particleGrid.N_ghost) : particleGrid.N
    
    # Define the position type (Float64 for 1D, Tuple for 2D)
    pos_type = D == 1 ? Float64 : Tuple{Float64,Float64}
    
    # Pre-allocate the storage vectors
    xs = Vector{Vector{pos_type}}(undef, snapshots+1)
    us = Vector{Vector{Float64}}(undef,snapshots+1)
    ts = Vector{Float64}(undef, snapshots+1)

    # --- Snapshot Time Points ---
    # Create an array of equidistant time points, including t=0 and t=tmax
    t_snap = range(0.0, settings.tmax, length=snapshots)
    snap_counter = 1

    # Save initial state (t=0)
    saveData!(xs, us, ts, snap_counter, particleGrid, t_snap[snap_counter], remove_ghosts)
    snap_counter += 1 # We are now looking for the 2nd snapshot

    # --- Main Time Loop ---
    t = 0.0
    k_step = 0
    p = Progress(convert(Int, ceil(settings.tmax / settings.dt)); desc = "Running Scalar Simulation...")
    
    elapsed_time = @elapsed while t < settings.tmax && snap_counter <= snapshots
        actual_dt = min(settings.dt, settings.tmax - t)
        if actual_dt <= 1e-12; break; end

        timeStepper(eq, particleGrid, settings, t, actual_dt)
        
        t += actual_dt
        k_step += 1
        # Check for and save snapshots
        # Use a while-loop in case dt spans multiple snapshot times
        while snap_counter <= snapshots && t >= t_snap[snap_counter]
            # Save data at the *exact* snapshot time
            saveData!(xs, us, ts, snap_counter, particleGrid, t, remove_ghosts)
            snap_counter += 1
        end
        
        ProgressMeter.next!(p)
    end
    saveData!(xs, us, ts, snap_counter, particleGrid, settings.tmax, remove_ghosts)
    num_saved_snapshots = snap_counter
    return elapsed_time, xs[1:num_saved_snapshots], us[1:num_saved_snapshots], ts[1:num_saved_snapshots]
end


"""
    mainTimeIntegrator!(system_timestepper, system_eqs, system_pgs, settings; ...)

The main time integration loop for a **system** of equations.
Adapted to handle dynamic particle numbers (AMR/ALE) by resizing storage for each snapshot.
"""
function mainTimeIntegrator!(
    system_timestepper::MeshfreeSystemTimeStepper, 
    system_eqs::DiagonalHyperbolicSystem{N,D}, 
    system_pgs::ParticleGridSystem{N, D}, 
    settings::SimSetting;
    snapshots::Integer = 10,
    remove_ghosts::Bool = false
) where {N,D}

    # --- 1. Pre-allocate Storage Containers ---
    # We use Vector of Vectors/Matrices to allow N to change between snapshots.
    # The inner elements are 'undef' until saveData! allocates them.
    
    pos_type = D == 1 ? Float64 : Tuple{Float64,Float64}
    
    xs = Vector{Vector{pos_type}}(undef, snapshots + 1)
    us_sys = Vector{Matrix{Float64}}(undef, snapshots + 1)
    ts = Vector{Float64}(undef, snapshots + 1)

    # --- 2. Snapshot Time Points ---
    t_snap = range(0.0, settings.tmax, length=snapshots+1)
    snap_counter = 1
    t = 0.0
    k_step = 0

    # --- 3. Save Initial State (t=0) ---
    saveData!(xs, us_sys, ts, snap_counter, system_pgs, t, remove_ghosts)
    snap_counter += 1 

    # --- 4. Main Time Loop ---
    p = Progress(convert(Int, ceil(settings.tmax / settings.dt)), desc="Running System Simulation...")

    elapsed_time = @elapsed while t < settings.tmax && snap_counter <= (snapshots + 1)
        
        dt = min(settings.dt, settings.tmax - t)
        if dt <= 1e-12; break; end

        # Step
        system_timestepper(system_eqs, system_pgs, settings, t, dt)
        
        t += dt
        k_step += 1

        # Save Snapshot(s)
        while snap_counter <= (snapshots + 1) && t >= t_snap[snap_counter]
            saveData!(xs, us_sys, ts, snap_counter, system_pgs, t, remove_ghosts)
            snap_counter += 1
        end
        
        next!(p)
    end

    # Force final save if missed (numerical epsilon issues)
    if snap_counter <= (snapshots + 1)
        saveData!(xs, us_sys, ts, snap_counter, system_pgs, t, remove_ghosts)
    end

    return elapsed_time, xs, us_sys, ts
end

end  # module TimeIntegration