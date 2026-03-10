export EulerUpwind, Upwind, RalstonRK2, RK3, RK4, RalstonRK2Limiter, RalstonRK2SmoothSwitch, RalstonRK2SmoothSwitch2

include("./TestUtils.jl")
using Base.Threads
# This assumes your ParticleGrid abstract type is accessible, e.g., via:
# using ..ParticleGrids 

# This function assumes that any timestepper `ts` you pass to it will have 
# mutable fields `neighbor_fs::Vector{Float64}` and `neighbor_dfs::Vector{Float64}`.

"""
    initFs!(ts::MeshfreeTimeStepper, pg::ParticleGrid, fVec::AbstractVector)

Initializes all buffers within the timestepper `ts` based on the particle grid `pg`.
It then runs a parallel "pre-gather" loop to fill the `neighbor_fs` and 
`neighbor_dfs` buffers using data from `fVec`.
"""
function initFs!(ts::MeshfreeTimeStepper, i, f_i, fVec, pg::ParticleGrid)
    # --- 4. Parallel Pre-Gather Loop ---
    # Get local aliases to the buffers for cleaner code in the loop
    neighbor_fs  = ts.neighbor_fs
    neighbor_dfs = ts.neighbor_dfs
    pointer = pg.neighbor.pointers[i]
    neighbor_slice = pointer:(pointer + pg.neighbor.amount[i] - 1)
    nb_indices = pg.neighbor.indices
    for k in neighbor_slice
        # `k` is the global index into the flat neighbor arrays
        
        # 1. GATHER: Get neighbor index `j`...
        j = nb_indices[k]
        # ...and then get its value `f_j`. This is the slow part.
        f_j = fVec[j] 
        #println(i,":",j)
        # 2. CALCULATE & STORE: Write to the pre-allocated buffers
        neighbor_fs[k]  = f_j
        neighbor_dfs[k] = f_j - f_i
    end
end
# A simple example for EulerUpwind adapted to the new structure
struct EulerUpwind{G1 <: GradientInterpolator, G2 <: GradientInterpolator, M <: MOODCriterion, GM <: GridMover} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::M
    moveGrid::GM
    
    # Buffers are now part of the struct to be reused
    rhoInit::Vector{Float64}      # Stores the state at the beginning of the step
    neighbor_fs::Vector{Float64}  # Pre-gathered neighbor values
    neighbor_dfs::Vector{Float64} # Pre-gathered neighbor differences

    function EulerUpwind(gradientInterpolator::G1, gm::GM; fallbackInterpolator::G2 = NoFallbackGrad(), mood::M = NoMOOD()) where {G1 <: GradientInterpolator, G2 <: GradientInterpolator, M <: MOODCriterion, GM <: GridMover}
        # Initialize with empty buffers
        new{G1, G2, M, GM}(gradientInterpolator, fallbackInterpolator, mood, gm, Float64[], Float64[], Float64[])
    end
end

function initAddTSBuffer!(eu::EulerUpwind, pg::ParticleGrid)
    num_particles = length(pg.neighbor.amount) 
        # --- 2. Resize Per-Particle Buffers (Size N) ---
    _ensure_capacity!(eu.rhoInit, num_particles)
end


"""
Functor for the EulerUpwind time stepper using the fused-loop structure.
"""
function (eu::EulerUpwind)(
    eq::ScalarHyperbolicPDE, 
    pg::ParticleGrid, 
    settings::SimSetting, 
    time::Real, 
    dt::Real
)
    N = pg.meta.N
    #initTS!(eu.pg)

    moveGrid = eu.moveGrid
    moveGrid(pg, dt)
    # --- 1. Preparation ---
    # Ensure buffers are correctly sized (only resizes if needed)
    initGIBuffers!(eu.gradientInterpolator, pg)
    initTSBuffer!(eu, pg) 

    # Copy initial state for the step
    eu.rhoInit[1:N] .= view(pg.rhos, 1:N)
    # apply_boundary_conditions!(pg, eu.rhoInit) # Apply BCs *before* pre-gather

    # Define chunks for parallel loops
    chunk_size = 100 # Adjust as needed
    chunks = collect(Iterators.partition(1:N, chunk_size))

    # --- 2. Fused Pre-Gather and Slope/Coefficient Calculation ---
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            fi = eu.rhoInit[p_idx]
            
            # Pre-gather neighbor data into ts.neighbor_fs/dfs
            # NOTE: Pass the correct rho vector (eu.rhoInit)
            initFs!(eu, p_idx, fi, eu.rhoInit, pg) 
            
            # Calculate slopes/coefficients needed by the gradient interpolator
            # NOTE: Pass the correct rho vector (eu.rhoInit)
            initGI!(eu.gradientInterpolator, p_idx, fi, pg, eu.neighbor_fs, eu.neighbor_dfs)
        end
    end
    
    # --- 3. Fused Divergence Calculation and Update ---
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            
            # Get initial state for this particle
            rho_initial = eu.rhoInit[p_idx]

            # Handle boundary particles: just keep initial state
            if pg.core.is_boundary[p_idx]
                continue 
            end
            
            # --- This code now only runs for INTERIOR particles ---
            # Get neighbor slice
            nb_slice = getNBSlice(pg, p_idx)

            # Calculate divergence using the local signature and pre-gathered data
            div = eu.gradientInterpolator(
                eq, 
                p_idx, 
                rho_initial, # Pass f_i from the start of the step 
                nb_slice, 
                pg, 
                eu.neighbor_fs, 
                eu.neighbor_dfs
            )
            rho_candidate = rho_initial - dt * div
            if !(eu.fallbackInterpolator isa NoFallbackGrad) && eu.mood(eu.gradientInterpolator, p_idx, rho_initial, nb_slice, rho_candidate, pg, eu.neighbor_fs)
                div = eu.fallbackInterpolator(eq, p_idx, rho_initial, nb_slice, pg, eu.neighbor_fs, eu.neighbor_dfs)
                rho_candidate = rho_initial - dt * div
            end           
            # Update particle state directly in the grid
            pg.rhos[p_idx] = rho_candidate
        end
    end
    # --- 4. Final Boundary Conditions ---
    apply_boundary_conditions!(pg, pg.rhos)
    
end

# """
# Initializes buffers specific to the EulerUpwind time stepper.
# """
# function initTSBuffer!(eu::EulerUpwind, pg::ParticleGrid)
#     N = pg.meta.N
#     M = length(pg.neighbor.indices) # Total number of interactions

#     _ensure_capacity!(eu.rhoInit, N)
#     _ensure_capacity!(eu.neighbor_fs, M)
#     _ensure_capacity!(eu.neighbor_dfs, M)
# end


# No longer needs Nx, Ny. Buffers are sized based on the grid passed during the call.
struct RalstonRK2{G1, G2, MOOD, GM} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    moveGrid::GM
    
    # Buffers are now part of the struct to be reused
    rhoInit::Vector{Float64}
    rhos::Vector{Float64}
    div1::Vector{Float64}

    # Buffers for efficient calculations
    neighbor_fs::Vector{Float64}
    neighbor_dfs::Vector{Float64}

    function RalstonRK2(grad::G1, fallback::G2, mood::M, gm::GM) where {G1 <: GradientInterpolator, G2 <: GradientInterpolator, M <: MOODCriterion, GM <: GridMover}
        # Initialize with empty buffers, they will be resized on the first step
        new{G1, G2, M, GM}(grad, fallback, mood, gm, Float64[], Float64[], Float64[], Float64[], Float64[])
    end
end

function initAddTSBuffer!(ralston::RalstonRK2, pg::ParticleGrid)
    # `num_particles` is the number of particles (N)
    num_particles = length(pg.neighbor.amount) 
        # --- 2. Resize Per-Particle Buffers (Size N) ---
    _ensure_capacity!(ralston.rhoInit, num_particles)
    _ensure_capacity!(ralston.rhos, num_particles)
    _ensure_capacity!(ralston.div1, num_particles)
end

# User-friendly constructor
function RalstonRK2(gradientInterpolator::G1; fallbackInterpolator::G2 = NoFallbackGrad(), mood::M = NoMOOD()) where {G1, G2, M}
    RalstonRK2(gradientInterpolator, fallbackInterpolator, mood)
end

function initTimeStepper(ralston::RalstonRK2, pg::ParticleGrid)
    sort_1d_particles!(pg)
    updateNeighbors!(pg)
end

# """
#     zero_vector_fields!(s)

# Iterates over all fields of a struct `s`. If a field is an `AbstractVector`,
# it fills that vector with zeros. This is useful for clearing workspace
# arrays for debugging.
# """
# function zero_vector_fields!(s)
#     for name in fieldnames(typeof(s))
#         field = getfield(s, name)
        
#         # Check if the field is a subtype of AbstractVector
#         if field isa AbstractVector
#             # Get the element type of the vector (e.g., Float64)
#             # and fill with the zero() of that type (e.g., 0.0)
#             fill_value = zero(eltype(field))
#             fill!(field, fill_value)
#         end
#     end
#     return s # Return the modified struct
# end

function (ralston::RalstonRK2)(eq::ScalarHyperbolicPDE, pg::ParticleGrid{D, M, S, WF}, settings::SimSetting, time::Real, dt::Real) where {D, M, S, WF}
    
    moveGrid = ralston.moveGrid
    moveGrid(pg, dt)
    #initTS!(ralston.pg)
    # --- Resize buffers only if necessary, using N ---
    initGIBuffers!(ralston.gradientInterpolator, pg)
    initGIBuffers!(ralston.fallbackInterpolator, pg)
    initTSBuffer!(ralston, pg)

    N = pg.meta.N
    # --- First RK Stage ---
    # 1. Start with the current, correct state of the grid
    ralston.rhoInit[1:N] .= view(pg.rhos, 1:N)

    # 4. Apply boundary conditions to the intermediate result stored in the buffer
    #apply_boundary_conditions!(pg, ralston.rhoInit)

    # Partition 1:N into chunks of 100, and schedule *those* dynamically
    Threads.@threads for p_idx in 1:N
            fi = ralston.rhoInit[p_idx]
            initFs!(ralston, p_idx, fi, ralston.rhoInit, pg)
            initGI!(ralston.gradientInterpolator, p_idx, fi, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
            initGI!(ralston.fallbackInterpolator, p_idx, fi, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
    end
    
    # 3. Calculate divergence for interior particles
    Threads.@threads for p_idx in 1:N
            if pg.core.is_boundary[p_idx]; continue; end # Skip ghost particles

            fi = ralston.rhoInit[p_idx]
            nb_slice = getNBSlice(pg, p_idx)

            ralston.div1[p_idx] = ralston.gradientInterpolator(eq, p_idx, fi, nb_slice, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
            rho_candidate = ralston.rhoInit[p_idx] - ralston.div1[p_idx] * dt * 2/3
            
            if !(ralston.fallbackInterpolator isa NoFallbackGrad) && ralston.mood(ralston.gradientInterpolator, p_idx, fi, nb_slice, rho_candidate, pg, ralston.neighbor_fs)
                ralston.div1[p_idx] = ralston.fallbackInterpolator(eq, p_idx, fi, nb_slice, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
                rho_candidate = ralston.rhoInit[p_idx] - ralston.div1[p_idx] * dt * 2/3
            end
            ralston.rhos[p_idx] = rho_candidate # Store intermediate result in the 'rhos' buffer
    end
    # 4. Apply boundary conditions to the intermediate result stored in the buffer
    apply_boundary_conditions!(pg, ralston.rhos)

    #moveGrid(pg, 1/3 * dt)

    N = pg.meta.N
    #updateNeighbors!(pg)
    #initTS!(ralston.pg)    
    initGIBuffers!(ralston.gradientInterpolator, pg)
    initGIBuffers!(ralston.fallbackInterpolator, pg)
    initTSBuffer!(ralston, pg)
    # Partition 1:N into chunks of 100, and schedule *those* dynamically
    Threads.@threads for p_idx in 1:N
            fi = ralston.rhos[p_idx]
            initFs!(ralston, p_idx, fi, ralston.rhos, pg)
            initGI!(ralston.gradientInterpolator, p_idx, fi, pg, ralston.neighbor_fs, ralston.neighbor_dfs) # ERROR IS HERE
            initGI!(ralston.fallbackInterpolator, p_idx, fi, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
    end

    # 2. Calculate final divergence for interior particles
    Threads.@threads for p_idx in 1:N
        if pg.core.is_boundary[p_idx]; continue; end # Skip ghost particles
    
        fi = ralston.rhos[p_idx]
        nb_slice = getNBSlice(pg, p_idx)
        # Pass the intermediate state (ralston.rhos) to the gradient calculation
        div2 = ralston.gradientInterpolator(eq, p_idx, fi, nb_slice, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
        rho_final = ralston.rhoInit[p_idx] - dt * (ralston.div1[p_idx] / 4 + 3 * div2 / 4)
        if !(ralston.fallbackInterpolator isa NoFallbackGrad) && ralston.mood(ralston.gradientInterpolator, p_idx, fi, nb_slice, rho_final, pg, ralston.neighbor_fs)
            div2 = ralston.fallbackInterpolator(eq, p_idx, fi, nb_slice, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
            rho_final = ralston.rhoInit[p_idx] - dt *  div2 / 3
        end
        # Directly write the final result for this particle into the grid
        pg.rhos[p_idx] = rho_final
    end
    # Final boundary condition application after the full step is complete
    apply_boundary_conditions!(pg, pg.rhos)
    
end

struct RK3{G1 <: GradientInterpolator, G2 <: GradientInterpolator, MOOD <: MOODCriterion} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    
    # --- Reusable Buffers (Workspace) ---
    rho_n::Vector{Float64}      # Stores the solution at the start of the step
    rho_stage1::Vector{Float64} # Stores the result of the first stage
    rho_stage2::Vector{Float64} # Stores the result of the second stage
    
    div1::Vector{Float64} # Stores divergence from stage 1
    div2::Vector{Float64} # Stores divergence from stage 2
    div3::Vector{Float64} # Stores divergence from stage 3

    # --- Buffers for efficient calculations (like in RK4) ---
    neighbor_fs::Vector{Float64}
    neighbor_dfs::Vector{Float64}

    function RK3(grad::G1, fallback::G2, mood::M) where {G1, G2, M}
        new{G1, G2, M}(grad, fallback, mood, 
            Float64[], Float64[], Float64[], # rho_n, rho_stage1, rho_stage2
            Float64[], Float64[], Float64[], # div1, div2, div3
            Float64[], Float64[]  # neighbor_fs, neighbor_dfs
        )
    end
end
# --- User-Friendly Constructor ---
function RK3(gradientInterpolator::G1; fallbackInterpolator::G2 = NoFallbackGrad(), mood::M = NoMOOD()) where {G1, G2, M}
    RK3(gradientInterpolator, fallbackInterpolator, mood)
end

# --- NEW: initAddTSBuffer! for RK3 ---
function initAddTSBuffer!(rk3::RK3, pg::ParticleGrid)
    num_particles = length(pg.neighbor.amount) 
    _ensure_capacity!(rk3.rho_n, num_particles)
    _ensure_capacity!(rk3.rho_stage1, num_particles)
    _ensure_capacity!(rk3.rho_stage2, num_particles)
    _ensure_capacity!(rk3.div1, num_particles)
    _ensure_capacity!(rk3.div2, num_particles)
    _ensure_capacity!(rk3.div3, num_particles)
end

# --- initTimeStepper function is removed (superseded by initGIBuffers!/initTSBuffer!) ---

function (rk3::RK3)(eq::ScalarHyperbolicPDE, pg::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    N = pg.meta.N
    
    # --- Define chunks for parallel loops ---
    chunk_size = 50 # Or any value you prefer
    chunks = collect(Iterators.partition(1:N, chunk_size))

    # ==================================================================
    # --- Stage 1: u^(1) = u^n - dt * div(u^n) ---
    # ==================================================================
    
    # 1.1: Init Buffers for Stage 1
    initGIBuffers!(rk3.gradientInterpolator, pg)
    initGIBuffers!(rk3.fallbackInterpolator, pg)
    initTSBuffer!(rk3, pg) # Resizes neighbor_fs/dfs and all rk3 buffers
    
    # --- Store Initial State ---
    rk3.rho_n[1:N] .= pg.rhos

    # 1.2: Threaded loop to calculate slopes/coefficients
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            fi = rk3.rho_n[p_idx]
            initFs!(rk3, p_idx, fi, rk3.rho_n, pg)
            initGI!(rk3.gradientInterpolator, p_idx, fi, pg, rk3.neighbor_fs, rk3.neighbor_dfs)
            initGI!(rk3.fallbackInterpolator, p_idx, fi, pg, rk3.neighbor_fs, rk3.neighbor_dfs)
        end
    end
    
    # 1.3: Threaded loop to calculate div1 and u^(1)
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            if pg.core.is_boundary[p_idx]; continue; end

            fi = rk3.rho_n[p_idx]
            nb_slice = getNBSlice(pg, p_idx)
            
            div1_val = rk3.gradientInterpolator(eq, p_idx, fi, nb_slice, pg, rk3.neighbor_fs, rk3.neighbor_dfs)
            
            rho_candidate = rk3.rho_n[p_idx] - dt * div1_val
            
            if !(rk3.fallbackInterpolator isa NoFallbackGrad) && rk3.mood(rk3.gradientInterpolator, p_idx, fi, nb_slice, rho_candidate, pg, rk3.neighbor_fs)
                div1_val = rk3.fallbackInterpolator(eq, p_idx, fi, nb_slice, pg, rk3.neighbor_fs, rk3.neighbor_dfs)
                rho_candidate = rk3.rho_n[p_idx] - dt * div1_val
            end
            rk3.div1[p_idx] = div1_val
            rk3.rho_stage1[p_idx] = rho_candidate
        end
    end

    # 1.4: Apply BCs to the intermediate state
    apply_boundary_conditions!(pg, rk3.rho_stage1)

    # ==================================================================
    # --- Stage 2: u^(2) = 3/4 u^n + 1/4 u^(1) - 1/4 dt * div(u^(1)) ---
    # ==================================================================
    
    # 2.1: Init Buffers for Stage 2
    initGIBuffers!(rk3.gradientInterpolator, pg)
    initGIBuffers!(rk3.fallbackInterpolator, pg)
    initTSBuffer!(rk3, pg)

    # 2.2: Threaded loop to calculate slopes/coefficients (using u^(1))
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            fi = rk3.rho_stage1[p_idx] # <-- Use u^(1)
            initFs!(rk3, p_idx, fi, rk3.rho_stage1, pg)
            initGI!(rk3.gradientInterpolator, p_idx, fi, pg, rk3.neighbor_fs, rk3.neighbor_dfs)
            initGI!(rk3.fallbackInterpolator, p_idx, fi, pg, rk3.neighbor_fs, rk3.neighbor_dfs)
        end
    end
    
    # 2.3: Threaded loop to calculate div2 and u^(2)
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            if pg.core.is_boundary[p_idx]; continue; end

            fi = rk3.rho_stage1[p_idx] # <-- Use u^(1)
            nb_slice = getNBSlice(pg, p_idx)
            
            div2_val = rk3.gradientInterpolator(eq, p_idx, fi, nb_slice, pg, rk3.neighbor_fs, rk3.neighbor_dfs)
            
            rho_candidate = 0.75 * rk3.rho_n[p_idx] + 0.25 * rk3.rho_stage1[p_idx] - 0.25 * dt * div2_val
            
            if !(rk3.fallbackInterpolator isa NoFallbackGrad) && rk3.mood(rk3.gradientInterpolator, p_idx, fi, nb_slice, rho_candidate, pg, rk3.neighbor_fs)
                div2_val = rk3.fallbackInterpolator(eq, p_idx, fi, nb_slice, pg, rk3.neighbor_fs, rk3.neighbor_dfs)
                rho_candidate = 0.75 * rk3.rho_n[p_idx] + 0.25 * rk3.rho_stage1[p_idx] - 0.25 * dt * div2_val
            end
            rk3.div2[p_idx] = div2_val
            rk3.rho_stage2[p_idx] = rho_candidate
        end
    end

    # 2.4: Apply BCs to the intermediate state
    apply_boundary_conditions!(pg, rk3.rho_stage2)

    # ==================================================================
    # --- Stage 3: u^{n+1} = 1/3 u^n + 2/3 u^(2) - 2/3 dt * div(u^(2)) ---
    # ==================================================================
    
    # 3.1: Init Buffers for Stage 3
    initGIBuffers!(rk3.gradientInterpolator, pg)
    initGIBuffers!(rk3.fallbackInterpolator, pg)
    initTSBuffer!(rk3, pg)

    # 3.2: Threaded loop to calculate slopes/coefficients (using u^(2))
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            fi = rk3.rho_stage2[p_idx] # <-- Use u^(2)
            initFs!(rk3, p_idx, fi, rk3.rho_stage2, pg)
            initGI!(rk3.gradientInterpolator, p_idx, fi, pg, rk3.neighbor_fs, rk3.neighbor_dfs)
            initGI!(rk3.fallbackInterpolator, p_idx, fi, pg, rk3.neighbor_fs, rk3.neighbor_dfs)
        end
    end
    
    # 3.3: Threaded loop to calculate div3 and Final Solution
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            if pg.core.is_boundary[p_idx]; continue; end

            fi = rk3.rho_stage2[p_idx] # <-- Use u^(2)
            nb_slice = getNBSlice(pg, p_idx)
            
            div3_val = rk3.gradientInterpolator(eq, p_idx, fi, nb_slice, pg, rk3.neighbor_fs, rk3.neighbor_dfs)
            
            rho_final = (1/3) * rk3.rho_n[p_idx] + (2/3) * rk3.rho_stage2[p_idx] - (2/3) * dt * div3_val
            
            if !(rk3.fallbackInterpolator isa NoFallbackGrad) && rk3.mood(rk3.gradientInterpolator, p_idx, fi, nb_slice, rho_final, pg, rk3.neighbor_fs)
                div3_val = rk3.fallbackInterpolator(eq, p_idx, fi, nb_slice, pg, rk3.neighbor_fs, rk3.neighbor_dfs)
                rho_final = (1/3) * rk3.rho_n[p_idx] + (2/3) * rk3.rho_stage2[p_idx] - (2/3) * dt * div3_val
            end
            rk3.div3[p_idx] = div3_val
            pg.rhos[p_idx] = rho_final # Write final solution
        end
    end

    # 3.4: Final Boundary Condition Application
    apply_boundary_conditions!(pg, pg.rhos)
    
end

struct RK4{G1, G2, MOOD} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    
    # --- Reusable Buffers (Workspace) ---
    rho_n::Vector{Float64}
    rho_stage::Vector{Float64} # A single buffer for all intermediate stages
    
    k1::Vector{Float64} # Stores divergence from stage 1
    k2::Vector{Float64} # Stores divergence from stage 2
    k3::Vector{Float64} # Stores divergence from stage 3
    k4::Vector{Float64} # Stores divergence from stage 4

    # --- Buffers for efficient calculations (like in RK2) ---
    neighbor_fs::Vector{Float64}
    neighbor_dfs::Vector{Float64}

    function RK4(grad::G1, fallback::G2, mood::M) where {G1, G2, M}
        new{G1, G2, M}(grad, fallback, mood, 
            Float64[], Float64[], # rho_n, rho_stage
            Float64[], Float64[], Float64[], Float64[], # k1-k4
            Float64[], Float64[]  # neighbor_fs, neighbor_dfs
        )
    end
end

# --- User-Friendly Constructor ---
function RK4(gradientInterpolator::G1; fallbackInterpolator::G2 = NoFallbackGrad(), mood::M = NoMOOD()) where {G1, G2, M}
    RK4(gradientInterpolator, fallbackInterpolator, mood)
end

function initAddTSBuffer!(rk4::RK4, pg::ParticleGrid)
    num_particles = length(pg.neighbor.amount) 
    _ensure_capacity!(rk4.rho_n, num_particles)
    _ensure_capacity!(rk4.rho_stage, num_particles)
    _ensure_capacity!(rk4.k1, num_particles)
    _ensure_capacity!(rk4.k2, num_particles)
    _ensure_capacity!(rk4.k3, num_particles)
    _ensure_capacity!(rk4.k4, num_particles)
end
function (rk4::RK4)(eq::ScalarHyperbolicPDE, pg::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    N = pg.meta.N
    
    # --- Define chunks for parallel loops ---
    chunk_size = 50 # Or any value you prefer
    chunks = collect(Iterators.partition(1:N, chunk_size))

    # ==================================================================
    # --- Stage 1: Calculate k1 = div(u^n) ---
    # ==================================================================
    
    # 1.1: Init Buffers for Stage 1
    initGIBuffers!(rk4.gradientInterpolator, pg)
    initGIBuffers!(rk4.fallbackInterpolator, pg)
    initTSBuffer!(rk4, pg) # Resizes neighbor_fs/dfs and k1-k4 etc.
    # --- Store Initial State ---
    rk4.rho_n[1:N] .= pg.rhos

    # 1.2: Threaded loop to calculate slopes/coefficients
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            fi = rk4.rho_n[p_idx]
            initFs!(rk4, p_idx, fi, rk4.rho_n, pg)
            initGI!(rk4.gradientInterpolator, p_idx, fi, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
            initGI!(rk4.fallbackInterpolator, p_idx, fi, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
        end
    end
    
    # 1.3: Threaded loop to calculate k1 (divergence)
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            if pg.core.is_boundary[p_idx]; continue; end

            fi = rk4.rho_n[p_idx]
            nb_slice = getNBSlice(pg, p_idx)
            
            k1_val = rk4.gradientInterpolator(eq, p_idx, fi, nb_slice, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
            
            rho_candidate = rk4.rho_n[p_idx] - 0.5 * dt * k1_val # u^(1) candidate
            if !(rk4.fallbackInterpolator isa NoFallbackGrad) && rk4.mood(rk4.gradientInterpolator, p_idx, fi, nb_slice, rho_candidate, pg, rk4.neighbor_fs)
                k1_val = rk4.fallbackInterpolator(eq, p_idx, fi, nb_slice, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
            end
            rk4.k1[p_idx] = k1_val
        end
    end

    # 1.4: Compute intermediate state u^(1) and apply BCs
    @. rk4.rho_stage = rk4.rho_n - 0.5 * dt * rk4.k1
    apply_boundary_conditions!(pg, rk4.rho_stage)

    # ==================================================================
    # --- Stage 2: Calculate k2 = div(u^(1)) ---
    # ==================================================================
    
    # 2.1: Init Buffers for Stage 2
    initGIBuffers!(rk4.gradientInterpolator, pg)
    initGIBuffers!(rk4.fallbackInterpolator, pg)
    initTSBuffer!(rk4, pg)

    # 2.2: Threaded loop to calculate slopes/coefficients (using u^(1))
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            fi = rk4.rho_stage[p_idx] # <-- Use u^(1) from rho_stage
            initFs!(rk4, p_idx, fi, rk4.rho_stage, pg)
            initGI!(rk4.gradientInterpolator, p_idx, fi, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
            initGI!(rk4.fallbackInterpolator, p_idx, fi, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
        end
    end
    
    # 2.3: Threaded loop to calculate k2 (divergence)
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            if pg.core.is_boundary[p_idx]; continue; end

            fi = rk4.rho_stage[p_idx] # <-- Use u^(1)
            nb_slice = getNBSlice(pg, p_idx)
            
            k2_val = rk4.gradientInterpolator(eq, p_idx, fi, nb_slice, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
            
            rho_candidate = rk4.rho_n[p_idx] - 0.5 * dt * k2_val # u^(2) candidate
            if !(rk4.fallbackInterpolator isa NoFallbackGrad) && rk4.mood(rk4.gradientInterpolator, p_idx, fi, nb_slice, rho_candidate, pg, rk4.neighbor_fs)
                k2_val = rk4.fallbackInterpolator(eq, p_idx, fi, nb_slice, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
            end
            rk4.k2[p_idx] = k2_val
        end
    end

    # 2.4: Compute intermediate state u^(2) and apply BCs
    @. rk4.rho_stage = rk4.rho_n - 0.5 * dt * rk4.k2 # Overwrite rho_stage
    apply_boundary_conditions!(pg, rk4.rho_stage)

    # ==================================================================
    # --- Stage 3: Calculate k3 = div(u^(2)) ---
    # ==================================================================
    
    # 3.1: Init Buffers for Stage 3
    initGIBuffers!(rk4.gradientInterpolator, pg)
    initGIBuffers!(rk4.fallbackInterpolator, pg)
    initTSBuffer!(rk4, pg)

    # 3.2: Threaded loop to calculate slopes/coefficients (using u^(2))
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            fi = rk4.rho_stage[p_idx] # <-- Use u^(2) from rho_stage
            initFs!(rk4, p_idx, fi, rk4.rho_stage, pg)
            initGI!(rk4.gradientInterpolator, p_idx, fi, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
            initGI!(rk4.fallbackInterpolator, p_idx, fi, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
        end
    end
    
    # 3.3: Threaded loop to calculate k3 (divergence)
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            if pg.core.is_boundary[p_idx]; continue; end

            fi = rk4.rho_stage[p_idx] # <-- Use u^(2)
            nb_slice = getNBSlice(pg, p_idx)
            
            k3_val = rk4.gradientInterpolator(eq, p_idx, fi, nb_slice, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
            
            rho_candidate = rk4.rho_n[p_idx] - dt * k3_val # u^(3) candidate
            if !(rk4.fallbackInterpolator isa NoFallbackGrad) && rk4.mood(rk4.gradientInterpolator, p_idx, fi, nb_slice, rho_candidate, pg, rk4.neighbor_fs)
                k3_val = rk4.fallbackInterpolator(eq, p_idx, fi, nb_slice, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
            end
            rk4.k3[p_idx] = k3_val
        end
    end

    # 3.4: Compute intermediate state u^(3) and apply BCs
    @. rk4.rho_stage = rk4.rho_n - dt * rk4.k3 # Overwrite rho_stage
    apply_boundary_conditions!(pg, rk4.rho_stage)

    # ==================================================================
    # --- Stage 4: Calculate k4 = div(u^(3)) and Final Solution ---
    # ==================================================================
    
    # 4.1: Init Buffers for Stage 4
    initGIBuffers!(rk4.gradientInterpolator, pg)
    initGIBuffers!(rk4.fallbackInterpolator, pg)
    initTSBuffer!(rk4, pg)

    # 4.2: Threaded loop to calculate slopes/coefficients (using u^(3))
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            fi = rk4.rho_stage[p_idx] # <-- Use u^(3) from rho_stage
            initFs!(rk4, p_idx, fi, rk4.rho_stage, pg)
            initGI!(rk4.gradientInterpolator, p_idx, fi, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
            initGI!(rk4.fallbackInterpolator, p_idx, fi, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
        end
    end
    
    # 4.3: Threaded loop to calculate k4 and Final Solution
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            if pg.core.is_boundary[p_idx]; continue; end

            fi = rk4.rho_stage[p_idx] # <-- Use u^(3)
            nb_slice = getNBSlice(pg, p_idx)
            
            k4_val = rk4.gradientInterpolator(eq, p_idx, fi, nb_slice, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
            rk4.k4[p_idx] = k4_val # Store k4
            
            rho_final = rk4.rho_n[p_idx] - (dt/6) * (rk4.k1[p_idx] + 2*rk4.k2[p_idx] + 2*rk4.k3[p_idx] + k4_val)
            
            if !(rk4.fallbackInterpolator isa NoFallbackGrad) && rk4.mood(rk4.gradientInterpolator, p_idx, fi, nb_slice, rho_final, pg, rk4.neighbor_fs)
                # Fallback to Euler step using u^(3) and k4
                pg.rhos[p_idx] = rk4.rho_stage[p_idx] - dt * rk4.k4[p_idx]
            else
                pg.rhos[p_idx] = rho_final
            end
        end
    end

    # 4.4: Final Boundary Condition Application
    apply_boundary_conditions!(pg, pg.rhos)
    
end



struct RalstonRK2SmoothSwitch{G1, G2, MOOD} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    tol::Float64
    
    # --- Reusable Buffers (Workspace) ---
    rho_n::Vector{Float64}
    rho_stage::Vector{Float64}
    rho_fallback::Vector{Float64}
    div1::Vector{Float64}
    
    # --- Propagation Buffers ---
    mood_indices::Vector{Int}
    prop_indices::Vector{Int}
    
    # Per-step flag to track which particles have been switched to fallback
    switched_to_fallback::BitVector

    # --- Buffers for efficient calculations (like in RK4) ---
    neighbor_fs::Vector{Float64}
    neighbor_dfs::Vector{Float64}

    function RalstonRK2SmoothSwitch(grad::G1, fallback::G2, mood::M; tol=1e-7) where {G1, G2, M}
        new{G1, G2, M}(grad, fallback, mood, tol,
            Float64[], Float64[], Float64[], Float64[], # Main buffers
            Int[], Int[], # Propagation buffers
            falses(0),    # Flag buffer
            Float64[], Float64[] # neighbor_fs, neighbor_dfs
        )
    end
end

# --- User-Friendly Constructor ---
function RalstonRK2SmoothSwitch(gradientInterpolator::G1; fallbackInterpolator::G2 = gradientInterpolator, mood::M = NoMOOD(), tol = 1e-7) where {G1, G2, M}
    RalstonRK2SmoothSwitch(gradientInterpolator, fallbackInterpolator, mood; tol=tol)
end

function initAddTSBuffer!(ralston::RalstonRK2SmoothSwitch, pg::ParticleGrid)
    num_particles = length(pg.neighbor.amount) 
    _ensure_capacity!(ralston.rho_n, num_particles)
    _ensure_capacity!(ralston.rho_stage, num_particles)
    _ensure_capacity!(ralston.rho_fallback, num_particles)
    _ensure_capacity!(ralston.div1, num_particles)
    
    # No need to resize mood_indices/prop_indices, 
    # as `push!` will grow them.
    
    # Resize BitVector
    if length(ralston.switched_to_fallback) < num_particles
        resize!(ralston.switched_to_fallback, num_particles)
    end
end
function (ralston::RalstonRK2SmoothSwitch)(eq::ScalarHyperbolicPDE, pg::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    N = pg.meta.N
    interior = 1:N # Assuming interior_indices is 1:N for now
    
    # --- Define chunks for parallel loops ---
    chunk_size = 50 
    chunks = collect(Iterators.partition(1:N, chunk_size))



    # ==================================================================
    # --- 1. Calculate Full Fallback Solution and Target Mass ---
    # ==================================================================
    
    # 1.1: Init Buffers for Fallback
    initGIBuffers!(ralston.fallbackInterpolator, pg)
    initTSBuffer!(ralston, pg) # Resizes neighbor_fs/dfs and all ralston buffers
    # --- Store Initial State ---
    ralston.rho_n[1:N] .= pg.rhos
    # 1.2: Threaded loop to calculate slopes/coefficients
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            fi = ralston.rho_n[p_idx]
            initFs!(ralston, p_idx, fi, ralston.rho_n, pg)
            # Use fallback interpolator for GI
            initGI!(ralston.fallbackInterpolator, p_idx, fi, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
        end
    end

    # 1.3: Threaded loop to calculate fallback divergence
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            if pg.core.is_boundary[p_idx]; continue; end

            fi = ralston.rho_n[p_idx]
            nb_slice = getNBSlice(pg, p_idx)
            
            div_fallback = ralston.fallbackInterpolator(eq, p_idx, fi, nb_slice, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
            ralston.rho_fallback[p_idx] = ralston.rho_n[p_idx] - div_fallback * dt
        end
    end
    
    # 1.4: Calculate target mass (Vectorized)
    # Assumes pg.volumes is available
    target_mass = dot(@view(ralston.rho_fallback[interior]), @view(pg.volumes[interior]))

    # ==================================================================
    # --- 2. Perform High-Order RalstonRK2 Step ---
    # ==================================================================

    # 2.1: Init Buffers for Stage 1
    initGIBuffers!(ralston.gradientInterpolator, pg)
    initTSBuffer!(ralston, pg)

    # 2.2: Threaded loop to calculate slopes/coefficients (High-order)
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            fi = ralston.rho_n[p_idx]
            initFs!(ralston, p_idx, fi, ralston.rho_n, pg)
            # Use high-order interpolator for GI
            initGI!(ralston.gradientInterpolator, p_idx, fi, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
        end
    end

    # 2.3: Threaded loop to calculate div1 and u^(1)
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            if pg.core.is_boundary[p_idx]; continue; end

            fi = ralston.rho_n[p_idx]
            nb_slice = getNBSlice(pg, p_idx)
            
            ralston.div1[p_idx] = ralston.gradientInterpolator(eq, p_idx, fi, nb_slice, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
            ralston.rho_stage[p_idx] = ralston.rho_n[p_idx] - ralston.div1[p_idx] * dt * 2/3
        end
    end

    # 2.4: Apply BCs to intermediate stage
    apply_boundary_conditions!(pg, ralston.rho_stage)

    # ==================================================================
    # --- 3. Final Stage (High-Order) & Serial MOOD Check ---
    # ==================================================================

    # 3.1: Init Buffers for Stage 2
    initGIBuffers!(ralston.gradientInterpolator, pg)
    initTSBuffer!(ralston, pg)

    # 3.2: Threaded loop to calculate slopes/coefficients (High-order, using u^(1))
    Threads.@threads for particle_range in chunks
        for p_idx in particle_range
            fi = ralston.rho_stage[p_idx]
            initFs!(ralston, p_idx, fi, ralston.rho_stage, pg)
            initGI!(ralston.gradientInterpolator, p_idx, fi, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
        end
    end

    # 3.3: SERIAL loop for final divergence, MOOD check, and mass calculation
    current_mass = 0.0
    empty!(ralston.mood_indices)
    fill!(ralston.switched_to_fallback, false)

    for p_idx in interior
        if pg.core.is_boundary[p_idx]
            # For boundary, just use the fallback value to contribute to mass
            current_mass += ralston.rho_fallback[p_idx] * pg.volumes[p_idx]
            continue
        end

        fi = ralston.rho_stage[p_idx]
        nb_slice = getNBSlice(pg, p_idx)

        div2 = ralston.gradientInterpolator(eq, p_idx, fi, nb_slice, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
        rho_final_candidate = ralston.rho_n[p_idx] - dt * (ralston.div1[p_idx]/4 + 3*div2/4)
        
        # --- Initial MOOD Check ---
        if ralston.mood(ralston.gradientInterpolator, p_idx, fi, nb_slice, rho_final_candidate, pg, ralston.neighbor_fs)
            pg.rhos[p_idx] = ralston.rho_fallback[p_idx]
            push!(ralston.mood_indices, p_idx)
            ralston.switched_to_fallback[p_idx] = true
        else
            pg.rhos[p_idx] = rho_final_candidate
        end
        current_mass += pg.rhos[p_idx] * pg.volumes[p_idx]
    end

    # ==================================================================
    # --- 4. Mass Conservation Propagation Loop (Serial) ---
    # ==================================================================
    if !isempty(ralston.mood_indices)
        # Build the initial propagation list from neighbors of MOOD events
        empty!(ralston.prop_indices)
        for p_idx in ralston.mood_indices
            for nb_idx in pg.neighbor.indices[p_idx]
                # Only add interior neighbors that haven't been switched yet
                if nb_idx in interior && !ralston.switched_to_fallback[nb_idx]
                    push!(ralston.prop_indices, nb_idx)
                end
            end
        end
        unique!(ralston.prop_indices) # Remove duplicates

        while abs(target_mass - current_mass) > ralston.tol && !isempty(ralston.prop_indices)
            p_idx = popfirst!(ralston.prop_indices)
            
            if ralston.switched_to_fallback[p_idx]; continue; end # Already switched
            
            # Switch this particle to the low-order solution
            local_mass_change = (ralston.rho_fallback[p_idx] - pg.rhos[p_idx]) * pg.volumes[p_idx]
            current_mass += local_mass_change
            pg.rhos[p_idx] = ralston.rho_fallback[p_idx]
            ralston.switched_to_fallback[p_idx] = true

            # Add its neighbors to the propagation list
            for nb_idx in pg.neighbor.indices[p_idx]
                if nb_idx in interior && !ralston.switched_to_fallback[nb_idx]
                    push!(ralston.prop_indices, nb_idx)
                end
            end
        end
    end

    # --- 5. Final Boundary Condition Application ---
    apply_boundary_conditions!(pg, pg.rhos)
end




