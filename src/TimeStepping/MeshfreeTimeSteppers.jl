# =========================================================================
# BUFFER INITIALIZATION
# =========================================================================

"""
    initFs!(ts::MeshfreeTimeStepper, i, f_i, nb_slice, fVec, pg::ParticleGrid)

Parallel "pre-gather" loop to fill the `neighbor_fs` and `neighbor_dfs` 
buffers using data from `fVec`. Works natively with SVector states.
"""
@inline function initFs!(ts::MeshfreeTimeStepper, i::Int, f_i::State{M}, nb_slice::UnitRange{Int}, fVec::AbstractVector{State{M}}, pg::ParticleGrid) where {M}
    neighbor_fs  = ts.neighbor_fs
    neighbor_dfs = ts.neighbor_dfs
    nb_indices   = get_neighbors(pg)
    
    @inbounds for k in nb_slice
        j = nb_indices[k]
        f_j = fVec[j] 
        
        neighbor_fs[k]  = f_j
        # SVector natively supports component-wise subtraction!
        neighbor_dfs[k] = f_j - f_i 
    end
end

function initTSBuffer!(eu::EulerUpwind{M}, pg::ParticleGrid) where {M}
    N = pg.meta.N
    M_neighbors = length(pg.neighbor.indices)

    if length(eu.rhoInit) < N
        resize!(eu.rhoInit, ceil(Int, N * 1.25))
    end
    if length(eu.neighbor_fs) < M_neighbors
        new_cap = ceil(Int, M_neighbors * 1.25)
        resize!(eu.neighbor_fs, new_cap)
        resize!(eu.neighbor_dfs, new_cap)
    end
end

function initTSBuffer!(ralston::RalstonRK2{M}, pg::ParticleGrid) where {M}
    N = pg.meta.N
    M_neighbors = length(pg.neighbor.indices)

    if length(ralston.rhoInit) < N
        new_cap = ceil(Int, N * 1.25)
        resize!(ralston.rhoInit, new_cap)
        resize!(ralston.rhos, new_cap)
        resize!(ralston.div1, new_cap)
    end
    if length(ralston.neighbor_fs) < M_neighbors
        new_cap = ceil(Int, M_neighbors * 1.25)
        resize!(ralston.neighbor_fs, new_cap)
        resize!(ralston.neighbor_dfs, new_cap)
    end
end

# =========================================================================
# 1. EULER UPWIND
# =========================================================================

"""
Functor for the EulerUpwind time stepper using the fused-loop structure.
"""
function (eu::EulerUpwind{M})(
    eq::HyperbolicPDE, 
    pg::ParticleGrid, 
    settings::SimSetting, 
    time::Real, 
    dt::Real
) where {M}
    N = pg.meta.N

    # --- 1. Grid Movement ---
    pg.mover(pg, dt)
    # Ensure neighbors are updated after movement
    pg.neighbor(pg)
   
    # --- 2. Preparation ---
    initGIBuffers!(eu.gradientInterpolator, pg)
    initTSBuffer!(eu, pg) 
    nb_slices = pg.neighbor.ranges

    # Copy initial state for the step
    eu.rhoInit[1:N] .= view(pg.rhos, 1:N)

    chunk_size = 100 
    chunks = collect(Iterators.partition(1:N, chunk_size))

    # --- 3. Fused Pre-Gather and Slope Calculation ---
    @batch for particle_range in chunks
        for p_idx in particle_range
            fi = eu.rhoInit[p_idx]
            nb_slice = nb_slices[p_idx]
            
            initFs!(eu, p_idx, fi, nb_slice, eu.rhoInit, pg) 
            initGI!(eu.gradientInterpolator, p_idx, fi, nb_slice, pg, eu.neighbor_fs, eu.neighbor_dfs)
        end
    end
    
    # --- 4. Fused Divergence Calculation and Update ---
    @batch for particle_range in chunks
        for p_idx in particle_range
            if pg.core.is_boundary[p_idx]; continue; end 
            
            rho_initial = eu.rhoInit[p_idx]
            nb_slice = nb_slices[p_idx]

            div = eu.gradientInterpolator(eq, p_idx, rho_initial, nb_slice, pg, eu.neighbor_fs, eu.neighbor_dfs)
            
            # SVector Math!
            rho_candidate = rho_initial - dt * div
            
            if !(eu.fallbackInterpolator isa NoFallbackGrad) && eu.mood(eu.gradientInterpolator, p_idx, rho_initial, nb_slice, rho_candidate, pg, eu.neighbor_fs)
                div = eu.fallbackInterpolator(eq, p_idx, rho_initial, nb_slice, pg, eu.neighbor_fs, eu.neighbor_dfs)
                rho_candidate = rho_initial - dt * div
            end           
            
            pg.rhos[p_idx] = rho_candidate
        end
    end
    
    # --- 5. Final Boundary Conditions ---
    apply_boundary_conditions!(pg, pg.rhos)
end

# =========================================================================
# 2. RALSTON RK2
# =========================================================================

function (ralston::RalstonRK2{M})(
    eq::HyperbolicPDE, 
    pg::ParticleGrid, 
    settings::SimSetting, 
    time::Real, 
    dt::Real
) where {M}
    N = pg.meta.N

    # ==================================================================
    # --- STAGE 1: Calculate intermediate state ---
    # ==================================================================
    
    # Move grid for the first RK stage and update neighbors
    pg.mover(pg, (2.0/3.0) * dt)
    #pg.reorder(pg)
    pg.neighbor(pg)
    
    initGIBuffers!(ralston.gradientInterpolator, pg)
    initGIBuffers!(ralston.fallbackInterpolator, pg)
    initTSBuffer!(ralston, pg)
    nb_slices = pg.neighbor.ranges

    # Store initial state
    ralston.rhoInit[1:N] .= view(pg.rhos, 1:N)

    @batch for p_idx in 1:N
        fi = ralston.rhoInit[p_idx]
        nb_slice = nb_slices[p_idx]
        
        initFs!(ralston, p_idx, fi, nb_slice, ralston.rhoInit, pg)
        
        idx = pg.core.is_boundary[p_idx] ? -p_idx : p_idx
        initGI!(ralston.gradientInterpolator, idx, fi, nb_slice, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
        initGI!(ralston.fallbackInterpolator, idx, fi, nb_slice, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
    end
    
    @batch for p_idx in 1:N
        if pg.core.is_boundary[p_idx]; continue; end 

        fi = ralston.rhoInit[p_idx]
        nb_slice = nb_slices[p_idx]

        div1_val = ralston.gradientInterpolator(eq, p_idx, fi, nb_slice, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
        rho_candidate = fi - dt * (2.0/3.0) * div1_val
        
        if !(ralston.fallbackInterpolator isa NoFallbackGrad) && ralston.mood(ralston.gradientInterpolator, p_idx, fi, nb_slice, rho_candidate, pg, ralston.neighbor_fs)
            div1_val = ralston.fallbackInterpolator(eq, p_idx, fi, nb_slice, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
            rho_candidate = fi - dt * (2.0/3.0) * div1_val
        end
        
        ralston.div1[p_idx] = div1_val
        ralston.rhos[p_idx] = rho_candidate 
    end
    
    apply_boundary_conditions!(pg, ralston.rhos)

    # ==================================================================
    # --- STAGE 2: Calculate final state ---
    # ==================================================================
    
    # Move grid for the second RK stage and update neighbors
    pg.mover(pg, (1.0/3.0) * dt)
    pg.neighbor(pg)

    initGIBuffers!(ralston.gradientInterpolator, pg)
    initGIBuffers!(ralston.fallbackInterpolator, pg)
    initTSBuffer!(ralston, pg)
    nb_slices = pg.neighbor.ranges

    @batch for p_idx in 1:N
        fi = ralston.rhos[p_idx] # <-- Use intermediate state
        nb_slice = nb_slices[p_idx]
        
        initFs!(ralston, p_idx, fi, nb_slice, ralston.rhos, pg)
        initGI!(ralston.gradientInterpolator, p_idx, fi, nb_slice, pg, ralston.neighbor_fs, ralston.neighbor_dfs) 
        initGI!(ralston.fallbackInterpolator, p_idx, fi, nb_slice, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
    end

    @batch for p_idx in 1:N
            if pg.core.is_boundary[p_idx]; continue; end 
    
            fi = ralston.rhos[p_idx]
            nb_slice = nb_slices[p_idx]
            
            div2 = ralston.gradientInterpolator(eq, p_idx, fi, nb_slice, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
            
            # SVector Math!
            rho_final = ralston.rhoInit[p_idx] - dt * (0.25 * ralston.div1[p_idx] + 0.75 * div2)
            
            if !(ralston.fallbackInterpolator isa NoFallbackGrad) && ralston.mood(ralston.gradientInterpolator, p_idx, fi, nb_slice, rho_final, pg, ralston.neighbor_fs)
                div2 = ralston.fallbackInterpolator(eq, p_idx, fi, nb_slice, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
                rho_final = ralston.rhoInit[p_idx] - dt * div2 # Ralston fallback math
            end
            
            pg.rhos[p_idx] = rho_final
    end
    
    apply_boundary_conditions!(pg, pg.rhos)
end

function RK3(grad::G1, fallback::G2, mood::M) where {G1, G2, M}
    RK3{G1, G2, M}(grad, fallback, mood, 
        Float64[], Float64[], Float64[], # rho_n, rho_stage1, rho_stage2
        Float64[], Float64[], Float64[], # div1, div2, div3
        Float64[], Float64[]  # neighbor_fs, neighbor_dfs
    )
end

# --- User-Friendly Constructor ---
function RK3(gradientInterpolator::G1; fallbackInterpolator::G2 = NoFallbackGrad(), mood::M = NoMOOD()) where {G1, G2, M}
    RK3(gradientInterpolator, fallbackInterpolator, mood)
end

# --- NEW: initAddTSBuffer! for RK3 ---
function initAddTSBuffer!(rk3::RK3, pg::ParticleGrid)
    num_particles = length(pg.neighbor.ranges) 
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
    nb_slices = pg.neighbor.ranges
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
    @batch for particle_range in chunks
        for p_idx in particle_range
            fi = rk3.rho_n[p_idx]
            nb_slice = nb_slices[p_idx]
            initFs!(rk3, p_idx, fi, nb_slice, rk3.rho_n, pg)
            initGI!(rk3.gradientInterpolator, p_idx, fi, nb_slice, pg, rk3.neighbor_fs, rk3.neighbor_dfs)
            initGI!(rk3.fallbackInterpolator, p_idx, fi, nb_slice, pg, rk3.neighbor_fs, rk3.neighbor_dfs)
        end
    end
    
    # 1.3: Threaded loop to calculate div1 and u^(1)
    @batch for particle_range in chunks
        for p_idx in particle_range
            if pg.core.is_boundary[p_idx]; continue; end

            fi = rk3.rho_n[p_idx]
            nb_slice = nb_slices[p_idx]
            
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
    @batch for particle_range in chunks
        for p_idx in particle_range
            fi = rk3.rho_stage1[p_idx] # <-- Use u^(1)
            nb_slice = nb_slices[p_idx]
            initFs!(rk3, p_idx, fi, nb_slice, rk3.rho_stage1, pg)
            initGI!(rk3.gradientInterpolator, p_idx, fi, nb_slice, pg, rk3.neighbor_fs, rk3.neighbor_dfs)
            initGI!(rk3.fallbackInterpolator, p_idx, fi, nb_slice, pg, rk3.neighbor_fs, rk3.neighbor_dfs)
        end
    end
    
    # 2.3: Threaded loop to calculate div2 and u^(2)
    @batch for particle_range in chunks
        for p_idx in particle_range
            if pg.core.is_boundary[p_idx]; continue; end

            fi = rk3.rho_stage1[p_idx] # <-- Use u^(1)
            nb_slice = nb_slices[p_idx]
            
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
    @batch for particle_range in chunks
        for p_idx in particle_range
            fi = rk3.rho_stage2[p_idx] # <-- Use u^(2)
            nb_slice = nb_slices[p_idx]
            initFs!(rk3, p_idx, fi, nb_slice, rk3.rho_stage2, pg)
            initGI!(rk3.gradientInterpolator, p_idx, fi, nb_slice, pg, rk3.neighbor_fs, rk3.neighbor_dfs)
            initGI!(rk3.fallbackInterpolator, p_idx, fi, nb_slice, pg, rk3.neighbor_fs, rk3.neighbor_dfs)
        end
    end
    
    # 3.3: Threaded loop to calculate div3 and Final Solution
    @batch for particle_range in chunks
        for p_idx in particle_range
            if pg.core.is_boundary[p_idx]; continue; end

            fi = rk3.rho_stage2[p_idx] # <-- Use u^(2)
            nb_slice = nb_slices[p_idx]
            
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

function RK4(grad::G1, fallback::G2, mood::M) where {G1, G2, M}
    RK4{G1, G2, M}(grad, fallback, mood, 
        Float64[], Float64[], # rho_n, rho_stage
        Float64[], Float64[], Float64[], Float64[], # k1-k4
        Float64[], Float64[]  # neighbor_fs, neighbor_dfs
    )
end

# --- User-Friendly Constructor ---
function RK4(gradientInterpolator::G1; fallbackInterpolator::G2 = NoFallbackGrad(), mood::M = NoMOOD()) where {G1, G2, M}
    RK4(gradientInterpolator, fallbackInterpolator, mood)
end

function initAddTSBuffer!(rk4::RK4, pg::ParticleGrid)
    num_particles = length(pg.neighbor.ranges) 
    _ensure_capacity!(rk4.rho_n, num_particles)
    _ensure_capacity!(rk4.rho_stage, num_particles)
    _ensure_capacity!(rk4.k1, num_particles)
    _ensure_capacity!(rk4.k2, num_particles)
    _ensure_capacity!(rk4.k3, num_particles)
    _ensure_capacity!(rk4.k4, num_particles)
end
function (rk4::RK4)(eq::ScalarHyperbolicPDE, pg::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    N = pg.meta.N
    nb_slices = pg.neighbor.ranges
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
    @batch for particle_range in chunks
        for p_idx in particle_range
            fi = rk4.rho_n[p_idx]
            nb_slice = nb_slices[p_idx]
            initFs!(rk4, p_idx, fi, nb_slice, rk4.rho_n, pg)
            initGI!(rk4.gradientInterpolator, p_idx, fi, nb_slice, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
            initGI!(rk4.fallbackInterpolator, p_idx, fi, nb_slice, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
        end
    end
    
    # 1.3: Threaded loop to calculate k1 (divergence)
    @batch for particle_range in chunks
        for p_idx in particle_range
            if pg.core.is_boundary[p_idx]; continue; end

            fi = rk4.rho_n[p_idx]
            nb_slice = nb_slices[p_idx]
            
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
    @batch for particle_range in chunks
        for p_idx in particle_range
            fi = rk4.rho_stage[p_idx] # <-- Use u^(1) from rho_st
            nb_slice = nb_slices[p_idx]age
            initFs!(rk4, p_idx, fi, nb_slice, rk4.rho_stage, pg)
            initGI!(rk4.gradientInterpolator, p_idx, fi, nb_slice, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
            initGI!(rk4.fallbackInterpolator, p_idx, fi, nb_slice, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
        end
    end
    
    # 2.3: Threaded loop to calculate k2 (divergence)
    @batch for particle_range in chunks
        for p_idx in particle_range
            if pg.core.is_boundary[p_idx]; continue; end

            fi = rk4.rho_stage[p_idx] # <-- Use u^(1)
            nb_slice = nb_slices[p_idx]
            
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
    @batch for particle_range in chunks
        for p_idx in particle_range
            fi = rk4.rho_stage[p_idx] # <-- Use u^(2) from rho_stage
            nb_slice = nb_slices[p_idx]
            initFs!(rk4, p_idx, fi, nb_slice, rk4.rho_stage, pg)
            initGI!(rk4.gradientInterpolator, p_idx, fi, nb_slice, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
            initGI!(rk4.fallbackInterpolator, p_idx, fi, nb_slice, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
        end
    end
    
    # 3.3: Threaded loop to calculate k3 (divergence)
    @batch for particle_range in chunks
        for p_idx in particle_range
            if pg.core.is_boundary[p_idx]; continue; end

            fi = rk4.rho_stage[p_idx] # <-- Use u^(2)
            nb_slice = nb_slices[p_idx]
            
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
    @batch for particle_range in chunks
        for p_idx in particle_range
            fi = rk4.rho_stage[p_idx] # <-- Use u^(3) from rho_stage
            nb_slice = nb_slices[p_idx]
            initFs!(rk4, p_idx, fi, nb_slice, rk4.rho_stage, pg)
            initGI!(rk4.gradientInterpolator, p_idx, fi, nb_slice, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
            initGI!(rk4.fallbackInterpolator, p_idx, fi, nb_slice, pg, rk4.neighbor_fs, rk4.neighbor_dfs)
        end
    end
    
    # 4.3: Threaded loop to calculate k4 and Final Solution
    @batch for particle_range in chunks
        for p_idx in particle_range
            if pg.core.is_boundary[p_idx]; continue; end

            fi = rk4.rho_stage[p_idx] # <-- Use u^(3)
            nb_slice = nb_slices[p_idx]
            
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

function RalstonSwitchRK2(grad::G1, fallback::G2, mood::M; tol=1e-7) where {G1, G2, M}
    RalstonSwitchRK2{G1, G2, M}(grad, fallback, mood, tol,
        Float64[], Float64[], Float64[], Float64[], # Main buffers
        Int[], Int[], # Propagation buffers
        falses(0),    # Flag buffer
        Float64[], Float64[] # neighbor_fs, neighbor_dfs
    )
end

# --- User-Friendly Constructor ---
function RalstonSwitchRK2(gradientInterpolator::G1; fallbackInterpolator::G2 = gradientInterpolator, mood::M = NoMOOD(), tol = 1e-7) where {G1, G2, M}
    RalstonSwitchRK2(gradientInterpolator, fallbackInterpolator, mood; tol=tol)
end

function initAddTSBuffer!(ralston::RalstonSwitchRK2, pg::ParticleGrid)
    num_particles = length(pg.neighbor.ranges) 
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
function (ralston::RalstonSwitchRK2)(eq::ScalarHyperbolicPDE, pg::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    N = pg.meta.N
    interior = 1:N # Assuming interior_indices is 1:N for now
    nb_slices = pg.neighbor.ranges
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
    @batch for particle_range in chunks
        for p_idx in particle_range
            fi = ralston.rho_n[p_idx]
            nb_slice = nb_slices[p_idx]
            initFs!(ralston, p_idx, fi, nb_slice, ralston.rho_n, pg)
            # Use fallback interpolator for GI
            initGI!(ralston.fallbackInterpolator, p_idx, fi, nb_slice, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
        end
    end

    # 1.3: Threaded loop to calculate fallback divergence
    @batch for particle_range in chunks
        for p_idx in particle_range
            if pg.core.is_boundary[p_idx]; continue; end

            fi = ralston.rho_n[p_idx]
            nb_slice = nb_slices[p_idx]
            
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
    @batch for particle_range in chunks
        for p_idx in particle_range
            fi = ralston.rho_n[p_idx]
            nb_slice = nb_slices[p_idx]
            initFs!(ralston, p_idx, fi, nb_slice, ralston.rho_n, pg)
            # Use high-order interpolator for GI
            initGI!(ralston.gradientInterpolator, p_idx, fi, nb_slice, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
        end
    end

    # 2.3: Threaded loop to calculate div1 and u^(1)
    @batch for particle_range in chunks
        for p_idx in particle_range
            if pg.core.is_boundary[p_idx]; continue; end

            fi = ralston.rho_n[p_idx]
            nb_slice = nb_slices[p_idx]
            
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
    @batch for particle_range in chunks
        for p_idx in particle_range
            fi = ralston.rho_stage[p_idx]
            nb_slice = nb_slices[p_idx]
            initFs!(ralston, p_idx, fi, nb_slice, ralston.rho_stage, pg)
            initGI!(ralston.gradientInterpolator, p_idx, fi, nb_slice, pg, ralston.neighbor_fs, ralston.neighbor_dfs)
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
        nb_slice = nb_slices[p_idx]

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




