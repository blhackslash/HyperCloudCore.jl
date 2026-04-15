# =========================================================================
# BUFFER INITIALIZATION
# =========================================================================

function initTSBuffer!(rk::GeneralRKTimeStepper{M}, pg::ParticleGrid) where {M}
    N = pg.meta.N
    M_neighbors = length(pg.neighbor.indices)
    s = length(rk.K_stages)

    if length(rk.rho_n) < N
        new_cap = ceil(Int, N * 1.25)
        resize!(rk.rho_n, new_cap)
        resize!(rk.rho_stage, new_cap)
        resize!(rk.mood_triggered, new_cap)
        for i in 1:s
            resize!(rk.K_stages[i], new_cap)
        end
    end
    if length(rk.neighbor_fs) < M_neighbors
        new_cap = ceil(Int, M_neighbors * 1.25)
        resize!(rk.neighbor_fs, new_cap)
        resize!(rk.neighbor_dfs, new_cap)
    end
end

# =========================================================================
# DYNAMIC DIVERGENCE & MOOD DISPATCH
# =========================================================================

# Base cases: NO MOOD or NO FALLBACK -> Skip candidate construction entirely!
@inline _get_divergence(grad, fallback::NoFallbackGrad, mood, eq, p_idx, fi, nb_slice, pg, n_fs, n_dfs, rk::GeneralRKTimeStepper, stage, dt) = grad(eq, p_idx, fi, nb_slice, pg, n_fs, n_dfs)
@inline _get_divergence(grad, fallback, mood::NoMOOD, eq, p_idx, fi, nb_slice, pg, n_fs, n_dfs, rk::GeneralRKTimeStepper, stage, dt) = grad(eq, p_idx, fi, nb_slice, pg, n_fs, n_dfs)
@inline _get_divergence(grad, fallback::NoFallbackGrad, mood::NoMOOD, eq, p_idx, fi, nb_slice, pg, n_fs, n_dfs, rk::GeneralRKTimeStepper, stage, dt) = grad(eq, p_idx, fi, nb_slice, pg, n_fs, n_dfs)

# Active MOOD case: Construct candidate and flag if unstable
@inline function _get_divergence(
    grad, fallback, mood, eq, p_idx, fi, nb_slice, pg, n_fs, n_dfs, rk::GeneralRKTimeStepper, stage, dt
)
    # FAST PATH: If already dropped to Euler, just return 1st-order fallback flux instantly
    if rk.mood_triggered[p_idx]
        return fallback(eq, p_idx, fi, nb_slice, pg, n_fs, n_dfs)
    end
    
    div_val = grad(eq, p_idx, fi, nb_slice, pg, n_fs, n_dfs)
    
    s = length(rk.K_stages)
    base_rho = rk.rho_n[p_idx]
    A_coef = 0.0
    
    # Predict the state at the NEXT step using strict RK weights
    if stage < s
        for j in 1:(stage-1)
            a_val = rk.tableau.A[stage+1, j]
            if a_val != 0.0; base_rho -= dt * a_val * rk.K_stages[j][p_idx]; end
        end
        A_coef = rk.tableau.A[stage+1, stage]
    else
        for j in 1:(s-1)
            b_val = rk.tableau.b[j]
            if b_val != 0.0; base_rho -= dt * b_val * rk.K_stages[j][p_idx]; end
        end
        A_coef = rk.tableau.b[stage]
    end
    
    rho_candidate = base_rho - dt * A_coef * div_val
    
    # If MOOD triggers, we flag this particle for the rest of the timestep
    if mood(grad, p_idx, fi, nb_slice, rho_candidate, pg, n_fs)
        div_val = fallback(eq, p_idx, fi, nb_slice, pg, n_fs, n_dfs)
        rk.mood_triggered[p_idx] = true 
    end
    
    return div_val
end

# =========================================================================
# GENERAL RUNGE-KUTTA (Explicit Space-Time MOOD)
# =========================================================================

function (rk::GeneralRKTimeStepper{M})(
    eq::HyperbolicPDE, pg::ParticleGrid, time::Real, dt::Real, 
    source_term::AbstractSourceTerm = NoSourceTerm()
) where {M}
    N = pg.meta.N
    s = length(rk.K_stages)
    A = rk.tableau.A
    b = rk.tableau.b
    c = rk.tableau.c
    
    initTSBuffer!(rk, pg)
    
    # ==========================================================
    # 1. UNPACK STRUCT FIELDS TO PREVENT CLOSURE INSTABILITY
    # ==========================================================
    rho_n          = rk.rho_n
    rho_stage      = rk.rho_stage
    K_stages       = rk.K_stages
    mood_triggered = rk.mood_triggered
    neighbor_fs    = rk.neighbor_fs
    neighbor_dfs   = rk.neighbor_dfs
    
    main_grad      = rk.gradientInterpolator
    fallback_grad  = rk.fallbackInterpolator
    mood_fun       = rk.mood
    
    rhos           = pg.rhos
    is_boundary    = pg.core.is_boundary
    
    # Initialize state
    rho_n[1:N] .= view(rhos, 1:N)
    fill!(mood_triggered, false) # Reset MOOD flags at start of full dt
    
    # Extract neighbor tracking arrays
    nb_slices  = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices
    
    for stage in 1:s
        # --- 1. Incremental Grid Movement ---
        delta_t = stage == 1 ? c[stage] * dt : (c[stage] - c[stage-1]) * dt
        if delta_t > 0
            pg.mover(pg, delta_t, eq, source_term)
            
            pg.neighbor(pg)
            
            # CRITICAL: Refresh neighbor arrays as pg.neighbor(pg) might have resized them!
            #nb_slices  = pg.neighbor.ranges
            #nb_indices = pg.neighbor.indices
        end

        # --- 2. Calculate U^{(stage)} ---
        if stage == 1
            rho_stage[1:N] .= view(rho_n, 1:N)
        else
            @batch for p_idx in 1:N
                if is_boundary[p_idx]; continue; end
                
                if mood_triggered[p_idx]
                    # Safe Euler Trajectory 
                    rho_stage[p_idx] = rho_n[p_idx] - c[stage] * dt * K_stages[1][p_idx]
                else
                    # High-Order RK Trajectory
                    u_stage = rho_n[p_idx]
                    for j in 1:(stage-1)
                        if A[stage, j] != 0.0
                            u_stage -= dt * A[stage, j] * K_stages[j][p_idx]
                        end
                    end
                    rho_stage[p_idx] = u_stage
                end
            end
            apply_boundary_conditions!(pg, rho_stage)
        end
        
        # --- 3. Pre-Gather (Optimized) ---
        initGIBuffers!(main_grad, pg)
        if !(fallback_grad isa NoFallbackGrad) && !(mood_fun isa NoMOOD)
            initGIBuffers!(fallback_grad, pg)
        end
        
        @batch for p_idx in 1:N
            fi = rho_stage[p_idx]
            nb_slice = nb_slices[p_idx]
            
            # Pass unpacked arrays cleanly
            initFs!(neighbor_fs, neighbor_dfs, nb_indices, fi, nb_slice, rho_stage)
            
            # Massive Speedup: If already dropped to Euler, skip allocating high-order matrices!
            if mood_triggered[p_idx]
                initGI!(fallback_grad, p_idx, fi, nb_slice, pg, neighbor_fs, neighbor_dfs)
            else
                initGI!(main_grad, p_idx, fi, nb_slice, pg, neighbor_fs, neighbor_dfs)
                if !(fallback_grad isa NoFallbackGrad) && !(mood_fun isa NoMOOD)
                    initGI!(fallback_grad, p_idx, fi, nb_slice, pg, neighbor_fs, neighbor_dfs)
                end
            end
        end
        
        # --- 4. Divergence & Dynamic MOOD Check ---
        @batch for p_idx in 1:N
            if is_boundary[p_idx]; continue; end
            
            fi = rho_stage[p_idx]
            nb_slice = nb_slices[p_idx]
            
            K_stages[stage][p_idx] = _get_divergence(
                main_grad, fallback_grad, mood_fun,
                eq, p_idx, fi, nb_slice, pg, neighbor_fs, neighbor_dfs, 
                rk, stage, dt
            )
        end
    end
    
    # --- 5. Final Assembly ---
    delta_t = (1.0 - c[s]) * dt
    if delta_t > 0
        pg.mover(pg, delta_t, eq, source_term)
        # pg.reorder(pg) # Recommended!
        pg.neighbor(pg)
    end

    @batch for p_idx in 1:N
        if is_boundary[p_idx]; continue; end
        
        if mood_triggered[p_idx]
            # Safe Full Euler Step
            rhos[p_idx] = rho_n[p_idx] - dt * K_stages[1][p_idx]
        else
            # Full RK Step
            rho_final = rho_n[p_idx]
            for j in 1:s
                if b[j] != 0.0
                    rho_final -= dt * b[j] * K_stages[j][p_idx]
                end
            end
            rhos[p_idx] = rho_final
        end
    end
    
    apply_boundary_conditions!(pg, rhos)
end