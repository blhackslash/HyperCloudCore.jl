# =========================================================================
# DYNAMIC MOOD DISPATCH (Stateless & Fallback-Free)
# =========================================================================

@inline _check_mood(grad, mood::NoMOOD, eq, p_idx, fi, nb_slice, pg, rk::GeneralRKTimeStepper, stage, dt) = grad(eq, p_idx, fi, nb_slice, pg, rk.int_buffer)

# Active MOOD case: Construct candidate and flag if unstable
@inline function _check_mood(
    grad, mood, eq, p_idx, fi, nb_slice, pg, rk::GeneralRKTimeStepper, stage, dt
)
    div_val = grad(eq, p_idx, fi, nb_slice, pg, rk.int_buffer)
    
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
    
    # If MOOD triggers, flag the particle to "opt out" of the remaining stages
    if mood(grad, p_idx, fi, nb_slice, rho_candidate, pg, rk.int_buffer.f)
        rk.mood_triggered[p_idx] = true 
    end
    
    return div_val
end

# =========================================================================
# GENERAL RUNGE-KUTTA (Explicit Space-Time MOOD)
# =========================================================================

function (rk::GeneralRKTimeStepper{D,M})(
    eq::HyperbolicPDE{D,M,R}, pg::ParticleGrid, time::Real, dt::Real, 
    source_term::AbstractSourceTerm = NoSourceTerm()
) where {D,M,R}
    
    N = pg.meta.N
    M_neighbors = length(pg.neighbor.indices)
    s = length(rk.K_stages)
    A = rk.tableau.A
    b = rk.tableau.b
    c = rk.tableau.c
    
    update_size!(rk, N, M_neighbors)
    
    # ==========================================================
    # 1. UNPACK STRUCT FIELDS TO PREVENT CLOSURE INSTABILITY
    # ==========================================================
    rho_n          = rk.rho_n
    rho_stage      = rk.rho_stage
    K_stages       = rk.K_stages
    mood_triggered = rk.mood_triggered
    int_buffer     = rk.int_buffer
    
    main_grad      = rk.gradientInterpolator
    fallback_grad  = rk.fallbackInterpolator
    mood_fun       = rk.mood
    
    rhos           = pg.rhos
    is_boundary    = pg.core.is_boundary
    
    # Initialize state
    rho_n[1:N] .= view(rhos, 1:N)
    fill!(mood_triggered, false) 
    
    nb_slices  = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices
    
    for stage in 1:s
        # --- 1. Incremental Grid Movement ---
        delta_t = stage == 1 ? c[stage] * dt : (c[stage] - c[stage-1]) * dt
        if delta_t > 0
            pg.mover(pg, delta_t, eq, source_term)
        end

        # --- 2. Calculate U^{(stage)} ---
        if stage == 1
            rho_stage[1:N] .= view(rho_n, 1:N)
        else
            @batch for p_idx in 1:N
                if is_boundary[p_idx]; continue; end
                
                # OPTIMIZATION: If MOOD triggered, lock state at U^n (provides safe donor for BCs)
                if mood_triggered[p_idx]
                    rho_stage[p_idx] = rho_n[p_idx]
                    continue
                end
                
                u_stage = rho_n[p_idx]
                for j in 1:(stage-1)
                    if A[stage, j] != 0.0
                        u_stage -= dt * A[stage, j] * K_stages[j][p_idx]
                    end
                end
                rho_stage[p_idx] = u_stage
            end
            apply_boundary_conditions!(pg, rho_stage)
        end
        
        # --- 3. Pre-Gather (High-Order Only) ---
        update_size!(main_grad, N)
        
        @batch for p_idx in 1:N
            # OPTIMIZATION: Skip completely if MOOD triggered!
            if is_boundary[p_idx] || mood_triggered[p_idx]; continue; end
            
            fi = rho_stage[p_idx]
            nb_slice = nb_slices[p_idx]
            
            update_content!(int_buffer, nb_indices, fi, nb_slice, rho_stage)
            update_content!(main_grad, p_idx, fi, nb_slice, pg, int_buffer)
        end
        
        # --- 4. Divergence & Dynamic MOOD Check ---
        @batch for p_idx in 1:N
            if is_boundary[p_idx] || mood_triggered[p_idx]; continue; end
            
            fi = rho_stage[p_idx]
            nb_slice = nb_slices[p_idx]
            
            K_stages[stage][p_idx] = _check_mood(
                main_grad, mood_fun, eq, p_idx, fi, nb_slice, pg, rk, stage, dt
            )
        end
    end
    
    # --- 5. Final Assembly (The Euler Catcher) ---
    delta_t = (1.0 - c[s]) * dt
    if delta_t > 0
        pg.mover(pg, delta_t, eq, source_term)
        pg.neighbor(pg)
    end

    @batch for p_idx in 1:N
        if is_boundary[p_idx]; continue; end
        
        if mood_triggered[p_idx]
            # PERFECT EULER FALLBACK: Evaluate safe 1st-order step directly from U^n
            fi = rho_n[p_idx]
            nb_slice = nb_slices[p_idx]
            
            update_content!(int_buffer, nb_indices, fi, nb_slice, rho_n)
            update_content!(fallback_grad, p_idx, fi, nb_slice, pg, int_buffer)
            
            div_fallback = fallback_grad(eq, p_idx, fi, nb_slice, pg, int_buffer)
            rhos[p_idx] = rho_n[p_idx] - dt * div_fallback
            
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