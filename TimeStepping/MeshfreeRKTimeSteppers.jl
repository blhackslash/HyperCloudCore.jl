# =========================================================================
# STAGE EVALUATION DISPATCH
# =========================================================================

# 1. Fallback for Generic Gradients and MUSCL{NoMOOD}
@inline function evaluate_stage_derivatives!(
    main_grad::GradientInterpolator, eq, pg, rk, stage, dt, rho_stage
)
    N = pg.meta.N
    nb_slices = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices
    is_boundary = pg.core.is_boundary
    int_buffer = rk.int_buffer
    K_stage = rk.K_stages[stage]

    update_size!(main_grad, N)

    use_threads = _use_threads()

    @smart_parallel use_threads for p_idx in 1:N
        if is_boundary[p_idx]; continue; end
        
        fi = rho_stage[p_idx]
        nb_slice = nb_slices[p_idx]
        
        update_content!(int_buffer, nb_indices, fi, nb_slice, rho_stage)
        update_content!(main_grad, p_idx, fi, nb_slice, pg, int_buffer)
    end
    
    @smart_parallel use_threads for p_idx in 1:N
        if is_boundary[p_idx]; continue; end
        
        fi = rho_stage[p_idx]
        nb_slice = nb_slices[p_idx]
        
        K_stage[p_idx] = main_grad(eq, p_idx, fi, nb_slice, pg, int_buffer)
    end
end

# 2. MUSCL Specialization (Iterative MOOD Order Dropping & Halo Effect)
# 2. MUSCL Specialization (Iterative MOOD Order Dropping)
@inline function evaluate_stage_derivatives!(
    main_grad::MUSCL{D, M, B_LEN, MAX_ORDER, DIV_ORDER, MOOD{S, C}}, 
    eq, pg, rk, stage, dt, rho_stage
) where {D, M, B_LEN, MAX_ORDER, DIV_ORDER, S <: MOODStrategy, C <: RealMOOD}
    
    mood_fun = main_grad.mood
    N = pg.meta.N
    nb_slices = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices
    is_boundary = pg.core.is_boundary
    int_buffer = rk.int_buffer
    K_stage = rk.K_stages[stage]
    
    orders = main_grad.particle_orders
    needs_recalc = pg.shared.bit_buffer

    update_size!(main_grad, N)

    if stage == 1
        fill!(orders, MAX_ORDER)
    end

    fill!(needs_recalc, true)

    use_threads = _use_threads()
    iteration = 0
    while true
        iteration += 1

        # Phase A: Pre-Gather Gradients
        @smart_parallel use_threads for p_idx in 1:N
            if is_boundary[p_idx] || !needs_recalc[p_idx]; continue; end
            
            fi = rho_stage[p_idx]
            nb_slice = nb_slices[p_idx]
            
            update_content!(int_buffer, nb_indices, fi, nb_slice, rho_stage)
            update_content!(main_grad, p_idx, fi, nb_slice, pg, int_buffer)
        end
        # Phase A.5: Precompute Effective Orders for ALL particles (O(N))
        effective_orders = pg.shared.int_buffer
        @smart_parallel use_threads for p_idx in 1:N
            if is_boundary[p_idx]; continue; end
            nb_slice = nb_slices[p_idx]
            effective_orders[p_idx] = get_effective_order(main_grad.mood.strategy, orders, p_idx, nb_slice, nb_indices)
        end

        # Phase B: Reconstruct Interface Fluxes
        @smart_parallel use_threads for p_idx in 1:N
            if is_boundary[p_idx] || !needs_recalc[p_idx]; continue; end
            
            fi = rho_stage[p_idx]
            nb_slice = nb_slices[p_idx]
            
            K_stage[p_idx] = main_grad(eq, p_idx, fi, nb_slice, pg, int_buffer)
        end

        # Phase C: MOOD Evaluator & Halo Reduction (External Dispatch)
        needs_another_pass = evaluate_mood_and_halo!(main_grad, pg, rk, stage, dt, rho_stage)
        
        if !needs_another_pass || iteration >= 20
            break
        end
    end
end

# =========================================================================
# GENERAL RUNGE-KUTTA (Explicit Space-Time)
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
    
    rho_n       = rk.rho_n
    rho_stage   = rk.rho_stage
    K_stages    = rk.K_stages
    main_grad   = rk.gradientInterpolator
    
    rhos        = pg.rhos
    is_boundary = pg.core.is_boundary
    
    rho_n[1:N] .= view(rhos, 1:N)
    
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
        
        # --- 3. Evaluate Spatial Derivatives (Iterative MOOD Loop) ---
        evaluate_stage_derivatives!(main_grad, eq, pg, rk, stage, dt, rho_stage)
    end
    
    # --- 4. Final Assembly ---
    delta_t = (1.0 - c[s]) * dt
    if delta_t > 0
        pg.mover(pg, delta_t, eq, source_term)
        pg.neighbor(pg)
    end

    @batch for p_idx in 1:N
        if is_boundary[p_idx]; continue; end
        
        rho_final = rho_n[p_idx]
        for j in 1:s
            if b[j] != 0.0
                rho_final -= dt * b[j] * K_stages[j][p_idx]
            end
        end
        rhos[p_idx] = rho_final
    end
    
    apply_boundary_conditions!(pg, rhos)
end