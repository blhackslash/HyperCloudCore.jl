"""
    evaluate_stage_derivatives_imex!(main_grad, eq_kin, pg, imex, i, dt, current_Y_i)

Evaluates the explicit kinematic spatial derivatives for a specified IMEX sub-stage. 

# Details
- Calculates the explicit negative divergence using the provided interpolator, storing the result in the corresponding explicit RK buffer `K_E_stage`. 
- **MUSCL Dispatch:** Similar to the standard explicit RK solver, if a `MUSCL` divergence evaluator is attached, this engages an iterative MOOD loop. It recursively evaluates effective orders and drops polynomial degrees where interface monotonicity fails, iterating up to a maximum of `MAX_ORDER` times.
"""
@inline function evaluate_stage_derivatives_imex!(
    main_grad::DivergenceInterpolator, eq_kin, pg, imex, i, dt, current_Y_i
)
    N_particles = pg.meta.N
    nb_slices = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices
    is_boundary = pg.core.is_boundary
    int_buffer = imex.int_buffer
    K_E_stage = imex.K_E_stages[i]

    update_size!(main_grad, N_particles)
    use_threads = _use_threads()
    
    @smart_parallel use_threads for p_idx in 1:N_particles
        if is_boundary[p_idx]; continue; end
        
        fi = current_Y_i[p_idx]
        nb_slice = nb_slices[p_idx]
        
        update_content!(int_buffer, nb_indices, fi, nb_slice, current_Y_i)
        update_content!(main_grad, p_idx, fi, nb_slice, pg, int_buffer)
    end
    
    @smart_parallel use_threads for p_idx in 1:N_particles
        if is_boundary[p_idx]
            K_E_stage[p_idx] = zero(eltype(K_E_stage))
            continue
        end
        
        fi = current_Y_i[p_idx]
        nb_slice = nb_slices[p_idx]
        
        K_E_stage[p_idx] = -main_grad(eq_kin, p_idx, fi, nb_slice, pg, int_buffer)
    end
end

@inline function evaluate_stage_derivatives_imex!(
    main_grad::MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, MOOD{S, C}, INTERPS, L, NF}, 
    eq_kin, pg, imex, i, dt, current_Y_i
) where {D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, S <: MOODStrategy, C <: RealMOOD, INTERPS, L, NF}
    
    N_particles = pg.meta.N
    nb_slices = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices
    is_boundary = pg.core.is_boundary
    int_buffer = imex.int_buffer
    K_E_stage = imex.K_E_stages[i]
    
    orders = main_grad.particle_orders
    needs_recalc = pg.shared.bit_buffer

    update_size!(main_grad, N_particles)

    if i == 1
        fill!(orders, MAX_ORDER)
    end

    fill!(needs_recalc, true)
    use_threads = _use_threads()
    iteration = 0
    
    while true
        iteration += 1

        @smart_parallel use_threads for p_idx in 1:N_particles
            if is_boundary[p_idx] || !needs_recalc[p_idx]; continue; end
            
            fi = current_Y_i[p_idx]
            nb_slice = nb_slices[p_idx]
            
            update_content!(int_buffer, nb_indices, fi, nb_slice, current_Y_i)
            update_content!(main_grad, p_idx, fi, nb_slice, pg, int_buffer)
        end

        effective_orders = pg.shared.int_buffer
        @smart_parallel use_threads for p_idx in 1:N_particles
            if is_boundary[p_idx]; continue; end
            nb_slice = nb_slices[p_idx]
            effective_orders[p_idx] = get_effective_order(main_grad.mood.strategy, orders, p_idx, nb_slice, nb_indices)
        end
        
        @smart_parallel use_threads for p_idx in 1:N_particles
            if is_boundary[p_idx] || !needs_recalc[p_idx]; continue; end
            
            fi = current_Y_i[p_idx]
            nb_slice = nb_slices[p_idx]
            
            K_E_stage[p_idx] = -main_grad(eq_kin, p_idx, fi, nb_slice, pg, int_buffer)
        end

        needs_another_pass = evaluate_mood_and_halo!(main_grad, pg, imex, i, dt, current_Y_i)
        
        if !needs_another_pass || iteration >= MAX_ORDER
            break
        end
    end
end

"""
    (imex::GeneralIMEXTimeStepper)(eq_kin::HyperbolicPDE, pg::ParticleGrid, time::Real, dt::Real)

Executes a comprehensive IMEX Runge-Kutta time step, integrating explicit advective spatial operators with implicit stiff source operators.

# Details
- Translates coordinates using `pg.mover` according to the explicit stage weights `c_t`.
- Assembles the intermediate stage candidates (`Y_local`) using both prior explicit flux evaluations (`K_E_stages`) and prior implicit evaluations (`K_I_stages`).
- Detects non-local relaxation source terms, dynamically updating the non-local potentials across the grid prior to the implicit solve.
- Evaluates the implicit solver directly on the candidate state to resolve stiff interactions defined by `source_term_object` scaled by the diagonal implicit Butcher weight `dt * bt.a[i,i]`.
- Finalizes particle states by combining both the explicit evaluations mapped via `b_t` weights and implicit evaluations mapped via `b` weights, followed by boundary condition application.
"""
function (imex::GeneralIMEXTimeStepper{D, M, T})(
    eq::HyperbolicPDE{D, M, T, R}, pg::ParticleGrid{D, M, T}, time::Real, dt::Real
) where {M, D, T, R}
    
    s = imex.num_stages
    bt = imex.tableau
    N_particles = pg.meta.N
    M_neighbors = length(pg.neighbor.indices)

    update_size!(imex, N_particles, M_neighbors)
    
    U_n = imex.rho_n
    U_n[1:N_particles] .= view(pg.rhos, 1:N_particles)

    for i in 1:s
        current_Y_i = imex.Y_stages[i]
        stage_time = time + bt.c_t[i] * dt

        @batch for p_idx in 1:N_particles
            if pg.core.is_boundary[p_idx]; current_Y_i[p_idx] = pg.rhos[p_idx]; continue; end
            
            Y_local = U_n[p_idx]
            for j in 1:(i-1)
                if bt.a_t[i,j] != zero(T)
                    Y_local += (dt * bt.a_t[i,j]) * imex.K_E_stages[j][p_idx]
                end
                if bt.a[i,j] != zero(T)
                    Y_local += (dt * bt.a[i,j]) * imex.K_I_stages[j][p_idx]
                end
            end
            current_Y_i[p_idx] = Y_local
        end

        # --- THE NEW PURE API PIPELINE ---
        
        # 1. API Hook: Let the source term update any global/non-local states
        pre_solve_update!(imex.source_term_object, current_Y_i, pg, stage_time)
        
        # 2. API Hook: Implicit Solve
        if abs(bt.a[i,i]) > T(1e-14)
            @batch for p_idx in 1:N_particles
                if pg.core.is_boundary[p_idx]; continue; end
                
                current_Y_i[p_idx] = implicit_solve(
                    imex.implicit_solver, current_Y_i[p_idx], dt * bt.a[i,i],
                    imex.source_term_object, p_idx, pg, stage_time
                )
            end
        end
        
        # 3. API Hook: Evaluate Source Term
        @batch for p_idx in 1:N_particles
            if pg.core.is_boundary[p_idx]
                imex.K_I_stages[i][p_idx] = zero(State{M, T})
                continue
            end
            
            imex.K_I_stages[i][p_idx] = evaluate_source(
                imex.source_term_object, current_Y_i[p_idx], p_idx, pg, stage_time
            )
        end
        
        # ---------------------------------
        
        apply_boundary_conditions!(pg, current_Y_i, imex, eq, stage_time)
        evaluate_stage_derivatives_imex!(imex.divergence_interpolator, eq, pg, imex, i, dt, current_Y_i)
    end
    
    @batch for p_idx in 1:N_particles
        if pg.core.is_boundary[p_idx]; continue; end
        
        rho_final = U_n[p_idx]
        for i in 1:s
            if bt.b_t[i] != zero(T)
                rho_final += (dt * bt.b_t[i]) * imex.K_E_stages[i][p_idx]
            end
            if bt.b[i] != zero(T)
                rho_final += (dt * bt.b[i]) * imex.K_I_stages[i][p_idx]
            end
        end
        pg.rhos[p_idx] = rho_final
    end

    apply_boundary_conditions!(pg, pg.rhos, imex, imex.pde, time + dt)
end