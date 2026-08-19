@inline function evaluate_stage_derivatives_imex!(
    main_grad::DivergenceInterpolator, eq_kin, pg, imex_ts, i, dt, current_Y_i
)
    N_particles = pg.meta.N
    nb_slices = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices
    is_boundary = pg.core.is_boundary
    int_buffer = imex_ts.int_buffer
    K_E_stage = imex_ts.K_E_stages[i]

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
    eq_kin, pg, imex_ts, i, dt, current_Y_i
) where {D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, S <: MOODStrategy, C <: RealMOOD, INTERPS, L, NF}
    
    N_particles = pg.meta.N
    nb_slices = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices
    is_boundary = pg.core.is_boundary
    int_buffer = imex_ts.int_buffer
    K_E_stage = imex_ts.K_E_stages[i]
    
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

        needs_another_pass = evaluate_mood_and_halo!(main_grad, pg, imex_ts, i, dt, current_Y_i)
        
        if !needs_another_pass || iteration >= MAX_ORDER
            break
        end
    end
end

function (imex_ts::GeneralIMEXTimeStepper{D, M, T})(
    eq_kin::HyperbolicPDE{D, M, T, R}, pg::ParticleGrid{D, M, T}, time::Real, dt::Real
) where {M, D, T, R}
    
    s = imex_ts.num_stages
    bt = imex_ts.tableau
    N_particles = pg.meta.N
    M_neighbors = length(pg.neighbor.indices)

    update_size!(imex_ts, N_particles, M_neighbors)
    
    U_n = imex_ts.rho_n
    U_n[1:N_particles] .= view(pg.rhos, 1:N_particles)

    for i in 1:s
        delta_t = i != s ? (bt.c_t[i+1] - bt.c_t[i]) * dt : (one(T) - bt.c_t[i]) * dt
        if delta_t > zero(T)
            pg.mover(pg, delta_t)           
        end
        
        current_Y_i = imex_ts.Y_stages[i]

        @batch for p_idx in 1:N_particles
            if pg.core.is_boundary[p_idx]; current_Y_i[p_idx] = pg.rhos[p_idx]; continue; end
            
            Y_local = U_n[p_idx]
            for j in 1:(i-1)
                if bt.a_t[i,j] != zero(T)
                    Y_local += (dt * bt.a_t[i,j]) * imex_ts.K_E_stages[j][p_idx]
                end
                if bt.a[i,j] != zero(T)
                    Y_local += (dt * bt.a[i,j]) * imex_ts.K_I_stages[j][p_idx]
                end
            end
            current_Y_i[p_idx] = Y_local
        end

        if imex_ts.source_term_object isa NonLocalRelaxationSourceTerm
            update_nonlocal_potential!(imex_ts.source_term_object, current_Y_i, pg, imex_ts.pde)  
        end
        
        if abs(bt.a[i,i]) > T(1e-14)
            @batch for p_idx in 1:N_particles
                if pg.core.is_boundary[p_idx]; continue; end
                
                current_Y_i[p_idx] = solve(
                    imex_ts.implicit_solver, current_Y_i[p_idx], dt * bt.a[i,i],
                    imex_ts.source_term_object, p_idx, imex_ts.pde, imex_ts.source_term_object.km
                )
            end
        end
        
        @batch for p_idx in 1:N_particles
            if pg.core.is_boundary[p_idx]
                imex_ts.K_I_stages[i][p_idx] = zero(State{M, T})
                continue
            end
            
            imex_ts.K_I_stages[i][p_idx] = imex_ts.source_term_object(
                current_Y_i[p_idx], p_idx, imex_ts.pde, imex_ts.source_term_object.km
            )
        end
        stage_time = time + bt.c_t[stage] * dt
        apply_boundary_conditions!(pg, current_Y_i, imex_ts.pde, stage_time)
        evaluate_stage_derivatives_imex!(imex_ts.divergence_interpolator, eq_kin, pg, imex_ts, i, dt, current_Y_i)
    end 
    
    @batch for p_idx in 1:N_particles
        if pg.core.is_boundary[p_idx]; continue; end
        
        rho_final = U_n[p_idx]
        for i in 1:s
            if bt.b_t[i] != zero(T)
                rho_final += (dt * bt.b_t[i]) * imex_ts.K_E_stages[i][p_idx]
            end
            if bt.b[i] != zero(T)
                rho_final += (dt * bt.b[i]) * imex_ts.K_I_stages[i][p_idx]
            end
        end
        pg.rhos[p_idx] = rho_final
    end

    apply_boundary_conditions!(pg, pg.rhos, imex_ts.pde, time + dt)
end