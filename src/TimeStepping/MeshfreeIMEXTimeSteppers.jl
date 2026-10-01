# Auto-Sorting Convenience Constructor for IMEX
function GeneralIMEXTimeStepper(
    pde::HyperbolicPDE{D, M, T}, div_interp::G, 
    all_sources::Tuple{Vararg{AbstractSourceTerm}}, 
    tableau::IMEXButcherTableau{T}
) where {D, M, T, G}
    explicit_sts = filter(st -> st isa AbstractExplicitSourceTerm, all_sources)
    implicit_sts = filter(st -> st isa AbstractImplicitSourceTerm, all_sources)
    
    return GeneralIMEXTimeStepper(pde, div_interp, explicit_sts, implicit_sts, tableau)
end

# Fallback for no source terms (Empty Tuples)
function GeneralIMEXTimeStepper(
    pde::HyperbolicPDE{D, M, T}, div_interp::G, tableau::IMEXButcherTableau{T}
) where {D, M, T, G}
    return GeneralIMEXTimeStepper(pde, div_interp, (), (), tableau)
end

function update_size!(ts::GeneralIMEXTimeStepper, N_particles::Int, M_neighbors::Int)
    ensure_capacity!(ts.rho_n, N_particles)
    
    for i in 1:ts.num_stages
        ensure_capacity!(ts.Y_stages[i], N_particles)
        ensure_capacity!(ts.K_E_stages[i], N_particles)
        ensure_capacity!(ts.K_I_stages[i], N_particles)
    end
    
    update_size!(ts.int_buffer, M_neighbors)
    return nothing
end


@inline function evaluate_stage_derivatives_imex!(
    main_grad::DivergenceInterpolator, eq_kin, pg, imex, i, dt, current_Y_i, stage_time
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
        
        div_F = main_grad(eq_kin, p_idx, fi, nb_slice, pg, int_buffer)
        S_expl = evaluate_sources(imex.explicit_sources, fi, p_idx, pg, stage_time)
        
        # IMEX accumulation uses addition, so K_E = -div_F + S_expl
        K_E_stage[p_idx] = -div_F + S_expl
    end
end

@inline function evaluate_stage_derivatives_imex!(
    main_grad::MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, MOOD{S, C}, INTERPS, L, NF}, 
    eq_kin, pg, imex, i, dt, current_Y_i, stage_time
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
    if i == 1; fill!(orders, MAX_ORDER); end
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
            
            div_F = main_grad(eq_kin, p_idx, fi, nb_slice, pg, int_buffer)
            S_expl = evaluate_sources(imex.explicit_sources, fi, p_idx, pg, stage_time)
            
            K_E_stage[p_idx] = -div_F + S_expl
        end

        needs_another_pass = evaluate_mood_and_halo!(main_grad, pg, imex, i, dt, current_Y_i)
        if !needs_another_pass || iteration >= MAX_ORDER; break; end
    end
end

function (imex::GeneralIMEXTimeStepper{D, M, T})(
    eq::HyperbolicPDE, pg::ParticleGrid, time::Real, dt::Real
) where {M, D, T}
    
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
        
        pre_solve_updates!(imex.implicit_sources, current_Y_i, pg, stage_time)
        
        if abs(bt.a[i,i]) > T(1e-14)
            @batch for p_idx in 1:N_particles
                if pg.core.is_boundary[p_idx]; continue; end
                
                current_Y_i[p_idx] = implicit_solve(
                    imex.implicit_sources, current_Y_i[p_idx], dt * bt.a[i,i], p_idx, pg, stage_time
                )
            end
        end
        
        @batch for p_idx in 1:N_particles
            if pg.core.is_boundary[p_idx]
                imex.K_I_stages[i][p_idx] = zero(State{M, T})
                continue
            end
            
            imex.K_I_stages[i][p_idx] = evaluate_sources(
                imex.implicit_sources, current_Y_i[p_idx], p_idx, pg, stage_time
            )
        end
        
        apply_boundary_conditions!(pg, current_Y_i, imex, eq, stage_time)
        
        # Pass stage_time cleanly down into the evaluator
        evaluate_stage_derivatives_imex!(imex.divergence_interpolator, eq, pg, imex, i, dt, current_Y_i, stage_time)
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

    apply_boundary_conditions!(pg, pg.rhos, imex, eq, time + dt)
end