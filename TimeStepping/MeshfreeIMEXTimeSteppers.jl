# =========================================================================
# STAGE EVALUATION DISPATCH (IMEX)
# =========================================================================

# 1. Fallback for Generic Gradients and MUSCL{NoMOOD}
@inline function evaluate_stage_derivatives_imex!(
    main_grad::GradientInterpolator, eq_kin, pg, imex_ts, i, dt, current_Y_i
)
    N_particles = pg.meta.N
    nb_slices = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices
    is_boundary = pg.core.is_boundary
    int_buffer = imex_ts.int_buffer
    K_E_stage = imex_ts.K_E_stages[i]

    update_size!(main_grad, N_particles)

    @batch for p_idx in 1:N_particles
        if is_boundary[p_idx]; continue; end
        
        fi = current_Y_i[p_idx]
        nb_slice = nb_slices[p_idx]
        
        update_content!(int_buffer, nb_indices, fi, nb_slice, current_Y_i)
        update_content!(main_grad, p_idx, fi, nb_slice, pg, int_buffer)
    end
    
    @batch for p_idx in 1:N_particles
        if is_boundary[p_idx]
            K_E_stage[p_idx] = zero(eltype(K_E_stage))
            continue
        end
        
        fi = current_Y_i[p_idx]
        nb_slice = nb_slices[p_idx]
        
        K_E_stage[p_idx] = -main_grad(eq_kin, p_idx, fi, nb_slice, pg, int_buffer)
    end
end

# 2. MUSCL Specialization (Iterative MOOD Order Dropping & External Halo Dispatch)
@inline function evaluate_stage_derivatives_imex!(
    main_grad::MUSCL{D, M, B_LEN, MAX_ORDER, DIV_ORDER, MOOD{S, C}, INTERPS, L, NF}, 
    eq_kin, pg, imex_ts, i, dt, current_Y_i
) where {D, M, B_LEN, MAX_ORDER, DIV_ORDER, S <: MOODStrategy, C <: RealMOOD, INTERPS, L, NF}
    
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

    iteration = 0
    while true
        iteration += 1

        # Phase A: Pre-Gather Gradients
        @batch for p_idx in 1:N_particles
            if is_boundary[p_idx] || !needs_recalc[p_idx]; continue; end
            
            fi = current_Y_i[p_idx]
            nb_slice = nb_slices[p_idx]
            
            update_content!(int_buffer, nb_indices, fi, nb_slice, current_Y_i)
            update_content!(main_grad, p_idx, fi, nb_slice, pg, int_buffer)
        end
        
        # Phase B: Reconstruct Interface Fluxes
        @batch for p_idx in 1:N_particles
            if is_boundary[p_idx] || !needs_recalc[p_idx]; continue; end
            
            fi = current_Y_i[p_idx]
            nb_slice = nb_slices[p_idx]
            
            K_E_stage[p_idx] = -main_grad(eq_kin, p_idx, fi, nb_slice, pg, int_buffer)
        end

        # Phase C: MOOD Evaluator & Halo Reduction (Delegated)
        needs_another_pass = evaluate_mood_and_halo!(main_grad, pg, imex_ts, i, dt, current_Y_i)
        
        if !needs_another_pass || iteration >= MAX_ORDER
            break
        end
    end
end

# =========================================================================
# MAIN IMEX FUNCTOR
# =========================================================================

function (imex_ts::GeneralIMEXTimeStepper{D, M})(
    eq_kin::HyperbolicPDE{D, M, R}, pg::ParticleGrid{D, M}, time_n::Real, dt::Real
) where {M, D, R}
    
    s = imex_ts.num_stages
    bt = imex_ts.butcher_tableau
    N_particles = pg.meta.N
    M_neighbors = length(pg.neighbor.indices)

    update_size!(imex_ts, N_particles, M_neighbors)
    
    U_n = imex_ts.U_n
    U_n[1:N_particles] .= view(pg.rhos, 1:N_particles)

    nb_slices  = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices

    for i in 1:s
        delta_t = i != s ? (bt.ct[i+1] - bt.ct[i]) * dt : (1.0 - bt.ct[i]) * dt
        if delta_t > 0
            imex_ts.grid_mover(pg, delta_t)           
        end
        
        current_Y_i = imex_ts.Y_stages[i]

        # PHASE 1: Accumulate Stages
        @batch for p_idx in 1:N_particles
            if pg.core.is_boundary[p_idx]; current_Y_i[p_idx] = pg.rhos[p_idx]; continue; end
            
            Y_local = U_n[p_idx]
            for j in 1:(i-1)
                if bt.At[i,j] != 0.0
                    Y_local += (dt * bt.At[i,j]) * imex_ts.K_E_stages[j][p_idx]
                end
                if bt.A[i,j] != 0.0
                    Y_local += (dt * bt.A[i,j]) * imex_ts.K_I_stages[j][p_idx]
                end
            end
            current_Y_i[p_idx] = Y_local
        end

        # PHASE 2: Non-Local Potential
        if imex_ts.source_term_object isa NonLocalRelaxationSourceTerm
            update_nonlocal_potential!(imex_ts.source_term_object, current_Y_i, pg, imex_ts.eq_macro)  
        end
        
        # PHASE 3: Implicit Solve & K_I Evaluation
        if abs(bt.A[i,i]) > 1e-14
            @batch for p_idx in 1:N_particles
                if pg.core.is_boundary[p_idx]; continue; end
                
                current_Y_i[p_idx] = solve(
                    imex_ts.implicit_solver, current_Y_i[p_idx], dt * bt.A[i,i],
                    imex_ts.source_term_object, p_idx, imex_ts.eq_macro, imex_ts.source_term_object.km
                )
            end
        end
        
        @batch for p_idx in 1:N_particles
            if pg.core.is_boundary[p_idx]
                imex_ts.K_I_stages[i][p_idx] = zero(State{M})
                continue
            end
            
            imex_ts.K_I_stages[i][p_idx] = imex_ts.source_term_object(
                current_Y_i[p_idx], p_idx, imex_ts.eq_macro, imex_ts.source_term_object.km
            )
        end

        # PHASE 4: Evaluate Explicit Spatial Derivatives (Iterative MOOD Loop)
        apply_boundary_conditions!(pg, current_Y_i)
        evaluate_stage_derivatives_imex!(imex_ts.gradientInterpolator, eq_kin, pg, imex_ts, i, dt, current_Y_i)
    end 
    
    # PHASE 5: Final Step Update
    @batch for p_idx in 1:N_particles
        if pg.core.is_boundary[p_idx]; continue; end
        
        rho_final = U_n[p_idx]
        for i in 1:s
            if bt.bt[i] != 0.0
                rho_final += (dt * bt.bt[i]) * imex_ts.K_E_stages[i][p_idx]
            end
            if bt.b[i] != 0.0
                rho_final += (dt * bt.b[i]) * imex_ts.K_I_stages[i][p_idx]
            end
        end
        pg.rhos[p_idx] = rho_final
    end

    apply_boundary_conditions!(pg, pg.rhos)
end