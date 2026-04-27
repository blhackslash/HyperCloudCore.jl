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
    fill!(imex_ts.mood_triggered, false)

    nb_slices  = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices

    for i in 1:s
        delta_t = i != s ? (bt.ct[i+1] - bt.ct[i]) * dt : (1.0 - bt.ct[i]) * dt
        if delta_t > 0
            imex_ts.grid_mover(pg, delta_t)           
        end
        
        current_Y_i = imex_ts.Y_stages[i]

        # ==================================================================
        # PHASE 1: Accumulate Stages
        # ==================================================================
        @batch for p_idx in 1:N_particles
            if pg.core.is_boundary[p_idx]; current_Y_i[p_idx] = pg.rhos[p_idx]; continue; end
            
            # Lock state at U^n if MOOD triggered
            if imex_ts.mood_triggered[p_idx]
                current_Y_i[p_idx] = U_n[p_idx]
                continue
            end

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

        # ==================================================================
        # PHASE 2: Non-Local Potential
        # ==================================================================
        if imex_ts.source_term_object isa NonLocalRelaxationSourceTerm
            update_nonlocal_potential!(imex_ts.source_term_object, current_Y_i, pg, imex_ts.eq_macro)  
        end
        
        # ==================================================================
        # PHASE 3: Implicit Solve & K_I Evaluation
        # ==================================================================
        if abs(bt.A[i,i]) > 1e-14
            @batch for p_idx in 1:N_particles
                if pg.core.is_boundary[p_idx] || imex_ts.mood_triggered[p_idx]; continue; end
                
                current_Y_i[p_idx] = solve(
                    imex_ts.implicit_solver, current_Y_i[p_idx], dt * bt.A[i,i],
                    imex_ts.source_term_object, p_idx, imex_ts.eq_macro, imex_ts.source_term_object.km
                )
            end
        end
        
        @batch for p_idx in 1:N_particles
            if pg.core.is_boundary[p_idx] || imex_ts.mood_triggered[p_idx]; continue; end
            
            imex_ts.K_I_stages[i][p_idx] = imex_ts.source_term_object(
                current_Y_i[p_idx], p_idx, imex_ts.eq_macro, imex_ts.source_term_object.km
            )
        end

        # ==================================================================
        # PHASE 4a: Explicit Gradients Pre-Gather
        # ==================================================================
        apply_boundary_conditions!(pg, current_Y_i)
        
        grad = imex_ts.gradientInterpolator
        has_fallback = !(imex_ts.fallbackInterpolator isa NoFallbackGrad)
        update_size!(grad, N_particles)
        
        @batch for p_idx in 1:N_particles
            if pg.core.is_boundary[p_idx] || imex_ts.mood_triggered[p_idx]; continue; end
            
            fi = current_Y_i[p_idx]
            nb_slice = nb_slices[p_idx]

            update_content!(imex_ts.int_buffer, nb_indices, fi, nb_slice, current_Y_i)
            update_content!(grad, p_idx, fi, nb_slice, pg, imex_ts.int_buffer)
        end 

        # ==================================================================
        # PHASE 4b: Explicit Flux Evaluation (K_E)
        # ==================================================================
        @batch for p_idx in 1:N_particles
            if pg.core.is_boundary[p_idx] || imex_ts.mood_triggered[p_idx]; continue; end
            
            fi = current_Y_i[p_idx]
            nb_slice = nb_slices[p_idx]

            div_high = grad(eq_kin, p_idx, fi, nb_slice, pg, imex_ts.int_buffer) 
            rho_candidate = fi - dt * div_high
            
            if has_fallback && imex_ts.mood(grad, p_idx, fi, nb_slice, rho_candidate, pg, imex_ts.int_buffer.f)
                imex_ts.mood_triggered[p_idx] = true # Triggers optical dropout!
            else
                imex_ts.K_E_stages[i][p_idx] = -div_high
            end
        end 
    end # End of stages loop
    
    # ==================================================================
    # PHASE 5: Final Step Update (With True IMEX Euler MOOD Fallback)
    # ==================================================================
    @batch for p_idx in 1:N_particles
        if pg.core.is_boundary[p_idx]; continue; end
        
        if imex_ts.mood_triggered[p_idx]
            # PERFECT IMEX EULER FALLBACK
            fi = U_n[p_idx]
            nb_slice = nb_slices[p_idx]
            
            update_content!(imex_ts.int_buffer, nb_indices, fi, nb_slice, U_n)
            update_content!(imex_ts.fallbackInterpolator, p_idx, fi, nb_slice, pg, imex_ts.int_buffer)
            
            div_fallback = imex_ts.fallbackInterpolator(eq_kin, p_idx, fi, nb_slice, pg, imex_ts.int_buffer)
            Y_pred = U_n[p_idx] - dt * div_fallback
            
            pg.rhos[p_idx] = solve(
                imex_ts.implicit_solver, Y_pred, dt,
                imex_ts.source_term_object, p_idx, imex_ts.eq_macro, imex_ts.source_term_object.km
            )
            
        else
            # High-Order IMEX Update
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
    end

    apply_boundary_conditions!(pg, pg.rhos)
end