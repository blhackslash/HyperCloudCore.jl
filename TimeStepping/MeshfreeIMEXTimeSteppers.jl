using StaticArrays
using Polyester

# =========================================================================
# IMEX TIME STEPPER STRUCT
# ===================================================================

function GeneralIMEXTimeStepper(
    gradientInterpolator::G1, fallbackInterpolator::G2, mood::MOOD,
    implicit_solver::IS, source_term_object::ST_OBJ, 
    grid_mover::GM, eq_macro::EQ_MACRO, butcher_tableau::BT, 
) where {G1, G2, MOOD, IS, ST_OBJ, BT, GM, EQ_MACRO}
    
    s = size(butcher_tableau.A, 1)
    M = length(source_term_object.scaled_inv_speeds) # NK
    
    return GeneralIMEXTimeStepper{M, G1, G2, MOOD, IS, ST_OBJ, BT, GM, EQ_MACRO}(
        gradientInterpolator, fallbackInterpolator, mood, 
        implicit_solver, source_term_object, butcher_tableau, grid_mover, eq_macro,
        State{M}[], 
        [State{M}[] for _ in 1:s], 
        [State{M}[] for _ in 1:s], 
        [State{M}[] for _ in 1:s], 
        Bool[], State{M}[], State{M}[], s
    )
end

function initAddTSBuffer!(ts::GeneralIMEXTimeStepper{M}, pg::ParticleGrid) where {M}
    N = pg.meta.N
    M_neighbors = length(pg.neighbor.indices)

    if length(ts.U_n) < N
        new_cap = ceil(Int, N * 1.25)
        resize!(ts.U_n, new_cap)
        resize!(ts.mood_triggered, new_cap)
        for i in 1:ts.num_stages
            resize!(ts.Y_stages[i], new_cap)
            resize!(ts.K_E_stages[i], new_cap)
            resize!(ts.K_I_stages[i], new_cap)
        end
    end
    
    if length(ts.neighbor_fs) < M_neighbors
        new_cap = ceil(Int, M_neighbors * 1.25)
        resize!(ts.neighbor_fs, new_cap)
        resize!(ts.neighbor_dfs, new_cap)
    end
end

# =========================================================================
# MAIN IMEX FUNCTOR
# =========================================================================

function (imex_ts::GeneralIMEXTimeStepper{M})(
    eq_kin::HyperbolicPDE{D, M}, pg::ParticleGrid{D, M}, time_n::Real, dt::Real
) where {M, D}

    s = imex_ts.num_stages
    bt = imex_ts.butcher_tableau
    N_particles = pg.meta.N 
    
    initAddTSBuffer!(imex_ts, pg)
    
    # 1. Unpack fields and reset step state
    U_n = imex_ts.U_n
    U_n[1:N_particles] .= view(pg.rhos, 1:N_particles)
    fill!(imex_ts.mood_triggered, false)

    # 2. Extract neighbor tracking arrays
    nb_slices  = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices

    for i in 1:s
        # Move grid
        delta_t = i != s ? (bt.ct[i+1] - bt.ct[i]) * dt : (1.0 - bt.ct[i]) * dt
        if delta_t > 0
            imex_ts.grid_mover(pg, delta_t)           
        end
        
        current_Y_i = imex_ts.Y_stages[i]

        # ==================================================================
        # PHASE 1: Accumulate Stages (Native SVector Math!)
        # ==================================================================
        @batch for p_idx in 1:N_particles
            if pg.core.is_boundary[p_idx]; current_Y_i[p_idx] = pg.rhos[p_idx]; continue; end

            # SVector math eliminates the M-component loop entirely!
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
        # PHASE 2: Non-Local Potential (Synchronization Point)
        # ==================================================================
        if imex_ts.source_term_object isa NonLocalRelaxationSourceTerm
            update_nonlocal_potential!(imex_ts.source_term_object, current_Y_i, pg, imex_ts.eq_macro)  
        end

        # ==================================================================
        # PHASE 3: Implicit Solve & K_I Evaluation
        # ==================================================================
        time_implicit = time_n + bt.c[i] * dt
        
        if abs(bt.A[i,i]) > 1e-14
            @batch for p_idx in 1:N_particles
                if pg.core.is_boundary[p_idx]; continue; end
                
                # Stack-allocated MVector allows in-place mutation safely
                Y_mut = MVector{M, Float64}(current_Y_i[p_idx])
                
                solve!(
                    imex_ts.implicit_solver, Y_mut, dt * bt.A[i,i],
                    imex_ts.source_term_object, p_idx, imex_ts.eq_macro, imex_ts.source_term_object.km
                )
                current_Y_i[p_idx] = State{M}(Y_mut)
            end
        end
        
        @batch for p_idx in 1:N_particles
            if pg.core.is_boundary[p_idx]; imex_ts.K_I_stages[i][p_idx] = zero(State{M}); continue; end
            
            S_mut = MVector{M, Float64}(undef)
            imex_ts.source_term_object(S_mut, current_Y_i[p_idx], p_idx, imex_ts.eq_macro, imex_ts.source_term_object.km)
            imex_ts.K_I_stages[i][p_idx] = State{M}(S_mut)
        end

        # ==================================================================
        # PHASE 4: Explicit Gradients K_E (Zero-Allocation Interpolators)
        # ==================================================================
        apply_boundary_conditions!(pg, current_Y_i)
        
        grad = imex_ts.gradientInterpolator
        fallback = imex_ts.fallbackInterpolator
        has_fallback = !(fallback isa NoFallbackGrad)
        
        initGIBuffers!(grad, pg)
        if has_fallback; initGIBuffers!(fallback, pg); end
        
        @batch for p_idx in 1:N_particles
            if pg.core.is_boundary[p_idx]; imex_ts.K_E_stages[i][p_idx] = zero(State{M}); continue; end
            
            fi = current_Y_i[p_idx]
            nb_slice = nb_slices[p_idx]

            initFs!(imex_ts.neighbor_fs, imex_ts.neighbor_dfs, nb_indices, fi, nb_slice, current_Y_i)
            
            if imex_ts.mood_triggered[p_idx]
                initGI!(fallback, p_idx, fi, nb_slice, pg, imex_ts.neighbor_fs, imex_ts.neighbor_dfs)
                div_val = fallback(eq_kin, p_idx, fi, nb_slice, pg, imex_ts.neighbor_fs, imex_ts.neighbor_dfs)
                imex_ts.K_E_stages[i][p_idx] = -div_val
            else
                initGI!(grad, p_idx, fi, nb_slice, pg, imex_ts.neighbor_fs, imex_ts.neighbor_dfs)
                if has_fallback; initGI!(fallback, p_idx, fi, nb_slice, pg, imex_ts.neighbor_fs, imex_ts.neighbor_dfs); end
                
                div_high = grad(eq_kin, p_idx, fi, nb_slice, pg, imex_ts.neighbor_fs, imex_ts.neighbor_dfs) 
                
                rho_candidate = fi - dt * div_high
                
                # Check MOOD on the full State{M} vector at once
                if has_fallback && imex_ts.mood(grad, p_idx, fi, nb_slice, rho_candidate, pg, imex_ts.neighbor_fs)
                    div_fallback = fallback(eq_kin, p_idx, fi, nb_slice, pg, imex_ts.neighbor_fs, imex_ts.neighbor_dfs)
                    imex_ts.K_E_stages[i][p_idx] = -div_fallback
                    imex_ts.mood_triggered[p_idx] = true
                else
                    imex_ts.K_E_stages[i][p_idx] = -div_high
                end
            end
        end 
    end # End of stages loop
    
    # ==================================================================
    # PHASE 5: Final Step Update (With True IMEX Euler MOOD Fallback)
    # ==================================================================
    @batch for p_idx in 1:N_particles
        if pg.core.is_boundary[p_idx]; continue; end
        
        if imex_ts.mood_triggered[p_idx]
            # PERFECT IMEX EULER FALLBACK:
            # 1. TVD Explicit Predictor (K_E_stages[1] was evaluated with the fallback gradient!)
            Y_mut = MVector{M, Float64}(U_n[p_idx] + dt * imex_ts.K_E_stages[1][p_idx])
            
            # 2. L-Stable Implicit Solve for the stiff source term (Backward Euler)
            solve!(
                imex_ts.implicit_solver, Y_mut, dt,
                imex_ts.source_term_object, p_idx, imex_ts.eq_macro, imex_ts.source_term_object.km
            )
            
            pg.rhos[p_idx] = State{M}(Y_mut)
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