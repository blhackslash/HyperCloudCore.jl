function GeneralIMEXTimeStepper(
        gradientInterpolator::G1, fallbackInterpolator::G2, mood::MOOD,
        implicit_solver::IS, source_term_object::ST_OBJ, butcher_tableau::BT
    ) where {G1, G2, MOOD, IS, ST_OBJ, BT}
    
    s = size(butcher_tableau.A, 1) # Number of stages
    M = source_term_object.num_total_kinetic_components
    
    GeneralIMEXTimeStepper{M, G1, G2, MOOD, IS, ST_OBJ, BT}(
        ntuple(_ -> deepcopy(gradientInterpolator), M), 
        ntuple(_ -> deepcopy(fallbackInterpolator), M), 
        mood, implicit_solver, source_term_object, butcher_tableau, grid_mover,
        Matrix{Float64}(undef, 0, 0), 
        [Matrix{Float64}(undef, 0, 0) for _ in 1:s],
        [Matrix{Float64}(undef, 0, 0) for _ in 1:s], 
        [Matrix{Float64}(undef, 0, 0) for _ in 1:s], 
        falses(0, M, s),
        Matrix{Float64}(undef, 0, M), 
        Matrix{Float64}(undef, 0, M),
        s
    )
end

function initAddTSBuffer!(imex_ts::GeneralIMEXTimeStepper{M}, pg::ParticleGrid) where {M}
    N_particles = pg.meta.N
    total_neighbors = length(pg.neighbor.indices)

    # 1. Resize Stage Matrices
    if size(imex_ts.U_n_sys, 1) < N_particles
        new_cap = ceil(Int, N_particles * 1.25)
        imex_ts.U_n_sys = Matrix{Float64}(undef, new_cap, M)
        
        for i in 1:imex_ts.num_stages
            imex_ts.Y_stages_sys[i]   = Matrix{Float64}(undef, new_cap, M)
            imex_ts.K_E_stages_sys[i] = Matrix{Float64}(undef, new_cap, M)
            imex_ts.K_I_stages_sys[i] = Matrix{Float64}(undef, new_cap, M)
        end
        imex_ts.mood_triggered = falses(new_cap, M, imex_ts.num_stages)
    end
    
    # 2. Resize Neighbor Buffers
    if size(imex_ts.all_neighbor_fs, 1) < total_neighbors
        new_nb_cap = ceil(Int, total_neighbors * 1.25)
        imex_ts.all_neighbor_fs  = Matrix{Float64}(undef, new_nb_cap, M)
        imex_ts.all_neighbor_dfs = Matrix{Float64}(undef, new_nb_cap, M)
    end
end

# --- Optimized Neighbor Extraction ---
# Extracts the neighbor function values for ALL components simultaneously 
# using Column-Major cache-friendly loops.
@inline function initFs!(
    ts::GeneralIMEXTimeStepper, 
    nb_slice::UnitRange{Int}, 
    f_i_vec::AbstractVector{Float64}, 
    Y_sys::AbstractMatrix{Float64}, 
    pg::ParticleGrid
)   
    neighbors = get_neighbors(pg)
    
    # Outer loop over COLUMNS
    @inbounds for c in axes(Y_sys, 2)
        
        # Hoist the center particle's value out of the inner loop
        f_ic = f_i_vec[c] 
        
        # Inner loop over ROWS (Sequential memory access!)
        for k in nb_slice
            j = neighbors[k] # This read is random, which is unavoidable
            
            f_j = Y_sys[j, c]
            ts.all_neighbor_fs[k, c]  = f_j
            ts.all_neighbor_dfs[k, c] = f_j - f_ic
        end
    end
end
# --- REFACTORED Functor for GeneralIMEXTimeStepper ---
function (imex_ts::GeneralIMEXTimeStepper{M, G1, G2, MOOD, IS, ST_OBJ, BT})(
        scalar_equations::DiagonalHyperbolicSystem{M, D},
        pg::ParticleGrid{D, M},
        settings::Any, # Replaced SimSetting with Any to decouple specific types if needed
        time_n::Real,
        dt::Real
    ) where {M, D, G1, G2, MOOD, IS, ST_OBJ, BT}

    s = imex_ts.num_stages
    bt = imex_ts.butcher_tableau
    
    # 1. Geometry updates on the unified grid
    manage_particles!(pg, imex_ts.source_term_object) 
    update_grid_velocities!(pg, imex_ts.grid_mover)
    
    N_particles = pg.meta.N 
    fill!(imex_ts.mood_triggered, false)

    for i in 1:s
        # Move the unified grid
        if i != s
            imex_ts.grid_mover(pg, (bt.ct[i+1] - bt.ct[i]) * dt; managed = false)
        else
            imex_ts.grid_mover(pg, (1.0 - bt.ct[i]) * dt; managed = false)           
        end
        
        initAddTSBuffer!(imex_ts, pg)       
        current_Y_i_sys = imex_ts.Y_stages_sys[i]

        # ==================================================================
        # PHASE 1: Accumulate Stages (Register Blocked over M)
        # ==================================================================
        @batch for p_idx in 1:N_particles
            if pg.core.is_boundary[p_idx]
                for k in 1:M
                    current_Y_i_sys[p_idx, k] = pg.rhos[p_idx, k]
                end
                continue
            end

            # Load pristine state into stack allocated MVector
            Y_local = MVector{M, Float64}(undef)
            for k in 1:M
                Y_local[k] = pg.rhos[p_idx, k]
            end
            
            # Accumulate explicit and implicit RK components
            for j in 1:(i-1)
                for k in 1:M
                    if imex_ts.mood_triggered[p_idx, k, j]
                        Y_local[k] += dt * (bt.ct[j+1] - bt.ct[j]) * imex_ts.K_E_stages_sys[j][p_idx, k]
                    else
                        Y_local[k] += dt * bt.At[i,j] * imex_ts.K_E_stages_sys[j][p_idx, k]
                    end
                    if bt.A[i,j] != 0.0
                        Y_local[k] += dt * bt.A[i,j] * imex_ts.K_I_stages_sys[j][p_idx, k]
                    end
                end
            end
            
            # Write accumulated state back to RAM
            for k in 1:M
                current_Y_i_sys[p_idx, k] = Y_local[k]
            end
        end

        # ==================================================================
        # PHASE 2: Non-Local Potential (Synchronization Point)
        # ==================================================================
        if imex_ts.source_term_object isa NonLocalRelaxationSourceTerm
            update_nonlocal_potential!(imex_ts.source_term_object, current_Y_i_sys, pg)  
        end

        # ==================================================================
        # PHASE 3: Implicit Solve & K_I Evaluation
        # ==================================================================
        time_implicit = time_n + bt.c[i] * dt
        
        if abs(bt.A[i,i]) > 1e-14
            @batch for p_idx in 1:N_particles
                # Pass view of the particle's full multi-component state directly
                u_particle_view = @view current_Y_i_sys[p_idx, :]
                
                solve!(
                    imex_ts.implicit_solver, 
                    u_particle_view, 
                    dt * bt.A[i,i],
                    imex_ts.source_term_object, p_idx,
                    pg.core.positions[p_idx], 
                    time_implicit, M
                )
            end
        end
        
        # Evaluate Source Term K_I
        @batch for p_idx in 1:N_particles
            imex_ts.source_term_object(
                @view(imex_ts.K_I_stages_sys[i][p_idx, :]), 
                @view(current_Y_i_sys[p_idx, :]), 
                p_idx,
                pg.core.positions[p_idx], 
                time_implicit
            )
        end

        # ==================================================================
        # PHASE 4: Explicit Gradients K_E
        # ==================================================================
        # 1. Apply single matrix boundary condition
        apply_boundary_conditions!(pg, current_Y_i_sys)
        
        for k in 1:M
            initGIBuffers!(imex_ts.gradientInterpolator[k], pg)
            initGIBuffers!(imex_ts.fallbackInterpolator[k], pg)
        end
        
        # 2. Extract neighbor information & solve divergences
        @batch for p_idx in 1:N_particles
            # Fuses extraction of all M f_j values at once
            f_i_vec = @view current_Y_i_sys[p_idx, :]
            nb_slice = pg.neighbor.ranges[p_idx]

            initFs!(imex_ts, nb_slice, f_i_vec, current_Y_i_sys, pg)
            
            for k in 1:M
                fi = current_Y_i_sys[p_idx, k]
                neighbor_fs_k  = @inbounds @view imex_ts.all_neighbor_fs[:, k]
                neighbor_dfs_k = @inbounds @view imex_ts.all_neighbor_dfs[:, k]
                
                initGI!(imex_ts.gradientInterpolator[k], p_idx, fi, nb_slice, pg, neighbor_fs_k, neighbor_dfs_k)
                initGI!(imex_ts.fallbackInterpolator[k], p_idx, fi, nb_slice, pg, neighbor_fs_k, neighbor_dfs_k)
                
                if !pg.core.is_boundary[p_idx]
                    eq = scalar_equations[k] 
                    div_high = imex_ts.gradientInterpolator[k](eq, p_idx, fi, nb_slice, pg, neighbor_fs_k, neighbor_dfs_k) 
                    rho_candidate = fi - dt * div_high
                    
                    # Apply MOOD Criterion per component
                    has_fallback = !(imex_ts.fallbackInterpolator[k] isa NoFallbackGrad)
                    if has_fallback && imex_ts.mood(imex_ts.gradientInterpolator[k], p_idx, fi, nb_slice, rho_candidate, pg, neighbor_fs_k)
                        div_fallback = imex_ts.fallbackInterpolator[k](eq, p_idx, fi, nb_slice, pg, neighbor_fs_k, neighbor_dfs_k)
                        imex_ts.K_E_stages_sys[i][p_idx, k] = -div_fallback
                        imex_ts.mood_triggered[p_idx, k, i] = true
                    else
                        imex_ts.K_E_stages_sys[i][p_idx, k] = -div_high
                    end
                end
            end
        end 
    end # End of stages loop
    
    # ==================================================================
    # PHASE 5: Final Step Update
    # ==================================================================
    for i in 1:s
        dt_bt = dt * bt.bt[i]
        dt_b  = dt * bt.b[i]
        
        @batch for p_idx in 1:N_particles
            if pg.core.is_boundary[p_idx]
                continue
            end
            
            for k in 1:M
                pg.rhos[p_idx, k] += dt_bt * imex_ts.K_E_stages_sys[i][p_idx, k] + dt_b * imex_ts.K_I_stages_sys[i][p_idx, k]
            end
        end
    end

    # Finally, apply boundary conditions to the unified pristine state
    apply_boundary_conditions!(pg, pg.rhos)
end

# --- IMEX Configurator Constructors ---

function ARS233(
    gradientInterpolator::G1, fallbackInterpolator::G2, mood_criterion::MOOD,
    implicit_solver::IS, source_term_object::ST_OBJ,
    gamma_coefficient::Float64 = (3.0 + sqrt(3.0))/6.0 
) where {G1, G2, MOOD, IS, ST_OBJ}
    
    tableau = IMEXARS233ButcherTableau(gamma_coefficient) 
    return GeneralIMEXTimeStepper(gradientInterpolator, fallbackInterpolator, mood_criterion, implicit_solver, source_term_object, tableau, grid_mover)
end

function PareschiRussoIMEXSSP3(
    gradientInterpolator::G1, fallbackInterpolator::G2, mood_criterion::MOOD,
    implicit_solver::IS, source_term_object::ST_OBJ
) where {G1, G2, MOOD, IS, ST_OBJ}
    
    tableau = PR_IMEX_SSP3_ButcherTableau() 
    return GeneralIMEXTimeStepper(gradientInterpolator, fallbackInterpolator, mood_criterion, implicit_solver, source_term_object, tableau, grid_mover)
end

function ARS222(
    gradientInterpolator::G1, fallbackInterpolator::G2, mood_criterion::MOOD,
    implicit_solver::IS, source_term_object::ST_OBJ,
    gamma_coefficient::Union{Float64,Nothing}=nothing 
) where {G1, G2, MOOD, IS, ST_OBJ}
    
    tableau = ARS222_ButcherTableau(gamma_coefficient) 
    return GeneralIMEXTimeStepper(gradientInterpolator, fallbackInterpolator, mood_criterion, implicit_solver, source_term_object, tableau, grid_mover)
end

function SSP2332(
    gradientInterpolator::G1, fallbackInterpolator::G2, mood_criterion::MOOD,
    implicit_solver::IS, source_term_object::ST_OBJ
) where {G1, G2, MOOD, IS, ST_OBJ}
    
    tableau = SSP2332ButcherTableau() 
    return GeneralIMEXTimeStepper(gradientInterpolator, fallbackInterpolator, mood_criterion, implicit_solver, source_term_object, tableau, grid_mover)
end