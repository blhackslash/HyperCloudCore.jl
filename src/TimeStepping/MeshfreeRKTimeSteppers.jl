"""
    evaluate_stage_derivatives!(main_grad, eq, pg, rk, stage, dt, rho_stage)

Evaluates the spatial divergence for a specific Runge-Kutta stage across the particle grid.

# Details
- Allocates or resizes interaction buffers and evaluates the `main_grad` divergence interpolator across non-boundary particles utilizing smart parallel thread execution.
- **MUSCL Dispatch:** If a `MUSCL` divergence interpolator configured with a MOOD strategy is provided, it triggers an iterative evaluation loop. It continuously resets effective spatial orders, evaluates the divergence, and invokes the MOOD halo trigger until the scheme is fully satisfied or a maximum cap of 20 iterations is reached.
"""
@inline function evaluate_stage_derivatives!(
    main_grad::DivergenceInterpolator, eq, pg, rk, stage, dt, rho_stage
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

@inline function evaluate_stage_derivatives!(
    main_grad::MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, MOOD{S, C}}, 
    eq, pg, rk, stage, dt, rho_stage
) where {D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, S <: MOODStrategy, C <: RealMOOD}
    
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

        @smart_parallel use_threads for p_idx in 1:N
            if is_boundary[p_idx] || !needs_recalc[p_idx]; continue; end
            
            fi = rho_stage[p_idx]
            nb_slice = nb_slices[p_idx]
            
            update_content!(int_buffer, nb_indices, fi, nb_slice, rho_stage)
            update_content!(main_grad, p_idx, fi, nb_slice, pg, int_buffer)
        end

        effective_orders = pg.shared.int_buffer
        @smart_parallel use_threads for p_idx in 1:N
            if is_boundary[p_idx]; continue; end
            nb_slice = nb_slices[p_idx]
            effective_orders[p_idx] = get_effective_order(main_grad.mood.strategy, orders, p_idx, nb_slice, nb_indices)
        end

        @smart_parallel use_threads for p_idx in 1:N
            if is_boundary[p_idx] || !needs_recalc[p_idx]; continue; end
            
            fi = rho_stage[p_idx]
            nb_slice = nb_slices[p_idx]
            
            K_stage[p_idx] = main_grad(eq, p_idx, fi, nb_slice, pg, int_buffer)
        end

        needs_another_pass = evaluate_mood_and_halo!(main_grad, pg, rk, stage, dt, rho_stage)
        
        if !needs_another_pass || iteration >= 20
            break
        end
    end
end

"""
    (rk::GeneralRKTimeStepper)(eq::HyperbolicPDE, pg::ParticleGrid, time::Real, dt::Real, source_term=NoSourceTerm())

Executes a complete explicit Runge-Kutta time step, advancing the particle states from `time` to `time + dt`.

# Details
- Resolves intermediate physical time steps for each stage and applies coordinate displacement via `pg.mover` if the temporal advance is greater than zero.
- Updates local states iteratively applying explicit RK weights (`A` and `b`), evaluating the stage derivatives, and strictly enforcing boundary conditions at each sub-stage.
- Repopulates the neighbor lists dynamically via `pg.neighbor(pg)` if a final temporal displacement shifts the mesh configuration before executing the final state summation.
"""
function (rk::GeneralRKTimeStepper{D, M, T})(
    eq::HyperbolicPDE{D, M, T, R}, pg::ParticleGrid, time::Real, dt::Real, 
    source_term::AbstractSourceTerm = NoSourceTerm()
) where {D, M, T, R}
    
    N = pg.meta.N
    M_neighbors = length(pg.neighbor.indices)
    s = length(rk.K_stages)
    A = rk.tableau.a
    b = rk.tableau.b
    c = rk.tableau.c
    
    update_size!(rk, N, M_neighbors)
    
    rho_n       = rk.rho_n
    rho_stage   = rk.rho_stage
    K_stages    = rk.K_stages
    main_grad   = rk.divergence_interpolator
    
    rhos        = pg.rhos
    is_boundary = pg.core.is_boundary
    
    rho_n[1:N] .= view(rhos, 1:N)
    
    for stage in 1:s
        delta_t = stage == 1 ? c[stage] * dt : (c[stage] - c[stage-1]) * dt

        if stage == 1
            rho_stage[1:N] .= view(rho_n, 1:N)
        else
            @batch for p_idx in 1:N
                if is_boundary[p_idx]; continue; end
                
                u_stage = rho_n[p_idx]
                for j in 1:(stage-1)
                    if A[stage, j] != zero(T)
                        u_stage -= dt * A[stage, j] * K_stages[j][p_idx]
                    end
                end
                rho_stage[p_idx] = u_stage
            end
            stage_time = time + c[stage] * dt
            apply_boundary_conditions!(pg, rho_stage, rk, eq, stage_time)
        end
        
        evaluate_stage_derivatives!(main_grad, eq, pg, rk, stage, dt, rho_stage)
    end

    @batch for p_idx in 1:N
        if is_boundary[p_idx]; continue; end
        
        rho_final = rho_n[p_idx]
        for j in 1:s
            if b[j] != zero(T)
                rho_final -= dt * b[j] * K_stages[j][p_idx]
            end
        end
        rhos[p_idx] = rho_final
    end
    
    apply_boundary_conditions!(pg, rhos, rk, eq, time + dt)
end