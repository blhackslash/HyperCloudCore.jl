# Auto-Sorting Convenience Constructor for standard RK
function GeneralRKTimeStepper(
    pde::HyperbolicPDE{D, M, T}, div_interp::G, 
    all_sources::Tuple{Vararg{AbstractSourceTerm}}, 
    tableau::RKButcherTableau{T}
) where {D, M, T, G}
    explicit_sts = filter(st -> st isa AbstractExplicitSourceTerm, all_sources)
    
    if length(explicit_sts) < length(all_sources)
        @warn "AbstractImplicitSourceTerm detected in a standard Runge-Kutta stepper. It will be ignored! Use IMEX if stiffness is present."
    end
    
    return GeneralRKTimeStepper(pde, div_interp, explicit_sts, tableau)
end

# Fallback for no source terms (Empty Tuple)
function GeneralRKTimeStepper(
    pde::HyperbolicPDE{D, M, T}, div_interp::G, tableau::RKButcherTableau{T}
) where {D, M, T, G}
    return GeneralRKTimeStepper(pde, div_interp, (), tableau)
end

"""
    update_size!(ib::InteractionBuffer, num_interactions)
    update_size!(ts::GeneralIMEXTimeStepper, N_particles, M_neighbors)
    update_size!(ts::GeneralRKTimeStepper, N_particles, M_neighbors)

Dynamically resizes internal buffers and state arrays to accommodate the current number of particles and neighbor interactions.

# Details
- For `InteractionBuffer`, it ensures sufficient capacity for fields like `f`, `df`, `df_flux`, and `mask`.
- For time steppers, it resizes the target RK/IMEX stage arrays (e.g., `Y_stages`, `K_stages`) and automatically cascades the update to the internal neighbor `InteractionBuffer`.
"""
function update_size!(ib::InteractionBuffer, num_interactions::Int)
    ensure_capacity!(ib.f, num_interactions)
    ensure_capacity!(ib.df, num_interactions)
    ensure_capacity!(ib.df_flux, num_interactions)
    ensure_capacity!(ib.df_scratch, num_interactions)
    ensure_capacity!(ib.mask, num_interactions)
    return nothing
end
function update_size!(ts::GeneralRKTimeStepper, N_particles::Int, M_neighbors::Int)
    ensure_capacity!(ts.rho_n, N_particles)
    ensure_capacity!(ts.rho_stage, N_particles)
    
    for i in 1:length(ts.K_stages)
        ensure_capacity!(ts.K_stages[i], N_particles)
    end
    
    update_size!(ts.int_buffer, M_neighbors)
    return nothing
end

"""
    update_content!(ib::InteractionBuffer, nb_indices, f_i, nb_slice, fVec)

Populates the interaction buffer for a given target particle.

# Details
- Retrieves neighbor states from `fVec` using the provided `nb_indices`.
- Directly stores the neighbor state into `ib.f` and computes the raw difference (`f_j - f_i`) into `ib.df` for immediate access during flux evaluation.
"""
@inline function update_content!(
    ib::InteractionBuffer{D, M, T},
    nb_indices::AbstractVector{Int}, 
    f_i::State{M, T}, 
    nb_slice::UnitRange{Int}, 
    fVec::AbstractVector{State{M, T}}
) where {D, M, T}
    
    @inbounds for k in nb_slice
        j = nb_indices[k]
        f_j = fVec[j] 
        
        ib.f[k]  = f_j
        ib.df[k] = f_j - f_i 
    end
    return nothing
end

@inline function evaluate_stage_derivatives!(
    main_grad::DivergenceInterpolator, eq, pg, rk, stage, dt, rho_stage, stage_time
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
        
        div_F = main_grad(eq, p_idx, fi, nb_slice, pg, int_buffer)
        S_expl = evaluate_sources(rk.explicit_sources, fi, p_idx, pg, stage_time)
        
        # RK accumulation uses subtraction, so K = div_F - S_expl
        K_stage[p_idx] = div_F - S_expl
    end
end

@inline function evaluate_stage_derivatives!(
    main_grad::MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, MOOD{S, C}}, 
    eq, pg, rk, stage, dt, rho_stage, stage_time
) where {D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, S <: MOODStrategy, C <: RealMOOD}
    
    N = pg.meta.N
    nb_slices = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices
    is_boundary = pg.core.is_boundary
    int_buffer = rk.int_buffer
    K_stage = rk.K_stages[stage]
    
    orders = main_grad.particle_orders
    needs_recalc = pg.shared.bit_buffer

    update_size!(main_grad, N)
    if stage == 1; fill!(orders, MAX_ORDER); end
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
            
            div_F = main_grad(eq, p_idx, fi, nb_slice, pg, int_buffer)
            S_expl = evaluate_sources(rk.explicit_sources, fi, p_idx, pg, stage_time)
            
            K_stage[p_idx] = div_F - S_expl
        end

        needs_another_pass = evaluate_mood_and_halo!(main_grad, pg, rk, stage, dt, rho_stage)
        if !needs_another_pass || iteration >= 20; break; end
    end
end

function (rk::GeneralRKTimeStepper{D, M, T})(
    eq::HyperbolicPDE, pg::ParticleGrid, time::Real, dt::Real
) where {D, M, T}
    
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
        stage_time = time + c[stage] * dt
        
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
            apply_boundary_conditions!(pg, rho_stage, rk, eq, stage_time)
        end
        
        # Pass stage_time to cleanly evaluate sources
        evaluate_stage_derivatives!(main_grad, eq, pg, rk, stage, dt, rho_stage, stage_time)
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