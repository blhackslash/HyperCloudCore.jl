"""
    GeneralIMEXTimeStepper{D, M, T, PDE, G, MO, EST, IST} <: TimeStepper
    GeneralIMEXTimeStepper(pde::HyperbolicPDE, div_interp, mood, all_sources::Tuple, tableau::IMEXButcherTableau)
    (imex::GeneralIMEXTimeStepper)(eq::HyperbolicPDE, pg::ParticleGrid, time::Real, dt::Real)

A comprehensive IMEX (Implicit-Explicit) time integration orchestrator designed for systems requiring implicit treatment of stiff sources alongside explicit flux evaluations. 

# Constructors

    GeneralIMEXTimeStepper(pde, div_interp, mood, all_sources, tableau)
    GeneralIMEXTimeStepper(pde, div_interp, mood, explicit_sources, implicit_sources, tableau)
    GeneralIMEXTimeStepper(pde, div_interp, mood, tableau)

- `pde`: The physical `HyperbolicPDE` governing the system.
- `div_interp`: The chosen spatial divergence interpolator.
- `mood`: The configured MOOD orchestrator.
- `sources`: Source term tuples. A convenience constructor accepts a mixed tuple (`all_sources`) and automatically separates them into explicit (`EST`) and implicit (`IST`) tuples.
- `tableau`: An `IMEXButcherTableau` containing the dual explicit (`a_t`, `b_t`) and implicit (`a`, `b`) weights.

# Callable / Functor

    (imex::GeneralIMEXTimeStepper)(eq, pg, time, dt)

Advances the simulation by `dt` utilizing an IMEX scheme. The execution follows these steps:
- Pre-allocates memory for both explicit (`K_E`) and implicit (`K_I`) evaluation stages.
- Iterates through the tableau stages, initially forming an explicit predictor state `Y_local`.
- Applies `pre_solve_updates!` for implicit sources, then triggers `implicit_solve` to invert stiff source components locally on the diagonal.
- Accumulates the explicit flux divergence and source derivatives into `K_E_stages` and implicit evaluations into `K_I_stages`.
- Actively evaluates MOOD limits against the combined IMEX candidate states, repeating divergence passes locally if order reduction is triggered.

# Fields
- `pde`, `divergence_interpolator`, `mood`, `explicit_sources`, `implicit_sources`, `tableau`, `num_stages`: Core structural properties.
- `rho_n`: State buffer at the beginning of the time step.
- `Y_stages`: Buffers holding intermediate combined predictor states for each stage.
- `K_E_stages`: Buffers for explicit right-hand side evaluations (flux divergences + explicit sources).
- `K_I_stages`: Buffers for implicit right-hand side evaluations.
- `int_buffer`: The shared `InteractionBuffer` used during explicit neighbor gathering.
"""
struct GeneralIMEXTimeStepper{D, M, T, PDE <: HyperbolicPDE, G <: DivergenceInterpolator, MO <: MOOD, EST <: Tuple{Vararg{AbstractExplicitSourceTerm}}, IST <: Tuple{Vararg{AbstractImplicitSourceTerm}}} <: TimeStepper
    pde::PDE
    divergence_interpolator::G
    mood::MO
    explicit_sources::EST
    implicit_sources::IST
    tableau::IMEXButcherTableau{T}
    
    rho_n::Vector{State{M, T}}
    Y_stages::Vector{Vector{State{M, T}}}
    K_E_stages::Vector{Vector{State{M, T}}}
    K_I_stages::Vector{Vector{State{M, T}}}
    
    int_buffer::InteractionBuffer{D, M, T}
    num_stages::Int

    # Primary strictly-typed constructor
    function GeneralIMEXTimeStepper(
        pde::HyperbolicPDE{D, M, T}, div_interp::G, mood::MO,
        explicit_sources::EST, implicit_sources::IST, 
        tableau::IMEXButcherTableau{T}
    ) where {D, M, T, G, MO <: MOOD, EST <: Tuple{Vararg{AbstractExplicitSourceTerm}}, IST <: Tuple{Vararg{AbstractImplicitSourceTerm}}}
        
        s = size(tableau.a, 1)
        
        new{D, M, T, typeof(pde), G, MO, EST, IST}(
            pde, div_interp, mood, explicit_sources, implicit_sources, tableau,
            State{M, T}[], 
            [State{M, T}[] for _ in 1:s], 
            [State{M, T}[] for _ in 1:s], 
            [State{M, T}[] for _ in 1:s], 
            InteractionBuffer{D, M, T}(), 
            s
        )
    end
end

# Auto-Sorting Convenience Constructor for IMEX
function GeneralIMEXTimeStepper(
    pde::HyperbolicPDE{D, M, T}, div_interp::G, mood::MO,
    all_sources::Tuple{Vararg{AbstractSourceTerm}}, 
    tableau::IMEXButcherTableau{T}
) where {D, M, T, G, MO}
    explicit_sts = filter(st -> st isa AbstractExplicitSourceTerm, all_sources)
    implicit_sts = filter(st -> st isa AbstractImplicitSourceTerm, all_sources)
    
    return GeneralIMEXTimeStepper(pde, div_interp, mood, explicit_sts, implicit_sts, tableau)
end

# Fallback for no source terms (Empty Tuples)
function GeneralIMEXTimeStepper(
    pde::HyperbolicPDE{D, M, T}, div_interp::G, mood::MO, tableau::IMEXButcherTableau{T}
) where {D, M, T, G, MO}
    return GeneralIMEXTimeStepper(pde, div_interp, mood, (), (), tableau)
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

# =========================================================================
# UNIVERSAL IMEX STAGE DERIVATIVE EVALUATOR
# =========================================================================

@inline function evaluate_stage_derivatives_imex!(
    main_grad::DivergenceInterpolator, eq_kin, pg, imex, i, dt, current_Y_i, stage_time
)
    N_particles = pg.meta.N
    nb_slices = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices
    is_boundary = pg.core.is_boundary
    int_buffer = imex.int_buffer
    K_E_stage = imex.K_E_stages[i]

    orders = pg.core.particle_orders
    needs_recalc = pg.shared.bit_buffer

    update_size!(main_grad, N_particles)
    
    if i == 1
        max_order = _extract_order(main_grad)
        fill!(orders, max_order)
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
        
        @smart_parallel use_threads for p_idx in 1:N_particles
            if is_boundary[p_idx]
                K_E_stage[p_idx] = zero(eltype(K_E_stage))
                continue
            end
            if !needs_recalc[p_idx]
                continue
            end
            
            fi = current_Y_i[p_idx]
            nb_slice = nb_slices[p_idx]
            
            div_F = main_grad(eq_kin, p_idx, fi, nb_slice, pg, int_buffer)
            S_expl = evaluate_sources(imex.explicit_sources, fi, p_idx, pg, stage_time)
            
            # IMEX accumulation uses addition, so K_E = -div_F + S_expl
            K_E_stage[p_idx] = -div_F + S_expl
        end

        needs_another_pass = evaluate_mood_and_halo!(main_grad, pg, imex, i, dt, current_Y_i)
        if !needs_another_pass || iteration >= 20; break; end
    end
end

@inline function evaluate_mood_and_halo!(
    main_grad::DivergenceInterpolator, pg::ParticleGrid{D, M, T}, 
    imex_ts::GeneralIMEXTimeStepper, i::Int, dt::Real, current_Y_i::AbstractVector
) where {D, M, T}
    
    mood_fun = imex_ts.mood
    
    # 1. Zero-Cost Fast Exit for Non-Adaptive Schemes
    if mood_fun.criterion isa NoMOOD
        return false
    end
    
    N = pg.meta.N
    nb_slices = pg.neighbor.ranges
    is_boundary = pg.core.is_boundary
    
    int_buffer = imex_ts.int_buffer
    orders = pg.core.particle_orders
    mood_triggered = pg.core.mood_triggered
    needs_recalc = pg.shared.bit_buffer
    bt = imex_ts.tableau
    
    fill!(mood_triggered, false)

    # 2. IMEX Candidate Evaluation Pass
    @batch for p_idx in 1:N
        if is_boundary[p_idx] || !needs_recalc[p_idx]; continue; end
        
        fi = current_Y_i[p_idx]
        nb_slice = nb_slices[p_idx]
        
        div_val = -imex_ts.K_E_stages[i][p_idx] 
        
        s = imex_ts.num_stages
        Y_local = imex_ts.rho_n[p_idx]
        
        if i < s
            for j in 1:(i-1)
                a_t_val = bt.a_t[i+1, j]
                if a_t_val != zero(T); Y_local += (dt * a_t_val) * imex_ts.K_E_stages[j][p_idx]; end
                
                a_val = bt.a[i+1, j]
                if a_val != zero(T); Y_local += (dt * a_val) * imex_ts.K_I_stages[j][p_idx]; end
            end
            
            Y_local += (dt * bt.a[i+1, i]) * imex_ts.K_I_stages[i][p_idx]
            Y_local += (dt * bt.a_t[i+1, i]) * (-div_val)
        else
            for j in 1:(s-1)
                b_t_val = bt.b_t[j]
                if b_t_val != zero(T); Y_local += (dt * b_t_val) * imex_ts.K_E_stages[j][p_idx]; end
                
                b_val = bt.b[j]
                if b_val != zero(T); Y_local += (dt * b_val) * imex_ts.K_I_stages[j][p_idx]; end
            end
            
            Y_local += (dt * bt.b[i]) * imex_ts.K_I_stages[i][p_idx]
            Y_local += (dt * bt.b_t[i]) * (-div_val)
        end
        
        if mood_fun(main_grad, p_idx, fi, nb_slice, Y_local, pg, int_buffer.f)
            mood_triggered[p_idx] = true
        end
    end
    
    fill!(needs_recalc, false)
    any_triggered = false

    # 3. Halo Propagation Pass
    for p_idx in 1:N
        if mood_triggered[p_idx] && orders[p_idx] > 1
            any_triggered = true
            orders[p_idx] -= 1
            needs_recalc[p_idx] = true 
            
            trigger_halo!(mood_fun.strategy, p_idx, pg, needs_recalc, orders)
        end
    end
    
    return any_triggered
end

# =========================================================================
# IMEX TIME STEP FUNCTOR
# =========================================================================

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