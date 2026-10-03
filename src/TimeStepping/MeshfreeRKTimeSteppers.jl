export GeneralRKTimeStepper
"""
    GeneralRKTimeStepper{D, M, T, PDE, G, MO, EST} <: TimeStepper
    GeneralRKTimeStepper(pde::HyperbolicPDE, div_interp, mood, all_sources::Tuple, tableau::RKButcherTableau)
    (rk::GeneralRKTimeStepper)(eq::HyperbolicPDE, pg::ParticleGrid, time::Real, dt::Real)

A standard explicit Runge-Kutta time integration orchestrator. It couples the physical PDE with the spatial divergence interpolator, MOOD adaptive logic, and an explicit Butcher tableau.

# Constructors

    GeneralRKTimeStepper(pde, div_interp, mood, all_sources, tableau)
    GeneralRKTimeStepper(pde, div_interp, mood, explicit_sources, tableau)
    GeneralRKTimeStepper(pde, div_interp, mood, tableau)

- `pde`: The physical `HyperbolicPDE` governing the system.
- `div_interp`: The chosen spatial divergence interpolator.
- `mood`: The configured MOOD orchestrator for adaptive spatial order reduction.
- `sources`: Source term tuples. If `all_sources` is passed, it automatically filters out implicit terms (issuing a warning) to retain only `AbstractExplicitSourceTerm`s.
- `tableau`: An `RKButcherTableau` defining the explicit stage weights.

# Callable / Functor

    (rk::GeneralRKTimeStepper)(eq, pg, time, dt)

Advances the particle grid `pg` forward in time by `dt` using an explicit Runge-Kutta method. The execution follows these steps:
- Pre-allocates or resizes stage buffers to match the current particle count.
- Iterates through the Runge-Kutta stages, computing explicit flux divergences and explicit source terms.
- Evaluates the MOOD criteria after each stage evaluation and dynamically re-triggers divergence computations for particles that require order degradation (propagating halos).
- Applies boundary conditions at intermediate stages and at the final time step.

# Fields
- `pde`, `divergence_interpolator`, `mood`, `explicit_sources`, `tableau`: Core structural components.
- `rho_n`: State buffer at the beginning of the time step.
- `rho_stage`: Buffer for the intermediate stage candidate state.
- `K_stages`: A vector of state arrays storing the combined divergence and explicit source evaluations for each RK stage.
- `int_buffer`: A shared `InteractionBuffer` for gathering neighbor states during flux evaluations.
"""
struct GeneralRKTimeStepper{D, M, T, PDE <: HyperbolicPDE, G <: DivergenceInterpolator, MO <: MOOD, EST <: Tuple{Vararg{AbstractExplicitSourceTerm}}} <: TimeStepper
    pde::PDE
    divergence_interpolator::G
    mood::MO
    explicit_sources::EST
    tableau::RKButcherTableau{T}
    
    rho_n::Vector{State{M, T}}
    rho_stage::Vector{State{M, T}}
    K_stages::Vector{Vector{State{M, T}}} 
    int_buffer::InteractionBuffer{D, M, T}

    # Primary strictly-typed constructor
    function GeneralRKTimeStepper(
        pde::HyperbolicPDE{D, M, T}, div_interp::G, mood::MO,
        explicit_sources::EST, tableau::RKButcherTableau{T}
    ) where {D, M, T, G, MO <: MOOD, EST <: Tuple{Vararg{AbstractExplicitSourceTerm}}}
        
        s = size(tableau.a, 1)
        
        new{D, M, T, typeof(pde), G, MO, EST}(
            pde, div_interp, mood, explicit_sources, tableau,
            State{M, T}[], State{M, T}[], 
            [State{M, T}[] for _ in 1:s],
            InteractionBuffer{D, M, T}()
        )
    end
end

# Auto-Sorting Convenience Constructor for standard RK
function GeneralRKTimeStepper(
    pde::HyperbolicPDE{D, M, T}, div_interp::G, mood::MO,
    all_sources::Tuple{Vararg{AbstractSourceTerm}}, 
    tableau::RKButcherTableau{T}
) where {D, M, T, G, MO}
    explicit_sts = filter(st -> st isa AbstractExplicitSourceTerm, all_sources)
    
    if length(explicit_sts) < length(all_sources)
        @warn "AbstractImplicitSourceTerm detected in a standard Runge-Kutta stepper. It will be ignored! Use IMEX if stiffness is present."
    end
    
    return GeneralRKTimeStepper(pde, div_interp, mood, explicit_sts, tableau)
end

# Fallback for no source terms (Empty Tuple)
function GeneralRKTimeStepper(
    pde::HyperbolicPDE{D, M, T}, div_interp::G, mood::MO, tableau::RKButcherTableau{T}
) where {D, M, T, G, MO}
    return GeneralRKTimeStepper(pde, div_interp, mood, (), tableau)
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

# =========================================================================
# UNIVERSAL STAGE DERIVATIVE EVALUATOR
# =========================================================================

@inline function evaluate_stage_derivatives!(
    main_grad::DivergenceInterpolator, eq, pg, rk, stage, dt, rho_stage, stage_time
)
    N = pg.meta.N
    nb_slices = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices
    is_boundary = pg.core.is_boundary
    int_buffer = rk.int_buffer
    K_stage = rk.K_stages[stage]
    
    orders = pg.core.particle_orders
    needs_recalc = pg.shared.bit_buffer

    update_size!(main_grad, N)
    
    # Initialize particle orders based on the configured maximum order of the scheme
    if stage == 1
        max_order = _extract_order(main_grad)
        fill!(orders, max_order)
    end
    fill!(needs_recalc, true)
    
    use_threads = _use_threads()
    iteration = 0
    
    while true
        iteration += 1

        # 1. Pre-Gather Pass (Calculate raw/limited gradients or stencils)
        @smart_parallel use_threads for p_idx in 1:N
            if is_boundary[p_idx] || !needs_recalc[p_idx]; continue; end
            
            fi = rho_stage[p_idx]
            nb_slice = nb_slices[p_idx]
            
            update_content!(int_buffer, nb_indices, fi, nb_slice, rho_stage)
            update_content!(main_grad, p_idx, fi, nb_slice, pg, int_buffer)
        end

        # 2. Flux/Divergence Pass
        @smart_parallel use_threads for p_idx in 1:N
            if is_boundary[p_idx] || !needs_recalc[p_idx]; continue; end
            
            fi = rho_stage[p_idx]
            nb_slice = nb_slices[p_idx]
            
            div_F = main_grad(eq, p_idx, fi, nb_slice, pg, int_buffer)
            S_expl = evaluate_sources(rk.explicit_sources, fi, p_idx, pg, stage_time)
            
            K_stage[p_idx] = div_F - S_expl
        end

        # 3. MOOD Validation & Halo Triggering (Now completely generic!)
        needs_another_pass = evaluate_mood_and_halo!(main_grad, pg, rk, stage, dt, rho_stage)
        
        # Break if all particles passed the MOOD criteria or we hit the hard fallback limit
        if !needs_another_pass || iteration >= 20; break; end
    end
end

"""
    evaluate_mood_and_halo!(main_grad, pg, ts, stage, dt, rho_stage)

Evaluates the configured MOOD criterion across the domain for either Runge-Kutta or IMEX time steppers and propagates order reduction halos.

# Details
- Safely early-exits if `NoMOOD` is configured, guaranteeing zero overhead for non-adaptive schemes.
- Calculates the candidate state locally by resolving the specific Runge-Kutta or IMEX tableau.
- Queries `pg.core` for universal particle orders and triggers order reduction and spatial halos universally across any spatial scheme.
"""
@inline function evaluate_mood_and_halo!(
    main_grad::DivergenceInterpolator, pg::ParticleGrid{D, M, T}, 
    rk::GeneralRKTimeStepper, stage::Int, dt::Real, rho_stage::AbstractVector
) where {D, M, T}
    
    mood_fun = rk.mood
    
    # 1. Zero-Cost Fast Exit for Non-Adaptive Schemes
    if mood_fun.criterion isa NoMOOD
        return false
    end
    
    N = pg.meta.N
    nb_slices = pg.neighbor.ranges
    is_boundary = pg.core.is_boundary
    
    int_buffer = rk.int_buffer
    orders = pg.core.particle_orders
    mood_triggered = pg.core.mood_triggered
    needs_recalc = pg.shared.bit_buffer
    
    fill!(mood_triggered, false)

    # 2. Candidate Evaluation Pass
    @batch for p_idx in 1:N
        if is_boundary[p_idx] || !needs_recalc[p_idx]; continue; end
        
        fi = rho_stage[p_idx]
        nb_slice = nb_slices[p_idx]
        div_val = rk.K_stages[stage][p_idx]

        s = length(rk.K_stages)
        base_rho = rk.rho_n[p_idx]
        
        # Accumulate the RK stage dynamically
        if stage < s
            A_coef = rk.tableau.a[stage+1, stage]
            for j in 1:(stage-1)
                a_val = rk.tableau.a[stage+1, j]
                if a_val != zero(T); base_rho -= dt * a_val * rk.K_stages[j][p_idx]; end
            end
        else
            A_coef = rk.tableau.b[stage]
            for j in 1:(s-1)
                b_val = rk.tableau.b[j]
                if b_val != zero(T); base_rho -= dt * b_val * rk.K_stages[j][p_idx]; end
            end
        end
        
        rho_candidate = base_rho - dt * A_coef * div_val
        
        # Evaluate generic criteria (DMP, U2, etc.)
        if mood_fun(main_grad, p_idx, fi, nb_slice, rho_candidate, pg, int_buffer.f)
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