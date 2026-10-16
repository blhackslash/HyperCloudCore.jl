export GeneralRKTimeStepper, GeneralIMEXTimeStepper, mood_flags

"""
    MOODWork(storage = Array)

Scratch owned by the time stepper: `triggered` (this pass), `changed` (orders that moved, i.e. C),
`dirty` (fluxes to recompute, C ⊕ reach), `scratch`, an `Int` order snapshot for forcing, and
`triggered_step`, a per-step diagnostic of every particle that dropped an order (see `mood_flags`).
The masks are `Growable` buffers with arrays of type `storage`.
"""
struct MOODWork{GB <: Growable{Bool}, GI <: Growable{Int}}
    triggered::GB
    changed::GB
    dirty::GB
    scratch::GB
    snapshot::GI
    triggered_step::GB
    N::Base.RefValue{Int}           # particles of the last `update_size!`: the active part of the masks
end
MOODWork(::Type{AT} = Array) where {AT <: AbstractArray} =
    MOODWork(Growable{Bool}(AT), Growable{Bool}(AT), Growable{Bool}(AT), Growable{Bool}(AT), Growable{Int}(AT), Growable{Bool}(AT), Ref(0))

function update_size!(w::MOODWork, N::Int)
    ensure_capacity!(w.triggered, N); ensure_capacity!(w.changed, N)
    ensure_capacity!(w.dirty, N);     ensure_capacity!(w.scratch, N)
    ensure_capacity!(w.snapshot, N);  ensure_capacity!(w.triggered_step, N)
    w.N[] = N
    return nothing
end

"""
    InteractionBuffer{D, M, T}(storage = Array)

Per-slot scratch of the schemes, indexed by global neighbour slot (so per-particle slices are
disjoint and the passes are thread-safe). The host object holds `Growable` buffers with arrays of
type `storage`; kernels receive it with the raw arrays (see `DeviceAdapt.jl`). Neighbour states are
not buffered: schemes read them from the state vector through `neighbour_states` /
`neighbour_differences`.

# Fields
- `df_flux`: numerical flux differences / non-conservative jumps.
- `df_scratch`: MLS scratch.
- `mask`: directional stencil filter.
"""
struct InteractionBuffer{D, M, T, VS <: AbstractVector{State{M, T}}, VF <: AbstractVector{Flux{D, M, T}}, VB <: AbstractVector{Bool}}
    df_flux::VF
    df_scratch::VS
    mask::VB
end
InteractionBuffer{D, M, T}(df_flux::VF, df_scratch::VS, mask::VB) where {D, M, T, VS, VF, VB} =
    InteractionBuffer{D, M, T, VS, VF, VB}(df_flux, df_scratch, mask)
InteractionBuffer{D, M, T}(::Type{AT} = Array) where {D, M, T, AT <: AbstractArray} =
    InteractionBuffer{D, M, T}(Growable{Flux{D, M, T}}(AT), Growable{State{M, T}}(AT), Growable{Bool}(AT))

function update_size!(ib::InteractionBuffer, num_interactions::Int)
    ensure_capacity!(ib.df_flux, num_interactions)
    ensure_capacity!(ib.df_scratch, num_interactions)
    ensure_capacity!(ib.mask, num_interactions)
    return nothing
end

# =========================================================================
# SOURCE TUPLE UNROLLERS
# =========================================================================

@inline @generated function evaluate_sources(sts::Tuple{Vararg{AbstractSourceTerm}}, U, p_idx::Int, pg, t::Real)
    N = length(sts.parameters)
    N == 0 && return :(zero(U))
    expr = :(evaluate_source(sts[1], U, p_idx, pg, t))
    for i in 2:N
        expr = :($expr + evaluate_source(sts[$i], U, p_idx, pg, t))
    end
    return expr
end

@inline function pre_solve_updates!(sts::Tuple{Vararg{AbstractImplicitSourceTerm}}, Y_stage, pg, t::Real)
    for st in sts
        pre_solve_update!(st, Y_stage, pg, t)
    end
end

@inline @generated function implicit_solve(sts::Tuple{Vararg{AbstractImplicitSourceTerm}}, U_in, dt_coeff::Real, p_idx::Int, pg, t::Real)
    N = length(sts.parameters)
    N == 0 && return :(U_in)
    quote
        U_out = U_in
        Base.Cartesian.@nexprs $N i -> U_out = implicit_solve(sts[i], U_out, dt_coeff, p_idx, pg, t)
        return U_out
    end
end

include("ButcherTableaus.jl")
include("MOOD.jl")
include("GridMovers.jl")

# =========================================================================
# THE TIME STEPPER
# =========================================================================
"""
    GeneralRKTimeStepper{D, M, T, PDE, G, MO, EST, IST} <: TimeStepper
    GeneralRKTimeStepper(pde, div_interp, mood, sources::Tuple, tableau; kwargs...)
    GeneralRKTimeStepper(pde, div_interp, mood, tableau; kwargs...)
    (ts::GeneralRKTimeStepper)(eq, pg, time, dt::T)

The one time integrator: explicit RK and IMEX share this struct, since an explicit tableau is an
`IMEXButcherTableau` with a zero implicit part (see `RKButcherTableau`). `sources` may mix explicit
and implicit terms. If `pg.mover` is not `NoGridMover`, particle positions are integrated as
additional stage variables of the same tableau (see `GridMovers.jl`).

# Algorithm (per time step)
Orders start at the interpolator's maximum consistency order. For each stage `i`:
1. Form the stage state `Y_i` (explicit predictor, implicit diagonal solve, `K_I[i]`, BCs) and,
   when moving, the stage positions `X_i`.
2. For `i > 1`, run the **stage-wise MOOD fixed point** against `Y_{i-1}` at geometry `X_{i-1}`:
   flagged particles drop an order, only the affected part of `K_E[i-1]` (set `C ⊕ _order_reach`)
   is recomputed, `Y_i` is re-formed, until nothing is flagged.
3. Switch the grid geometry to `X_i`, then compute `K_E[i]` (and `K_X[i] = w`) for all interior
   particles.
The final combination is checked the same way against `Y_s`; the final positions are written back
and a full neighbour search is run. Every stage is treated as a forward-Euler substep, which is
exact for SSP tableaus.

# Keyword options
- `reset_orders_per_stage = false`: restore the maximum order before each stage's full RHS pass.
- `full_search_per_stage = false`: rebuild the neighbour topology at every stage position
  (default: freeze topology, refresh distances/weights only).
- `volume_estimator = ShepardVolumes()`: how `pg.core.volumes` is recomputed after each step of a
  moving grid (`nothing` leaves the volumes untouched).
- `storage = Array`: array type of the stage buffers, e.g. `ROCArray` for a stepper that advances a
  grid built with `device_grid` (the interpolator needs the same `storage`).

# Fields
- `rho_n`, `rho_cand`, `Y_stages`, `K_E_stages`, `K_I_stages`: states and RHS (`K_E = -div(F − w_i ρ) + S_expl`).
- `x_n`, `x_cand`, `X_stages`, `K_X_stages`: positions and grid velocities (sized only when moving).
- `int_buffer`, `work`: neighbour gather buffer and MOOD scratch.
All of them are `Growable` buffers, sized by `update_size!` at the start of every step.
"""
struct GeneralRKTimeStepper{D, M, T, PDE <: HyperbolicPDE, G <: DivergenceInterpolator, MO <: MOOD,
                            EST <: Tuple{Vararg{AbstractExplicitSourceTerm}},
                            IST <: Tuple{Vararg{AbstractImplicitSourceTerm}},
                            VE <: Union{Nothing, VolumeEstimator}, S,
                            GS <: Growable{State{M, T}}, GP <: Growable{Space{D, T}},
                            IB <: InteractionBuffer{D, M, T}, WORK <: MOODWork,
                            BT <: IMEXButcherTableau{T}} <: TimeStepper
    pde::PDE
    divergence_interpolator::G
    mood::MO
    explicit_sources::EST
    implicit_sources::IST
    tableau::BT          # static form (`static_tableau`): a bits kernel argument of concrete type
    reset_orders_per_stage::Bool
    full_search_per_stage::Bool
    volume_estimator::VE

    rho_n::GS
    rho_cand::GS
    # Stage arrays as tuples (one entry per stage): indexable inside kernels, unlike a Vector of Vectors.
    Y_stages::NTuple{S, GS}
    K_E_stages::NTuple{S, GS}
    K_I_stages::NTuple{S, GS}

    x_n::GP
    x_cand::GP
    X_stages::NTuple{S, GP}
    K_X_stages::NTuple{S, GP}

    int_buffer::IB
    work::WORK
    num_stages::Int

    function GeneralRKTimeStepper(
        pde::HyperbolicPDE{D, M, T}, div_interp::G, mood::MO,
        explicit_sources::EST, implicit_sources::IST, tableau::IMEXButcherTableau{T};
        reset_orders_per_stage::Bool = false, full_search_per_stage::Bool = false,
        volume_estimator::VE = ShepardVolumes(), storage::Type{AT} = Array
    ) where {D, M, T, G, MO <: MOOD, EST <: Tuple{Vararg{AbstractExplicitSourceTerm}}, IST <: Tuple{Vararg{AbstractImplicitSourceTerm}},
             VE <: Union{Nothing, VolumeEstimator}, AT <: AbstractArray}
        s  = num_stages(tableau)
        bt = static_tableau(tableau, Val(s))
        states    = () -> Growable{State{M, T}}(AT)
        positions = () -> Growable{Space{D, T}}(AT)
        ib   = InteractionBuffer{D, M, T}(AT)
        work = MOODWork(AT)
        new{D, M, T, typeof(pde), G, MO, EST, IST, VE, s, typeof(states()), typeof(positions()), typeof(ib), typeof(work), typeof(bt)}(
            pde, div_interp, mood, explicit_sources, implicit_sources, bt,
            reset_orders_per_stage, full_search_per_stage, volume_estimator,
            states(), states(),
            ntuple(_ -> states(), s), ntuple(_ -> states(), s), ntuple(_ -> states(), s),
            positions(), positions(),
            ntuple(_ -> positions(), s), ntuple(_ -> positions(), s),
            ib, work, s
        )
    end
end

const GeneralIMEXTimeStepper = GeneralRKTimeStepper

# Mixed source tuple → sorted
function GeneralRKTimeStepper(pde::HyperbolicPDE{D, M, T}, div_interp, mood::MOOD,
                              all_sources::Tuple{Vararg{AbstractSourceTerm}}, tableau::IMEXButcherTableau{T};
                              kwargs...) where {D, M, T}
    explicit_sts = filter(st -> st isa AbstractExplicitSourceTerm, all_sources)
    implicit_sts = filter(st -> st isa AbstractImplicitSourceTerm, all_sources)
    if !isempty(implicit_sts) && all(iszero, tableau.a)
        @warn "Implicit source terms given with a purely explicit tableau; the implicit solve is skipped (a[i,i] = 0)."
    end
    return GeneralRKTimeStepper(pde, div_interp, mood, explicit_sts, implicit_sts, tableau; kwargs...)
end

# No sources
GeneralRKTimeStepper(pde::HyperbolicPDE{D, M, T}, div_interp, mood::MOOD, tableau::IMEXButcherTableau{T}; kwargs...) where {D, M, T} =
    GeneralRKTimeStepper(pde, div_interp, mood, (), (), tableau; kwargs...)

"""
    mood_flags(ts::GeneralRKTimeStepper) -> AbstractVector{Bool}

Per-step diagnostic: `true` for every particle that dropped an order during the last step. A view of the
stepper's mask over the particles of that step (the buffer behind it is over-allocated).
"""
mood_flags(ts::GeneralRKTimeStepper) = view(parent(ts.work.triggered_step), 1:ts.work.N[])

@inline _is_moving(pg::ParticleGrid) = !(pg.mover isa NoGridMover)

function update_size!(ts::GeneralRKTimeStepper, N::Int, M_neighbors::Int, moving::Bool)
    ensure_capacity!(ts.rho_n, N)
    ensure_capacity!(ts.rho_cand, N)
    for i in 1:ts.num_stages
        ensure_capacity!(ts.Y_stages[i], N)
        ensure_capacity!(ts.K_E_stages[i], N)
        ensure_capacity!(ts.K_I_stages[i], N)
    end
    if moving
        ensure_capacity!(ts.x_n, N)
        ensure_capacity!(ts.x_cand, N)
        for i in 1:ts.num_stages
            ensure_capacity!(ts.X_stages[i], N)
            ensure_capacity!(ts.K_X_stages[i], N)
        end
    end
    update_size!(ts.int_buffer, M_neighbors)
    update_size!(ts.work, N)
    return nothing
end
update_size!(ts::GeneralRKTimeStepper, N::Int, M_neighbors::Int) = update_size!(ts, N, M_neighbors, true)

# =========================================================================
# BUILDING BLOCKS
# =========================================================================

"""
    compute_explicit_rhs!(ts, eq, pg, K_E, K_X, Y, t, full)

Scheme pre-pass (cached gradients) and flux pass at the CURRENT grid geometry:
`K_E = -div(F − w_i ρ) + S_expl` (the PDE wrapped in an `ALEFrame` with the particle's own grid
velocity), and `K_X = w` when the grid moves. Both passes read neighbour states straight from `Y`.
`full = true` covers all interior particles; otherwise gradients are recomputed on `work.changed`
and fluxes on `work.dirty`.
"""
function compute_explicit_rhs!(ts::GeneralRKTimeStepper{D, M, T}, eq, pg, K_E, K_X, Y, t, full::Bool) where {D, M, T}
    N       = pg.meta.N
    interp  = ts.divergence_interpolator
    moving  = _is_moving(pg)
    backend = get_backend()

    prepare_geometry!(interp, pg)   # refreshes geometry caches only if the grid changed

    foreach_particle(_rhs_gather!, backend, N,
        (Y, pg.core.is_boundary, pg.neighbor.ranges, ts.int_buffer, interp, pg, ts.work.changed, full))
    foreach_particle(_rhs_flux!, backend, N,
        (eq, Y, K_E, K_X, pg.core.is_boundary, pg.neighbor.ranges, ts.int_buffer, interp, ts.mood.strategy, pg.mover, pg,
         ts.explicit_sources, ts.work.dirty, full, moving, t, zero(State{M, T}), zero(Space{D, T})))
    return nothing
end

# Kernel bodies of the right-hand side (see Backend.jl): the scheme's pre-pass (cached gradients), then the flux pass.
@inline function _rhs_gather!(p, Y, is_boundary, nb_slices, ib, interp, pg, grad_mask, full)
    (is_boundary[p] || !(full || grad_mask[p])) && return nothing
    fi = Y[p]; nb = nb_slices[p]
    update_content!(interp, p, fi, nb, pg, Y, ib)
    return nothing
end

@inline function _rhs_flux!(p, eq, Y, K_E, K_X, is_boundary, nb_slices, ib, interp, strategy, mover, pg, sources,
                            flux_mask, full, moving, t, zero_state, zero_space)
    if is_boundary[p]
        if full
            K_E[p] = zero_state
            moving && (K_X[p] = zero_space)
        end
        return nothing
    end
    (full || flux_mask[p]) || return nothing
    fi = Y[p]; nb = nb_slices[p]
    if moving
        w      = grid_velocity(mover, eq, p, fi, pg, t)
        K_X[p] = w
        div_F  = interp(ALEFrame(eq, w), p, fi, nb, pg, Y, ib, strategy)   # relative flux F − w_i ρ
    else
        div_F  = interp(eq, p, fi, nb, pg, Y, ib, strategy)
    end
    K_E[p] = -div_F + evaluate_sources(sources, fi, p, pg, t)
    return nothing
end

"""
    form_candidate!(ts, eq, pg, i, dt, time) -> (target, t_i)

For `i ≤ s`: builds `Y_i` (explicit predictor, implicit diagonal solve, `K_I[i]`, BCs) and `X_i`.
For `i == s + 1`: builds the final state into `rho_cand` / `x_cand` (BCs applied).
Positions are NOT applied to the grid here; see `set_positions!`.
"""
function form_candidate!(ts::GeneralRKTimeStepper{D, M, T}, eq, pg, i::Int, dt::T, time::Real) where {D, M, T}
    N        = pg.meta.N
    s        = ts.num_stages
    bt       = ts.tableau
    moving   = _is_moving(pg)
    backend  = get_backend()

    is_final = i > s
    target   = is_final ? ts.rho_cand : ts.Y_stages[i]
    x_target = is_final ? ts.x_cand   : ts.X_stages[i]
    t_i      = is_final ? time + dt : time + bt.c_t[i] * dt
    n_prev   = is_final ? s : i - 1

    foreach_particle(_stage_predictor!, backend, N,
        (target, x_target, ts.rho_n, ts.x_n, ts.K_E_stages, ts.K_I_stages, ts.K_X_stages, pg.rhos,
         pg.core.is_boundary, bt, i, n_prev, is_final, dt, moving, zero(T), zero(Space{D, T})))

    if !is_final
        pre_solve_updates!(ts.implicit_sources, parent(target), pg, t_i)
        a_ii = bt.a[i, i]
        if abs(a_ii) > T(1e-14)
            foreach_particle(_stage_implicit_solve!, backend, N,
                (target, pg.core.is_boundary, ts.implicit_sources, dt * a_ii, pg, t_i))
        end
        foreach_particle(_stage_implicit_rhs!, backend, N,
            (ts.K_I_stages[i], target, pg.core.is_boundary, ts.implicit_sources, pg, t_i, zero(State{M, T})))
    end

    apply_boundary_conditions!(pg, parent(target), ts, eq, t_i)
    return target, t_i
end

# Kernel bodies of the stage assembly (see Backend.jl).
@inline function _stage_predictor!(p, target, x_target, U_n, x_n, K_E, K_I, K_X, rhos, is_boundary, bt, i, n_prev,
                                   is_final, dt, moving, zero_T, zero_space)
    if is_boundary[p]
        target[p] = rhos[p]
        moving && (x_target[p] = x_n[p])
        return nothing
    end
    Y = U_n[p]
    X = moving ? x_n[p] : zero_space
    for j in 1:n_prev
        a_t = is_final ? bt.b_t[j] : bt.a_t[i, j]
        a   = is_final ? bt.b[j]   : bt.a[i, j]
        if a_t != zero_T
            Y += (dt * a_t) * K_E[j][p]
            moving && (X += (dt * a_t) * K_X[j][p])
        end
        a != zero_T && (Y += (dt * a) * K_I[j][p])
    end
    target[p] = Y
    moving && (x_target[p] = X)
    return nothing
end

@inline function _stage_implicit_solve!(p, target, is_boundary, sources, dt_ii, pg, t_i)
    is_boundary[p] && return nothing
    target[p] = implicit_solve(sources, target[p], dt_ii, p, pg, t_i)
    return nothing
end

@inline function _stage_implicit_rhs!(p, K_I_i, target, is_boundary, sources, pg, t_i, zero_state)
    K_I_i[p] = is_boundary[p] ? zero_state : evaluate_sources(sources, target[p], p, pg, t_i)
    return nothing
end

"""
    mood_checkpoint!(ts, eq, pg, i, dt, time)

Forms the candidate for checkpoint `i` (`Y_i`, or the final state for `i == s+1`) and runs the
stage-wise MOOD fixed point against `Y_{i-1}`. The grid geometry must still be that of stage
`i-1` when this is called, because `K_E[i-1]` is recomputed here. Terminates because every
iteration strictly lowers at least one order.
"""
function mood_checkpoint!(ts::GeneralRKTimeStepper{D, M, T}, eq, pg, i::Int, dt::T, time::Real) where {D, M, T}
    interp = ts.divergence_interpolator
    work   = ts.work
    N      = pg.meta.N
    Y_prev = ts.Y_stages[i - 1]
    t_prev = time + ts.tableau.c_t[i - 1] * dt
    reach  = _order_reach(interp, ts.mood.strategy)

    full = true
    while true
        target, _ = form_candidate!(ts, eq, pg, i, dt, time)
        detect_and_drop!(ts.mood, work, interp, pg, Y_prev, target, full) || return nothing

        copyto!(parent(work.dirty), 1, parent(work.changed), 1, N)
        expand_mask!(parent(work.dirty), parent(work.scratch), pg, reach)
        compute_explicit_rhs!(ts, eq, pg, ts.K_E_stages[i - 1], ts.K_X_stages[i - 1], Y_prev, t_prev, false)
        full = false
    end
end

# Switch the grid to stage geometry X (frozen topology or full search)
function _apply_stage_geometry!(ts::GeneralRKTimeStepper, pg, X)
    set_positions!(pg, X, ts.full_search_per_stage)
    ts.full_search_per_stage && update_size!(ts.int_buffer, length(pg.neighbor.indices))
    return nothing
end

# =========================================================================
# TIME STEP
# =========================================================================

function (ts::GeneralRKTimeStepper{D, M, T})(eq::HyperbolicPDE, pg::ParticleGrid, time::Real, dt::T) where {D, M, T}
    N      = pg.meta.N
    s      = ts.num_stages
    interp = ts.divergence_interpolator
    orders = pg.core.particle_orders
    moving = _is_moving(pg)

    update_size!(ts, N, length(pg.neighbor.indices), moving)
    update_size!(interp, N)                       # once per step; gradients persist across checkpoints

    copyto!(parent(ts.rho_n), 1, pg.rhos, 1, N)
    moving && copyto!(parent(ts.x_n), 1, pg.core.positions, 1, N)

    max_order = _extract_order(interp)
    fill!(view(orders, 1:N), max_order)
    fill!(view(ts.work.triggered_step, 1:N), false)

    for i in 1:s
        if i == 1
            form_candidate!(ts, eq, pg, 1, dt, time)      # X_1 == x_n, grid already there
        else
            mood_checkpoint!(ts, eq, pg, i, dt, time)      # uses geometry of stage i-1
            moving && _apply_stage_geometry!(ts, pg, ts.X_stages[i])
        end
        ts.reset_orders_per_stage && fill!(view(orders, 1:N), max_order)
        compute_explicit_rhs!(ts, eq, pg, ts.K_E_stages[i], ts.K_X_stages[i], ts.Y_stages[i],
                              time + ts.tableau.c_t[i] * dt, true)
    end

    mood_checkpoint!(ts, eq, pg, s + 1, dt, time)          # uses geometry of stage s
    copyto!(pg.rhos, 1, parent(ts.rho_cand), 1, N)

    if moving
        copyto!(pg.core.positions, 1, parent(ts.x_cand), 1, N)
        pg.neighbor(pg)                                    # full search + periodic wrap for the next step
        ts.volume_estimator === nothing || update_volumes!(pg, ts.volume_estimator)
    end
    return nothing
end
