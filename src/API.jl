export flux, max_eigenvalue, is_admissible, prim2cons, cons2prim, velocity, fluid_velocity
export evaluate_nc_jump
export pre_solve_update!, evaluate_source, implicit_solve
export update_size!, update_content!, prepare_geometry!, _extract_order, _divergence_degree, _order_reach
export has_cached_gradients, cached_gradients

# =========================================================================
# 1. PDE API (Physical Equations)
# =========================================================================

"""
    flux(eq::HyperbolicPDE{D, M, T}, U::State{M, T})

Computes the physical flux vector for a given state across all spatial dimensions. 

# Arguments
- `eq::HyperbolicPDE`: The physical equation system.
- `U::State{M, T}`: The local conservative state vector.

# Returns
- A `Flux{D, M, T}` object containing the flux evaluations for all `D` dimensions.
"""
@inline flux(eq::HyperbolicPDE, u::State) = error("`flux` not implemented for $(typeof(eq))")

"""
    max_eigenvalue(eq::HyperbolicPDE{D, M, T}, U::State{M, T}, d::Int)

Calculates the maximum absolute wave speed of the system along a specific spatial axis.

# Arguments
- `eq::HyperbolicPDE`: The physical equation system.
- `U::State{M, T}`: The local conservative state.
- `d::Int`: The spatial dimension index.

# Returns
- A scalar of type `T` representing the highest observed eigenvalue for that dimension.
"""
@inline max_eigenvalue(eq::HyperbolicPDE, u::State, dim::Int) = error("`max_eigenvalue` not implemented for $(typeof(eq))")

"""
    is_admissible(eq::HyperbolicPDE, U::State) -> Bool

Physical admissibility of a state (the PAD of MOOD): a MOOD time stepper drops the order of a particle
whose candidate state is not admissible, whatever its criterion says. The default only rejects
non-finite states (NaN or Inf); systems with an admissible set (e.g. positive density and pressure for
the Euler equations) add a method. Runs inside kernels.
"""
@inline is_admissible(eq::HyperbolicPDE, U::State) = all(isfinite, U)

"""
    max_eigenvalue(eq::HyperbolicPDE, U::State, d::Int, shift::Real)

Maximum absolute eigenvalue of the flux Jacobian along `d` after shifting the spectrum by
`-shift` (used by `ALEFrame` on moving grids). The default `max_eigenvalue(eq, U, d) + |shift|`
is a safe bound; PDEs whose spectrum is known can override it with the exact `max |λ - shift|`
to recover the full dissipation reduction of a Lagrangian frame.
"""
@inline max_eigenvalue(eq::HyperbolicPDE, u::State, dim::Int, shift::Real) = max_eigenvalue(eq, u, dim) + abs(shift)

"""
    prim2cons(eq::HyperbolicPDE, U::State)

Converts a state vector from primitive to conservative variables. Defaults to identity mapping if not overridden.

# Arguments
- `eq::HyperbolicPDE`: The physical equation system.
- `U::State`: The primitive state vector.

# Returns
- The transformed conservative `State`.
"""
@inline prim2cons(eq::HyperbolicPDE, u::State) = u

"""
    cons2prim(eq::HyperbolicPDE, W::State)

Converts a state vector from conservative to primitive variables. Defaults to identity mapping if not overridden.

# Arguments
- `eq::HyperbolicPDE`: The physical equation system.
- `W::State`: The conservative state vector.

# Returns
- The transformed primitive `State`.
"""
@inline cons2prim(eq::HyperbolicPDE, u::State) = u

"""
    velocity(eq::HyperbolicPDE{D, M, T}, U::State{M, T}, d::Int)

Evaluates the exact advective velocity matrix (the flux Jacobian) for a specified spatial dimension. Required for Upwind flux algorithms.

# Arguments
- `eq::HyperbolicPDE`: The physical equation system.
- `U::State{M, T}`: The local conservative state.
- `d::Int`: The spatial dimension index.

# Returns
- An `SMatrix{M, M, T, M*M}` representing the exact flux Jacobian. 
"""
@inline velocity(eq::HyperbolicPDE, u::State, d::Int) = error("`velocity` not implemented for $(typeof(eq))")

"""
    fluid_velocity(eq::HyperbolicPDE{D, M, T}, U::State{M, T}) -> Space{D, T}

The material (transport) velocity of the state. Only required by Lagrangian-type grid movers
(`LagrangianMover`, `ScaledLagrangianMover`).
"""
@inline fluid_velocity(eq::HyperbolicPDE, u::State) = error("`fluid_velocity` not implemented for $(typeof(eq))")

"""
    evaluate_nc_jump(eq::HyperbolicPDE, f_L, f_R, dist_k)

Evaluates the non-conservative path integral jump across an interface. Has to be overwritten explicitly for non-conservative PDEs.

# Arguments
- `eq::HyperbolicPDE`: The physical equation system.
- `f_L::State`, `f_R::State`: The reconstructed left and right interface states.
- `dist_k::Space`: The spatial distance vector between the two points.

# Returns
- A `Flux{D, M, T}` representing the path-dependent non-conservative jump. Defaults to strictly zero.
"""
@inline function evaluate_nc_jump(
    eq::HyperbolicPDE{D, M, T,}, f_L::Flux{D, M, T}, f_R::Flux{D, M, T}, dist_k::Space{D, T}
) where {D, M, T}
    return Flux{D, M, T}(ntuple(Val(D)) do d
        zero(State{M, T})
    end)
end


# =========================================================================
# 2. SOURCE TERM API
# =========================================================================

"""
    evaluate_source(st::AbstractSourceTerm, U::State, p_idx::Int, pg::ParticleGrid, t::Real)

Evaluates volumetric source terms (both explicit and implicit).

# Arguments
- `st::AbstractSourceTerm`: The specific source term to evaluate.
- `U::State`: The local state vector.
- `p_idx::Int`: The index of the current particle.
- `pg::ParticleGrid`: The global particle grid.
- `t::Real`: The current physical time (or stage time).

# Returns
- A `State{M, T}` containing the computed source components. Defaults to zero if the source term is empty.
"""
@inline evaluate_source(::AbstractExplicitSourceTerm, U, p_idx, pg, t) = error("`evaluate_source` not implemented!")
@inline evaluate_source(::NoExplicitSource, U, p_idx, pg, t) = zero(U)

@inline evaluate_source(::AbstractImplicitSourceTerm, U, p_idx, pg, t) = error("`evaluate_source` not implemented!")
@inline evaluate_source(::NoImplicitSource, U, p_idx, pg, t) = zero(U)

"""
    pre_solve_update!(st::AbstractImplicitSourceTerm, Y_stage, pg::ParticleGrid, t::Real)

Executes global state updates or reductions prior to the implicit solve step. Defaults to a no-op.
"""
pre_solve_update!(::AbstractImplicitSourceTerm, Y_stage, pg, t::Real) = nothing

"""
    implicit_solve(st::AbstractImplicitSourceTerm, U_in::State, dt_coeff::Real, p_idx::Int, pg::ParticleGrid, t::Real)

Executes the local implicit solve: `U_out - dt_coeff * S(U_out) = U_in`.

# Arguments
- `st::AbstractImplicitSourceTerm`: The implicit source term configuration.
- `U_in::State`: The accumulated explicit/implicit state prior to the solve.
- `dt_coeff::Real`: The stage-specific time step multiplier (e.g., `dt * a_ii`).
- `p_idx::Int`: The local particle index.
- `pg::ParticleGrid`: The global particle grid.
- `t::Real`: The current stage time.

# Returns
- The strictly solved `State{M, T}` (`U_out`). Defaults to `U_in` for empty sources.
"""
@inline implicit_solve(::AbstractImplicitSourceTerm, U_in, dt_coeff::Real, p_idx::Int, pg, t::Real) = error("`implicit_solve` not implemented!")
@inline implicit_solve(::NoImplicitSource, U_in, dt_coeff::Real, p_idx::Int, pg, t::Real) = U_in


# =========================================================================
# 3. DIVERGENCE INTERPOLATOR API
# =========================================================================

"""
    update_size!(interp::DivergenceInterpolator, N_particles::Int)

Dynamically resizes internal evaluation buffers (like cached gradients) to match the particle grid capacity.
"""
@inline update_size!(::DivergenceInterpolator, ::Int) = error("`update_size!` not implemented!")

"""
    update_content!(interp::DivergenceInterpolator, i::Int, f_i::State, nb_slice::OrdinalRange, pg::ParticleGrid, U::AbstractVector{State}, ib::InteractionBuffer)

Executes the pre-pass computations (such as resolving MLS gradients or limiting slopes) prior to the
primary flux pass; `U` is the stage state vector. Defaults to a no-op for stateless algorithms.
"""
@inline update_content!(::DivergenceInterpolator, args...) = nothing

"""
    prepare_geometry!(interp::DivergenceInterpolator, pg::ParticleGrid)

Called by the time stepper before every right-hand-side pass. Schemes that cache geometry-dependent data
(MUSCL caches its inverse MLS moment matrices) refresh it here when `pg.meta.geometry_version` has moved on
since the last call. Defaults to a no-op.
"""
@inline prepare_geometry!(::DivergenceInterpolator, pg) = nothing

"""
    _extract_order(interp::DivergenceInterpolator) -> Int

Returns the maximum structural polynomial order of the configured spatial scheme.
"""
@inline _extract_order(::DivergenceInterpolator) = error("`_extract_order` not implemented!")

"""
    _divergence_degree(interp::DivergenceInterpolator) -> Int

Polynomial degree of the MLS fit used for the divergence at the maximum order (CFL estimate).
Defaults to the consistency order; MUSCL overrides with `order - 1`.
"""
@inline _divergence_degree(g::DivergenceInterpolator) = _extract_order(g)

"""
    _order_reach(interp::DivergenceInterpolator, strategy::MOODStrategy) -> Int

Hops to which the flux of a particle depends on neighbours' *orders*. Drives the MOOD recompute
set `C ⊕ reach`. Defaults to 0; MUSCL overrides (1, or 2 for `EPD2`).
"""
@inline _order_reach(::DivergenceInterpolator, ::MOODStrategy) = 0

"""
    has_cached_gradients(interp) -> Bool
    cached_gradients(interp) -> AbstractVector{SVector{B_LEN, State}}

Trait for schemes that keep per-particle reconstruction coefficients between the pre-gather and
flux passes (MUSCL). `MOODu2` uses the second-order coefficients `D+1 … 2D` for its curvature test.
"""
@inline has_cached_gradients(::DivergenceInterpolator) = false
cached_gradients(g::DivergenceInterpolator) = error("`cached_gradients` not implemented for $(typeof(g))")


# =========================================================================
# 3b. GRID MOVER API
# =========================================================================

"""
    grid_velocity(mover::GridMover, eq, p_idx, U, pg, t) -> Space{D, T}

Velocity of particle `p_idx` under the mover, evaluated at the current stage geometry and state.
`NoGridMover` returns zero and disables the moving-grid machinery at compile time.
"""
function grid_velocity end


# =========================================================================
# 4. TIMESTEPPER API
# =========================================================================

"""
    update_size!(ts::TimeStepper, N_particles::Int, M_neighbors::Int)

Dynamically resizes all internal Runge-Kutta / IMEX stage buffers and interaction masks.
"""
@inline update_size!(::TimeStepper, ::Int, ::Int) = error("`update_size!` not implemented!")
