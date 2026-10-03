export flux, max_eigenvalue, prim2cons, cons2prim, velocity
export evaluate_nc_jump
export pre_solve_update!, evaluate_source, implicit_solve
export update_size!, update_content!, _extract_order

# =========================================================================
# 1. PDE API (Physical Equations)
# =========================================================================

"""
    flux(eq::HyperbolicPDE{D, M, T, R}, U::State{M, T})

Computes the physical flux vector for a given state across all spatial dimensions. 

# Arguments
- `eq::HyperbolicPDE`: The physical equation system.
- `U::State{M, T}`: The local conservative state vector.

# Returns
- A `Flux{D, M, T}` object containing the flux evaluations for all `D` dimensions.
"""
@inline flux(eq::HyperbolicPDE, u::State) = error("`flux` not implemented for $(typeof(eq))")

"""
    max_eigenvalue(eq::HyperbolicPDE{D, M, T, R}, U::State{M, T}, d::Int)

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
    velocity(eq::HyperbolicPDE{D, M, T, R}, U::State{M, T}, d::Int)

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
    evaluate_nc_jump(eq::HyperbolicPDE, f_L, f_R, dist_k)

Evaluates the non-conservative path integral jump across an interface.

# Arguments
- `eq::HyperbolicPDE`: The physical equation system.
- `f_L::State`, `f_R::State`: The reconstructed left and right interface states.
- `dist_k::Space`: The spatial distance vector between the two points.

# Returns
- A `Flux{D, M, T}` representing the path-dependent non-conservative jump. Defaults to strictly zero for `Conservative` PDEs.
"""
@inline function evaluate_nc_jump(
    eq::HyperbolicPDE{D, M, T, Conservative}, f_L::Flux{D, M, T}, f_R::Flux{D, M, T}, dist_k::Space{D, T}
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
    update_content!(interp::DivergenceInterpolator, i::Int, f_i::State, nb_slice::UnitRange, pg::ParticleGrid, ib::InteractionBuffer)

Executes pre-gather computations (such as resolving MLS gradients or limiting slopes) prior to the primary flux pass. Defaults to a no-op for stateless algorithms.
"""
@inline update_content!(::DivergenceInterpolator, args...) = nothing

"""
    _extract_order(interp::DivergenceInterpolator) -> Int

Returns the maximum structural polynomial order of the configured spatial scheme.
"""
@inline _extract_order(::DivergenceInterpolator) = error("`_extract_order` not implemented!")


# =========================================================================
# 4. TIMESTEPPER API
# =========================================================================

"""
    update_size!(ts::TimeStepper, N_particles::Int, M_neighbors::Int)

Dynamically resizes all internal Runge-Kutta / IMEX stage buffers and interaction masks.
"""
@inline update_size!(::TimeStepper, ::Int, ::Int) = error("`update_size!` not implemented!")
