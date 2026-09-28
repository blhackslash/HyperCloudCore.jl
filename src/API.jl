
export flux, max_eigenvalue, prim2cons, cons2prim, velocity, update_size!, update_content!, _extract_order, solve

# =========================================================================
# PDE API -> Must be set for every PDE
# =========================================================================
"""
    flux(eq::HyperbolicPDE{D, M, T, R}, U::State{M, T})

Computes the physical flux vector for a given state across all spatial dimensions. 

# Arguments
- `eq`: The physical equation system.
- `U::State{M, T}`: The local state vector of length `M` containing the conservative variables.

# Returns
- A `Flux{D, M, T}` object, which encapsulates the flux evaluations for all `D` dimensions.

# Implementation Example
For diagonal linear advection, this returns a dynamically sized tuple where the state is element-wise multiplied by the predefined advection velocity in each dimension (`eq.vel[d] .* U`).
"""
@inline flux(eq::HyperbolicPDE, u::State) = error("flux not implemented for $(typeof(eq))")

"""
    max_eigenvalue(eq::HyperbolicPDE{D, M, T, R}, U::State{M, T}, d::Int)

Calculates the maximum absolute wave speed of the system along a specific spatial axis.

# Arguments
- `eq`: The physical equation system.
- `U::State{M, T}`: The local conservative state.
- `d::Int`: The spatial dimension index (e.g., 1 for X, 2 for Y, 3 for Z).

# Returns
- A scalar of type `T` representing the highest observed eigenvalue for that dimension.

# Implementation Example
In a linear advection system, the maximum eigenvalue is simply the maximum absolute value within the pre-stored diagonal velocity vector for dimension `d` (`maximum(abs.(eq.vel[d]))`).
"""
@inline max_eigenvalue(eq::HyperbolicPDE, u::State, dim::Int) = error("max_eigenvalue not implemented for $(typeof(eq))")
# These can default to identity if the PDE doesn't use primitive forms
"""
    prim2cons(eq::HyperbolicPDE, U::State)

Converts a state vector from primitive to conservative variable representations. 

# Arguments
- `eq`: The physical equation system.
- `U` or `W`: The local `State{M, T}` vector to be transformed.

# Returns
- A transformed `State{M, T}` vector.

# Implementation Example
These functions safely default to an identity mapping (`U -> U`) if not explicitly overridden, which perfectly handles systems like linear advection that do not differentiate between primitive and conservative states.
"""
@inline prim2cons(eq::HyperbolicPDE, u::State) = u
"""
    cons2prim(eq::HyperbolicPDE, W::State)

Converts a state vector from conservative to primitive variable representations. 

# Arguments
- `eq`: The physical equation system.
- `U` or `W`: The local `State{M, T}` vector to be transformed.

# Returns
- A transformed `State{M, T}` vector.

# Implementation Example
These functions safely default to an identity mapping (`U -> U`) if not explicitly overridden, which perfectly handles systems like linear advection that do not differentiate between primitive and conservative states.
"""
@inline cons2prim(eq::HyperbolicPDE, u::State) = u

# Used by the Upwind Flux (for M=1) or Custom Grid Movers
"""
    velocity(eq::HyperbolicPDE{D, M, T, R}, U::State{M, T}, d::Int)

Evaluates the exact advective velocity matrix (the flux Jacobian) for a specified spatial dimension. This is strictly required for upwind flux evaluations or dynamic grid moving operations.

# Arguments
- `eq`: The physical equation system.
- `U::State{M, T}`: The local conservative state.
- `d::Int`: The spatial dimension index.

# Returns
- An `SMatrix{M, M, T, M*M}` representing the exact flux Jacobian. 

# Implementation Example
For a diagonal linear advection system, this method dynamically assembles an `M x M` statically sized matrix where the diagonal entries map to `eq.vel[d][i]` and all off-diagonal entries are strictly evaluated as `zero(T)`.
"""
@inline velocity(eq::HyperbolicPDE, u::State, d::Int) = error("velocity not implemented for $(typeof(eq))")

# =========================================================================
# INTERPOLATOR API -> Must be set for every DivergenceInterpolator
# =========================================================================
"""
    update_size!(div::DivergenceInterpolator, N_particles::Int)

Dynamically resizes internal evaluation buffers inside a divergence interpolator before a time step begins. 

# Arguments
- `div`: The selected spatial interpolator (e.g., MUSCL, WENO, Central).
- `N_particles::Int`: The current total number of active particles in the simulation domain.

# Returns
- Must return `nothing` or act as a zero-cost no-op for stateless interpolators.
"""
update_size!(div::DivergenceInterpolator, N_particles::Int) =  error("Size update of the buffers has to be set! Set no-op for stateless interpolators!")

"""
    update_content!(div::DivergenceInterpolator, nb_slice::UnitRange{Int}, pg::ParticleGrid, ib::InteractionBuffer)

Executes any pre-calculation passes required by specific interpolators before the primary flux loop executes.

# Arguments
- `div`: The divergence interpolator.
- `nb_slice`: A `UnitRange` pointing to the target particle's neighbors.
- `pg`: The active `ParticleGrid`.
- `ib`: The `InteractionBuffer` holding the locally extracted neighbor states.

# Returns
- Acts as a no-op for stateless interpolators. For stateful interpolators like MUSCL, this computes the raw gradients and applies slope limiters prior to the interface reconstruction.
"""
update_content!(div::DivergenceInterpolator, nb_slice, pg, ib) = error("Content update of the buffers has to be set! Set no-op for stateless interpolators! ")

"""
    _extract_order(div::DivergenceInterpolator)

Queries the underlying baseline polynomial or numerical order of the configured spatial scheme.

# Returns
- An `Int` representing the numerical order of the solver (e.g., 2 for a standard second-order MUSCL). This returned integer is critically required by the dynamic geometric CFL calculator to properly scale the stable time step.
"""
@inline _extract_order(div::DivergenceInterpolator) = error("Order of the method has to be defined for CFL calculation!")

export pre_solve_update!, evaluate_source, implicit_solve

# API 1: Called once per stage before the implicit solve (e.g., to update non-local potentials)
pre_solve_update!(st::AbstractSourceTerm, Y_stage, pg, t::Real) = nothing

# API 2: Evaluates the source term vector S(U)
function evaluate_source(st::AbstractSourceTerm, U, p_idx::Int, pg, t::Real)
    error("`evaluate_source` not implemented for $(typeof(st))")
end

# API 3: Executes the implicit solve: U - dt * a_ii * S(U) = U_in
function implicit_solve(solver::AbstractImplicitSolver, U_in, dt_coeff::Real, st::AbstractSourceTerm, p_idx::Int, pg, t::Real)
    error("`implicit_solve` not implemented for $(typeof(solver))")
end

export kinetic_wave_speed

"""
    kinetic_wave_speed(eq::HyperbolicPDE, d::Int, k::Int)
    
Returns the advection speed of the `k`-th kinetic component in the `d`-th spatial dimension.
Must be implemented by any PDE used as a kinetic relaxation system.
"""
function kinetic_wave_speed(eq::HyperbolicPDE, d::Int, k::Int)
    error("`kinetic_wave_speed` not implemented for $(typeof(eq)).")
end


