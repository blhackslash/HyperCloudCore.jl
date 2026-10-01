
export flux, max_eigenvalue, prim2cons, cons2prim, velocity

export pre_solve_update!, evaluate_source, evaluate_sources, implicit_solve

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

Evaluates the exact advective velocity matrix (the flux Jacobian) for a specified spatial dimension. This is strictly required for upwind flux evaluations.

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
# EXPLICIT SOURCE TERM API
# =========================================================================

"""
    evaluate_source(st::AbstractExplicitSourceTerm, U, p_idx::Int, pg::ParticleGrid, t::Real)

Evaluates explicit volumetric source terms. Returns a State vector.
"""
@inline evaluate_source(::AbstractExplicitSourceTerm, U, p_idx, pg, t) = error("`evaluate_source` not implemented!")
@inline evaluate_source(::NoExplicitSource, U, p_idx, pg, t) = zero(U)

# =========================================================================
# IMPLICIT SOURCE TERM API
# =========================================================================

"""
    pre_solve_update!(st::AbstractImplicitSourceTerm, Y_stage, pg::ParticleGrid, t::Real)

Called once per RK/IMEX stage before the implicit solve. Useful for updating global potentials.
"""
pre_solve_update!(::AbstractImplicitSourceTerm, Y_stage, pg, t::Real) = nothing

"""
    evaluate_source(st::AbstractImplicitSourceTerm, U, p_idx::Int, pg::ParticleGrid, t::Real)

Evaluates the implicit source term for explicit assembly in the IMEX tableau (K_I).
"""
@inline evaluate_source(::AbstractImplicitSourceTerm, U, p_idx, pg, t) = error("`evaluate_source` not implemented!")
@inline evaluate_source(::NoImplicitSource, U, p_idx, pg, t) = zero(U)

"""
    implicit_solve(st::AbstractImplicitSourceTerm, U_in, dt_coeff::Real, p_idx::Int, pg::ParticleGrid, t::Real)

Executes the implicit solve: U_out - dt_coeff * S(U_out) = U_in.
"""
@inline implicit_solve(::AbstractImplicitSourceTerm, U_in, dt_coeff::Real, p_idx::Int, pg, t::Real) = error("`implicit_solve` not implemented!")
@inline implicit_solve(::NoImplicitSource, U_in, dt_coeff::Real, p_idx::Int, pg, t::Real) = U_in

# =========================================================================
# ZERO-COST TUPLE UNROLLERS (For stacking multiple source terms)
# =========================================================================

# Automatically sums the evaluated states of all source terms in a tuple
@inline @generated function evaluate_sources(sts::Tuple{Vararg{AbstractSourceTerm}}, U, p_idx::Int, pg, t::Real)
    N = length(sts.parameters)
    
    if N == 0
        return :(zero(U))
    elseif N == 1
        return :(evaluate_source(sts[1], U, p_idx, pg, t))
    else
        # Iteratively build the AST: S1 + S2 + ... + SN
        expr = :(evaluate_source(sts[1], U, p_idx, pg, t))
        for i in 2:N
            expr = :($expr + evaluate_source(sts[$i], U, p_idx, pg, t))
        end
        return expr
    end
end

# Chains pre-solve updates sequentially
@inline function pre_solve_updates!(sts::Tuple{Vararg{AbstractImplicitSourceTerm}}, Y_stage, pg, t::Real)
    for st in sts
        pre_solve_update!(st, Y_stage, pg, t)
    end
end

# Operator Splitting: Chains implicit solves sequentially (Lie-Trotter splitting)
@inline @generated function implicit_solve(sts::Tuple{Vararg{AbstractImplicitSourceTerm}}, U_in, dt_coeff::Real, p_idx::Int, pg, t::Real)
    N = length(sts.parameters)
    if N == 0; return :(U_in); end
    quote
        U_out = U_in
        Base.Cartesian.@nexprs $N i -> U_out = implicit_solve(sts[i], U_out, dt_coeff, p_idx, pg, t)
        return U_out
    end
end



