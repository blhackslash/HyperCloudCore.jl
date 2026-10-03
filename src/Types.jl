export MLSWeightFunction, AbstractBoundaryCondition, NoGridMover, GridMover
export AbstractSlopeLimiter, DivergenceInterpolator, NumericalFluxFunction
export HyperbolicPDE
export AbstractSourceTerm, AbstractExplicitSourceTerm, AbstractImplicitSourceTerm, NoExplicitSource, NoImplicitSource
export TimeStepper, MOODStrategy, MOODCriterion, Halo, MOOD, NoMOOD, NoStrategy

"""
    MLSWeightFunction

Base type for Moving Least Squares (MLS) kernel weight functions.

# API Requirements
- Must be callable as `(wf::MLSWeightFunction)(r::Real) -> Real`, returning the scaling weight for a given radial distance `r`.
- Should track its internal cutoff/interpolation range.
"""
abstract type MLSWeightFunction end

"""
    GridMover

Base type for Arbitrary Lagrangian-Eulerian (ALE) or dynamic grid adjustment strategies.
"""
abstract type GridMover end
struct NoGridMover <: GridMover end

"""
    AbstractBoundaryCondition

Base type for particle grid boundary conditions.

# API Requirements
- Evaluators must implement a dispatch handling ghost particle modifications, typically matching the internal signature required by `apply_boundary_conditions!`.
"""
abstract type AbstractBoundaryCondition end

"""
    NumericalFluxFunction

Base type for approximate Riemann solvers and numerical interface fluxes.

# API Requirements
- Must implement the functor: `(::NumericalFluxFunction)(f_L, f_R, F_L, F_R, eq::HyperbolicPDE)`
- Returns a `Flux{D, M, T}` representing the resolved numerical flux at the interface between the left and right reconstructed states.
"""
abstract type NumericalFluxFunction end

"""
    MOODStrategy

Base type dictating how polynomial order degradation cascades through the spatial neighborhood during Multi-Dimensional Optimal Order Detection (MOOD).

# API Requirements
- Must implement `trigger_halo!(strategy::MOODStrategy, p_idx, pg, needs_recalc, orders)` to propagate the degradation flag to neighboring particles.
"""
abstract type MOODStrategy end

struct NoStrategy <: MOODStrategy end

"""
    MOODCriterion

Base type for discrete maximum principles (DMP) and physical admissibility checks.

# API Requirements
- Must implement the functor: `(::MOODCriterion)(main_grad, p_idx, f_i, nb_slice, f_new, pg, f_all) -> Bool`
- Returns `true` if the locally updated state `f_new` violates the criterion, triggering an order drop.
"""
abstract type MOODCriterion end
struct NoMOOD <: MOODCriterion end

"""
    MOOD{S <: MOODStrategy, C <: MOODCriterion}

Wrapper coupling a `MOODCriterion` (the trigger) with a `MOODStrategy` (the response).
"""
struct MOOD{S <: MOODStrategy, C <: MOODCriterion}
    strategy::S
    criterion::C
end

# Default fallback for schemes without MOOD
MOOD() = MOOD(NoStrategy(), NoMOOD())

"""
    AbstractSlopeLimiter

Base type for spatial reconstruction limiters (e.g., Venkatakrishnan, MinMod).

# API Requirements
- Must implement `_limit_slopes(limiter::AbstractSlopeLimiter, raw_grad, nb_slice, f_i, f_all, pg, dist_all, max_degree)` returning a bounded gradient `SVector`.
"""
abstract type AbstractSlopeLimiter end

"""
    DivergenceInterpolator

Central component for executing spatial reconstruction and computing the flux divergence.

# API Requirements
- Must implement the primary execution functor: `(::DivergenceInterpolator)(eq::HyperbolicPDE, i::Int, f_i::State, nb_slice::UnitRange, pg::ParticleGrid, ib::InteractionBuffer) -> State{M, T}`. (Note: The return value is typically scaled by 2.0 depending on the internal integration rules).
- Must implement `update_size!(interp, N_particles)` for memory allocation.
- Must implement `update_content!(interp, i, f_i, nb_slice, pg, ib)` for pre-gather passes (e.g., caching gradients).
- Must implement `@inline _extract_order(interp) -> Int` to expose the maximum polynomial order.
"""
abstract type DivergenceInterpolator end

"""
    HyperbolicPDE{D, M, T}

The pure physical system of equations. 
Defines the spatial dimensions `D`, number of equations `M` and numeric type `T`.

# API Requirements
- `flux(eq::HyperbolicPDE, U::State)`: Returns the analytical `Flux{D, M, T}`.
- `max_eigenvalue(eq::HyperbolicPDE, U::State, d::Int)`: Returns the maximum local wave speed in dimension `d`.
- `velocity(eq::HyperbolicPDE, U::State, d::Int)`: Returns the `M x M` Jacobian/Velocity matrix.
# Optional API
- `prim2cons(eq, U)` / `cons2prim(eq, U)`: Conversion utilities between state representations.
"""
abstract type HyperbolicPDE{D, M, T} end

"""
    AbstractSourceTerm

Base type for RHS source terms (e.g., gravity, kinetic relaxation, reactions).
"""
abstract type AbstractSourceTerm end

"""
    AbstractExplicitSourceTerm <: AbstractSourceTerm

Source term evaluated explicitly during the explicit Runge-Kutta or IMEX explicit stages.

# API Requirements
- Must implement `evaluate_sources(st::AbstractExplicitSourceTerm, f_i, i, pg, t) -> State{M, T}`.
"""
abstract type AbstractExplicitSourceTerm <: AbstractSourceTerm end

"""
    AbstractImplicitSourceTerm <: AbstractSourceTerm

Source term evaluated implicitly via stiff solvers during IMEX integration.

# API Requirements
- Must implement `implicit_solve(st::AbstractImplicitSourceTerm, f_i, dt, i, pg, t) -> State{M, T}`.
- Must implement `pre_solve_updates!(st, current_Y_i, pg, t)` for global pre-computations.
"""
abstract type AbstractImplicitSourceTerm <: AbstractSourceTerm end

struct NoExplicitSource <: AbstractExplicitSourceTerm end
struct NoImplicitSource <: AbstractImplicitSourceTerm end

"""
    TimeStepper

The global orchestrator advancing the PDE over time. 

# API Requirements
- Must implement the primary execution functor: `(::TimeStepper)(eq::HyperbolicPDE, pg::ParticleGrid, t::Real, dt::Real)` which updates the grid in-place.
- Must manage intermediate allocations and buffer sizing internally via `update_size!`.
"""
abstract type TimeStepper end