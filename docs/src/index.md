```@meta
CurrentModule = HyperCloudCore
```

# HyperCloudCore: A Meshfree Hyperbolic Solver

HyperCloudCore is a meshfree collocation framework for hyperbolic systems

```math
\partial_t U + \nabla \cdot F(U) = S(U), \qquad U \in \mathbb{R}^M,\; x \in \mathbb{R}^D .
```

The solution lives on a cloud of particles. Spatial derivatives come from Moving Least Squares (MLS) fits over each particle's neighbourhood, time integration uses one general explicit/IMEX Runge–Kutta stepper, and robustness near discontinuities comes from *a priori* slope limiters, *a posteriori* order reduction (MOOD), or both. Particles can stay fixed or move with an arbitrary grid velocity (ALE), and the moving grid is integrated by the same tableau as the state.

The package is a backend: it provides the numerical machinery and expects you to bring the PDE (and, optionally, source terms, boundary conditions and grid movers) by implementing a small set of methods. See [API Contracts](@ref) for the full list.

[HyperCloud.jl](@extref HyperCloud HyperCloud.jl) is the physics front end on top of it: it adds equations (linear advection, Burgers, Euler), initial conditions, kinetic relaxation and exact solutions, and runs whole simulations from a flat parameter dictionary through the [config-driven builders](@ref "Config-Driven Builders") of this package. Use HyperCloud to run simulations, and this package directly to write a scheme or an equation of your own or to embed the solver in another driver.

## Quick start

A 1-D periodic advection problem, with a third-order MUSCL scheme, MOOD, and SSP-RK3:

```julia
using HyperCloudCore, StaticArrays
import HyperCloudCore: flux, max_eigenvalue, velocity

# 1. The PDE: ∂t u + ∂x (c u) = 0
struct Advection1D{T} <: HyperbolicPDE{1, 1, T}
    c::T
end
flux(eq::Advection1D{T}, u::State) where {T}       = Flux{1, 1, T}((eq.c * u,))
max_eigenvalue(eq::Advection1D, u::State, d::Int)  = abs(eq.c)
velocity(eq::Advection1D{T}, u::State, d::Int) where {T} = SMatrix{1, 1, T}(eq.c)

# 2. Geometry and particles
T    = Float64
dx   = (0.01,)
geom = get_rectangular_domain(T, (-1.0,), (1.0,); is_periodic = true)
pg   = ParticleGrid(geom, dx, ExponentialWeightFunction(1.0, 2.5 * dx[1]), 1)

for i in 1:pg.meta.N
    x = pg.core.positions[i][1]
    pg.rhos[i] = State{1, T}(exp(-x^2 / 0.01))
end

# 3. Spatial scheme, MOOD and time stepper
eq     = Advection1D(1.0)
scheme = MUSCL(T, 1, 1, 3, NoLimiter(), RusanovFlux())
mood   = MOOD(EPD1(), MOODu2(1e-4 * dx[1]))
ts     = GeneralRKTimeStepper(eq, scheme, mood, RK3_SSP_Tableau(T))

# 4. Time loop
t, tmax, cfl = zero(T), T(1), T(0.4)
while t < tmax
    dt = min(cfl * get_time_step(pg, eq, ts.divergence_interpolator), tmax - t)
    ts(eq, pg, t, dt)
    t += dt
end
```

To move the particles with the flow instead, construct the grid with `ParticleGrid(...; mover = LagrangianMover())` and define `fluid_velocity` for your PDE. Nothing else changes; see [Moving Grids (ALE)](@ref).

## How the pieces fit together

| Layer | Main types | Responsibility |
|---|---|---|
| Geometry | `GeometricDomain` | Shape, interior test, boundary tags, boundary-condition map |
| Topology | `ParticleGrid` | Particles, neighbour lists, MLS weights, per-particle order, grid mover |
| Spatial operator | `DivergenceInterpolator` (`CentralDivergence`, `UpwindDivergence`, `MUSCL`, `WENO`) | $\nabla\cdot F$ at one particle from its neighbourhood |
| Interface fluxes | `NumericalFluxFunction` (`RusanovFlux`, `UpwindFlux`) | Riemann problem between reconstructed states |
| Time integration | `GeneralRKTimeStepper`, `IMEXButcherTableau` | Stages, implicit sources, boundary conditions, grid motion |
| Robustness | `AbstractSlopeLimiter`, `MOOD` | *A priori* limiting and *a posteriori* order reduction |

A call `ts(eq, pg, t, dt)` advances one step. Per stage, it forms the stage state (explicit predictor, then the implicit diagonal solve, then boundary conditions), runs the MOOD check against the previous stage, switches the grid to the stage positions if the grid moves, and evaluates the explicit right-hand side. At the end of the step it writes the new state and positions back into the grid and rebuilds the neighbour lists.

## Conventions used throughout

- **Consistency order.** Every scheme is configured, and every particle's order in `pg.core.particle_orders` is stored, as a *consistency order* $k$, i.e. the expected convergence rate. Central and WENO use an MLS fit of polynomial degree $k$. MUSCL uses a reconstruction of degree $k-1$. Upwind is fixed at $k = 1$.
- **Sign of the right-hand side.** The stepper integrates $\mathrm{d}U/\mathrm{d}t = K_E + K_I$ with $K_E = -\nabla\cdot F + S_E$ and $K_I = S_I$. A divergence interpolator returns $+\nabla\cdot F$.
- **Element type.** The time step `dt` must have the grid's element type `T`; a `Float64` step on a `Float32` grid is a `MethodError` rather than a silent promotion.
- **Parallel loops.** All particle loops go through `foreach_particle` and run on the backend chosen with `set_backend!`: Polyester's `@batch` (default), `Threads.@threads`, or a KernelAbstractions backend (see [Execution backends](gpu_roadmap.md)).

## Package layout and load order

The time-integration files only refer to abstract types, so they can be loaded before the spatial schemes. A scheme that wants to customise stepper behaviour (MOOD locality, CFL degree, cached gradients) defines those hook methods *after* its own type, next to the scheme. MUSCL's hooks, for example, sit at the end of `MUSCL.jl`. A method signature that names a concrete type has to come after that type is defined; a reference inside a function body does not.
