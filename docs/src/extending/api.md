```@meta
CurrentModule = HyperCloudCore
```

# API Contracts

HyperCloudCore is extended by subtyping one of its abstract types and implementing a few methods. This page lists, per extension point, what is required and what is optional, followed by the reference for every method in `API.jl`.

Import the methods you extend (`import HyperCloudCore: flux, max_eigenvalue, ...`) or qualify them (`HyperCloudCore.flux(...) = ...`).

## A new PDE (`HyperbolicPDE{D, M, T}`)

| Method | Required | Used by |
|---|---|---|
| `flux(eq, U) -> Flux{D,M,T}` | always | all schemes |
| `max_eigenvalue(eq, U, d) -> T` | always | Rusanov flux, CFL estimate |
| `velocity(eq, U, d) -> SMatrix{M,M,T}` | for Upwind (Tiwari, Praveen), WENO, `UpwindFlux` | advection speed / Jacobian |
| `evaluate_nc_jump(eq, f_L, f_R, dist) -> Flux` | for non-conservative PDEs | MUSCL, Upwind Classic |
| `fluid_velocity(eq, U) -> Space{D,T}` | for Lagrangian movers | `LagrangianMover`, `ScaledLagrangianMover` |
| `max_eigenvalue(eq, U, d, shift) -> T` | optional | exact relative wave speed on moving grids |
| `prim2cons`, `cons2prim` | optional | conversions (identity by default) |

Write PDE-dependent code against these methods rather than against fields of `eq`. On moving grids the schemes receive the PDE wrapped in an `ALEFrame`, and only the API methods are forwarded through that wrapper.

## A new source term

| Kind | Methods |
|---|---|
| `AbstractExplicitSourceTerm` | `evaluate_source(st, U, i, pg, t)` |
| `AbstractImplicitSourceTerm` | `evaluate_source(st, U, i, pg, t)`, `implicit_solve(st, U_in, γ, i, pg, t)`; optional `pre_solve_update!(st, Y, pg, t)` |

`implicit_solve` must return the solution $U$ of $U - \gamma S(U) = U_{\text{in}}$ for the particle `i`.

## A new spatial scheme (`DivergenceInterpolator`)

| Method | Required | Purpose |
|---|---|---|
| `(interp)(eq, i, f_i, nb_slice, pg, ib, strategy::MOODStrategy) -> State` | always | $\nabla\cdot F$ at particle `i` |
| `_extract_order(interp) -> Int` | always | maximum consistency order |
| `update_size!(interp, N)` | always (may be a no-op) | grow internal per-particle buffers |
| `update_content!(interp, i, f_i, nb_slice, pg, ib)` | optional | pre-gather pass (e.g. cache gradients) |
| `_divergence_degree(interp) -> Int` | optional | MLS degree for the CFL estimate (default: the order) |
| `_order_reach(interp, strategy) -> Int` | optional | hops over which fluxes depend on neighbours' orders (default 0) |
| `has_cached_gradients(interp)`, `cached_gradients(interp)` | optional | enables `MOODu2`'s curvature test |

Inside the functor:

- Read the particle's order from `pg.core.particle_orders[i]` (the consistency order) and lower it locally if the neighbourhood is too small.
- Neighbour states are in `ib.f[k]` and differences in `ib.df[k]` for `k ∈ nb_slice`; `pg.neighbor.indices[k]` is the neighbour's particle index.
- Write per-neighbour scratch only to slots `k ∈ nb_slice` of `ib` (they are disjoint between particles, which makes the passes thread-safe).
- If the scheme caches per-particle data in `update_content!`, `_order_reach` must cover how far that data is read: MUSCL reads neighbours' gradients, so its reach is at least 1.

Define the optional hooks **after** the scheme's type, next to the scheme; the time-stepping files only refer to the abstract type.

## A new MOOD criterion or strategy

See [Multi-Dimensional Optimal Order Detection (MOOD)](@ref) for the signatures of the criterion functor, `_get_interface_orders` and `trigger_halo!`.

## A new grid mover, boundary condition or weight function

- **Grid mover:** `grid_velocity(mover, eq, p, U, pg, t) -> Space{D,T}`; see [Moving Grids (ALE)](@ref).
- **Boundary condition:** functor `(bc)(pg, rhos_buffer, tag, ts, eq, t)`; see [Boundary Conditions](@ref).
- **Weight function:** functor on the squared distance plus `get_cutoff`; see [MLS Weight Functions](@ref).

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["API.jl"]
```
