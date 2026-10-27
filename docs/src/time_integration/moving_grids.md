# Moving Grids (ALE)

Particles can move with an arbitrary grid velocity $w(x, U, t)$: fixed (Eulerian), with the flow (Lagrangian), or anything in between. The motion is part of the same Runge–Kutta step as the state, so no operator splitting is involved, and the coupled scheme has the order of the tableau.

## Formulation

For a particle moving with velocity $w_i$, the rate of change of $U$ along its path is

```math
\frac{\mathrm d U_i}{\mathrm d t} = \partial_t U + w_i \cdot \nabla U = -\nabla\cdot F(U) + w_i\cdot\nabla U + S .
```

With $w_i$ frozen at the particle (a constant within the stencil), $w_i\cdot\nabla U = \nabla\cdot(w_i U)$, so

```math
\frac{\mathrm d x_i}{\mathrm d t} = w_i, \qquad
\frac{\mathrm d U_i}{\mathrm d t} = -\nabla\cdot\big(F(U) - w_i\, U\big)\Big|_{x_i} + S(U_i).
```

The second equation is the original PDE seen from a frame moving with $w_i$. HyperCloudCore implements it by wrapping the PDE in an **`ALEFrame(eq, w_i)`** before calling the divergence interpolator:

| Method | `ALEFrame(eq, w)` |
|---|---|
| `flux(a, U)` | $F_d(U) - w_d\, U$ |
| `velocity(a, U, d)` | $\partial F_d/\partial U - w_d I$ |
| `max_eigenvalue(a, U, d)` | `max_eigenvalue(eq, U, d, w_d)` = $\max \lvert\lambda_d - w_d\rvert$ |
| `evaluate_nc_jump`, `prim2cons`, `cons2prim` | forwarded to `eq` |

Every scheme therefore reconstructs, upwinds and dissipates with respect to the *relative* velocity, without knowing that the grid moves. For $w = u$ in linear advection the relative flux vanishes identically and the state is carried exactly; the test suite checks this for every scheme.

Positions use only the explicit part of the tableau: $X_k = x^n + \Delta t \sum_{j<k} \tilde a_{kj} K_{X,j}$ with $K_{X,j} = w(X_j, Y_j)$.

## Movers

| Mover | Grid velocity | Requires |
|---|---|---|
| `NoGridMover()` | $0$ (static grid; the moving-grid code paths are compiled away) | – |
| `LagrangianMover()` | `fluid_velocity(eq, U)` | `fluid_velocity` for the PDE |
| `ScaledLagrangianMover(α)` | $\alpha\,$`fluid_velocity(eq, U)` | `fluid_velocity` |
| `PrescribedMover(f)` | `f(x::Space, t)` | – |

Pass the mover when building the grid: `ParticleGrid(geom, dx, wf, M; mover = LagrangianMover())`. Boundary (ghost) particles do not move.

A custom mover subtypes `GridMover` and implements

```julia
grid_velocity(::MyMover, eq, p, U, pg, t) -> Space{D, T}
```

evaluated at the current stage positions (`pg.core.positions`) and stage state `U`. Mesh-quality corrections, such as a repulsion term that keeps Lagrangian particles from clustering, belong here.

## Geometry during a step

Within a step the neighbour **topology is frozen**: each particle keeps the neighbour list from the start of the step. At every stage, `set_positions!` writes the stage positions and `refresh_geometry!` recomputes distances (minimum image) and weights. Pairs that move beyond the cutoff get weight zero; pairs that come within the cutoff during the step are picked up by the full neighbour search at the end of the step. This is the usual neighbour-skin assumption, and it holds when particles move only a fraction of the cutoff per step. With `full_search_per_stage = true` the stepper instead runs a full search at every stage.

The MOOD checkpoint for stage $k$ recomputes $K_{E,k-1}$, so the geometry is switched to $X_k$ only after that checkpoint has finished.

## PDE API for moving grids

```julia
fluid_velocity(eq, U) -> Space{D, T}                       # Lagrangian movers only
max_eigenvalue(eq, U, d, shift) -> T                       # optional, recommended
```

The default of the four-argument `max_eigenvalue` is the safe bound $\max|\lambda_d| + |w_d|$. That keeps the scheme stable but gives it more dissipation than the relative speed calls for. Override it with the exact $\max_k |\lambda_{d,k} - w_d|$ to obtain the full benefit of a Lagrangian frame; this matters for Rusanov fluxes in particular.

## Limitations

- **Time step.** `get_time_step` uses the absolute wave speeds $|\lambda|$, not $|\lambda - w|$. For co-moving grids this is conservative. If the grid moves *against* the waves, the relative speed is larger, and the CFL number has to be reduced accordingly. The particle displacement $|w|\,\Delta t$ should also stay well below the cutoff radius.
- **Volumes** in `pg.core.volumes` are recomputed after every step from the new geometry with the stepper's `volume_estimator`; see [Particle Volumes and Adaptivity](@ref).
- **Conservation.** Like the static collocation scheme, the ALE scheme is not exactly conservative; mass errors converge with the scheme's order. Particle insertion and merging between steps (`adapt_particles!`) *are* exactly conservative with respect to `total_mass`, also for calls with several events and, with the adaptivity's `volume_estimator` set to the stepper's (the builders do this), across the following volume re-estimation. Mass on a moving grid is only a convergent quantity with `MLSVolumes`; `ShepardVolumes` have a resolution-independent bias on stretched clouds.
- **Particle distribution.** Lagrangian motion clusters particles in compressions. Use `ParticleAdaptivity` between steps, and/or a mover that adds a smoothing velocity.
- **Reordering** has to happen between steps, as for static grids.

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["GridMovers.jl"]
Private = false
```
