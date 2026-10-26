# Time Stepper

`GeneralRKTimeStepper` is the only time integrator. Explicit Runge–Kutta methods and implicit–explicit (IMEX) methods share it, because an explicit tableau is an IMEX tableau with a zero implicit part. `GeneralIMEXTimeStepper` is an alias for the same type.

## Semi-discrete system

For every interior particle $i$ the stepper integrates

```math
\frac{\mathrm d U_i}{\mathrm d t} = K_E(U)_i + K_I(U_i), \qquad
K_E = -\nabla\cdot F + S_E, \qquad K_I = S_I ,
```

where $\nabla\cdot F$ comes from the divergence interpolator, $S_E$ from the explicit source terms (non-stiff, evaluated at stage states), and $S_I$ from the implicit source terms (stiff, local to each particle). On a moving grid the positions are integrated as well, and $\nabla\cdot F$ is replaced by the divergence of the relative flux; see [Moving Grids (ALE)](@ref).

With an IMEX tableau $(\tilde A, \tilde b, \tilde c)$ (explicit) and $(A, b, c)$ (implicit, lower triangular), stage $k$ is

```math
Y_k = U^n + \Delta t \sum_{j<k} \tilde a_{kj}\, K_E(Y_j) + \Delta t \sum_{j<k} a_{kj}\, K_I(Y_j) + \Delta t\, a_{kk}\, K_I(Y_k),
```

and the new state is

```math
U^{n+1} = U^n + \Delta t \sum_{j} \tilde b_j\, K_E(Y_j) + \Delta t \sum_{j} b_j\, K_I(Y_j).
```

The implicit relation in $Y_k$ is local to each particle and is solved by the source term's `implicit_solve`. For an explicit tableau ($A = 0$, $b = 0$) the implicit sums vanish and the scheme reduces to classical explicit RK.

## Construction

```julia
GeneralRKTimeStepper(pde, interp, mood, tableau; kwargs...)                    # no sources
GeneralRKTimeStepper(pde, interp, mood, sources::Tuple, tableau; kwargs...)    # mixed tuple, sorted automatically
GeneralRKTimeStepper(pde, interp, mood, explicit::Tuple, implicit::Tuple, tableau; kwargs...)
```

- `interp` is a `DivergenceInterpolator`, and `mood` a `MOOD` object; `MOOD()` disables order reduction.
- `tableau` is an `IMEXButcherTableau`; every built-in explicit tableau already returns one (see [Butcher Tableaus](@ref)).
- A mixed `sources` tuple is split into `AbstractExplicitSourceTerm`s and `AbstractImplicitSourceTerm`s. Implicit sources with a purely explicit tableau trigger a warning, because the implicit solve is skipped when $a_{kk} = 0$.

Keyword options:

| Keyword | Default | Effect |
|---|---|---|
| `reset_orders_per_stage` | `false` | Restore the maximum order before every stage's RHS pass. By default, orders lowered by MOOD stay lowered for the rest of the step. |
| `full_search_per_stage` | `false` | On moving grids, rebuild the neighbour topology at every stage position instead of only refreshing distances and weights. |
| `volume_estimator` | `ShepardVolumes()` | How `pg.core.volumes` is recomputed after each step of a moving grid; `nothing` disables it. |

All buffers (per-stage states $Y_k$ and right-hand sides $K_{E,k}$, $K_{I,k}$, position stages, neighbour gather buffer, MOOD scratch) are owned by the stepper and grown on demand, so one stepper can be reused for any number of steps and grid sizes.

## One time step

`ts(eq, pg, t, dt::T)` performs:

1. Copy $U^n$ (and the positions $x^n$ on a moving grid). Reset every particle's order to the scheme's maximum.
2. For each stage $k = 1, \dots, s$:
   1. **Form $Y_k$:** explicit predictor, `pre_solve_update!` for each implicit source, the diagonal implicit solve if $a_{kk} \ne 0$, then $K_{I,k}$, then boundary conditions. With a single implicit source $K_{I,k} = (Y_k - Y_k^{\text{pred}}) / (\Delta t\, a_{kk})$ follows from the solve itself (no second evaluation of $S_I$); with several sources, or when $a_{kk} = 0$, $S_I(Y_k)$ is evaluated. On a moving grid, also form $X_k = x^n + \Delta t \sum_{j<k} \tilde a_{kj} K_{X,j}$.
   2. **MOOD checkpoint** ($k \ge 2$): test $Y_k$ against the neighbourhood of $Y_{k-1}$. Wherever it fails, lower orders, recompute $K_{E,k-1}$ only where it changed, re-form $Y_k$, and repeat until nothing fails. Details are on the [MOOD page](@ref "Multi-Dimensional Optimal Order Detection (MOOD)").
   3. **Move** the grid to $X_k$ (moving grids only).
   4. **Explicit RHS:** pre-gather pass (`update_content!`), then flux pass, giving $K_{E,k}$ and, on a moving grid, the grid velocities $K_{X,k}$.
3. **Final checkpoint:** form $U^{n+1}$ (and $x^{n+1}$), apply boundary conditions, and run the MOOD fixed point against $Y_s$.
4. Write $U^{n+1}$ to `pg.rhos`. On a moving grid, write $x^{n+1}$, run a full neighbour search and recompute the particle volumes.

Stage 1 has no MOOD check: $Y_1$ contains no explicit flux contribution yet, so no order could change it.

The element type of `dt` must be the grid's `T`.

## Source term API

| Method | For | Purpose |
|---|---|---|
| `evaluate_source(st, U, i, pg, t)` | explicit and implicit | $S(U)$ at particle `i` |
| `implicit_solve(st, U_in, γ, i, pg, t)` | implicit | Solve $U - \gamma\, S(U) = U_{\text{in}}$, with $\gamma = \Delta t\, a_{kk}$ |
| `pre_solve_update!(st, Y, pg, t)` | implicit, optional | Global precomputation on the predicted stage (e.g. non-local potentials) before the per-particle solves |

Several implicit sources are applied one after another in `implicit_solve` (a splitting within the stage). If they interact strongly, combine them into one source term.

## Running a simulation

The package does not prescribe a driver loop. A minimal one:

```julia
t = zero(T)
while t < tmax
    dt = min(cfl * get_time_step(pg, eq, ts.divergence_interpolator), tmax - t)
    ts(eq, pg, t, dt)
    t += dt
end
```

`get_time_step` uses the grid geometry of the current step. On a moving grid, call it every step, since the geometry changes. To keep a Lagrangian cloud well-spaced, call `adapt_particles!(pg, adaptivity, _extract_order(interp))` after each step; see [Particle Volumes and Adaptivity](@ref).

## Reference

```@docs
GeneralRKTimeStepper
mood_flags
```
