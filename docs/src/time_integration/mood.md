# Multi-Dimensional Optimal Order Detection (MOOD)

MOOD is an *a posteriori* order reduction. A stage is first computed at the highest order. Every particle whose result is inadmissible then has its order lowered, and only the affected part of the stage is recomputed. Smooth regions keep full order, and discontinuities fall back towards first order, without the blanket dissipation of an *a priori* limiter.

```julia
MOOD()                          # disabled
MOOD(criterion)                 # strategy EPD1()
MOOD(strategy, criterion)
```

## Orders

Each particle carries a **consistency order** $k_i \in \{1, \dots, k_{\max}\}$ in `pg.core.particle_orders`, where $k_{\max}$ is `_extract_order(interp)`. All schemes read it with the same meaning (Central/WENO: fit degree $k$; MUSCL: reconstruction degree $k - 1$). Orders are reset to $k_{\max}$ at the start of every time step. They can also be lowered outside MOOD: MUSCL sets a particle with too few neighbours for its reconstruction to order 1.

## Admissibility criteria (`MOODCriterion`)

Every criterion compares the candidate value $U_i^\ast$ with the neighbourhood of the previous stage, $\{U_j\}_{j \in \nu(i) \cup \{i\}}$.

**`MOODu1(δ)`: relaxed discrete maximum principle.** For each component $m$, a particle fails if

```math
\max_j U_{j,m} - \min_j U_{j,m} \ge \delta^3
\quad\text{and}\quad
U^\ast_{i,m} \notin \left[\min_j U_{j,m} - \delta,\; \max_j U_{j,m} + \delta\right].
```

The first condition ignores flat regions, where round-off would otherwise trigger. The relaxation $\delta$ absorbs small, harmless overshoots. Scale $\delta$ with the resolution (e.g. $\delta = 10^{-4} \cdot \Delta x^D$).

**`MOODu2(δ)`: DMP with a smooth-extremum override.** First the `MOODu1` test. If it fails, and the scheme exposes cached reconstruction coefficients (`has_cached_gradients`, true for MUSCL) of degree ≥ 2 (both $k_{\max} \ge 3$ and $k_i \ge 3$), the pure second derivatives $\partial_{dd} U$ of $i$ and its neighbours are inspected. The particle is accepted as a smooth extremum if, in every direction $d$ and for every component,

- the curvature does not change sign across the neighbourhood: $\min_j \partial_{dd} U_j \cdot \max_j \partial_{dd} U_j > -\delta$, and
- the curvatures are comparable: $\min_j |\partial_{dd} U_j| / \max_j |\partial_{dd} U_j| \ge \tfrac12$, or they are all below $\delta$.

Without cached coefficients, or at order ≤ 2, `MOODu2` behaves like `MOODu1`. At order 2 a smooth peak is therefore always clipped, which costs some accuracy at extrema.

**`OnlyMOOD()`** flags every particle (useful for testing: one step drives everything to order 1). **`NoMOOD`** (via `MOOD()`) never flags.

## Strategies (`MOODStrategy`)

A strategy answers two questions:

1. **Interface degrees** (`_get_interface_orders`): which degrees the two reconstructions use on an edge $(i, j)$. Only interface-based schemes (MUSCL) ask this question.
2. **Forcing** (`trigger_halo!`): whether a drop at $p$ also lowers its neighbours.

| Strategy | Interface degrees | Forcing |
|---|---|---|
| `EPD0` | $(k_i, k_j)$ | – |
| `StrictEPD0` | $(k_i, k_j)$ | neighbours of $p$ lowered to $k_p$ (one hop, no cascade) |
| `EPD1` | $\min(k_i, k_j)$ on both sides | – |
| `EPD2` | $\min$ over $\nu(i) \cup \nu(j) \cup \{i, j\}$ on both sides | – |

`Halo{Force_N}` is the abstract parent; `Force_N` is the forcing depth (0 or 1). See [MUSCL Reconstruction and Limiting](@ref) for what the interface rules mean geometrically.

## The stage-wise fixed point

The stepper runs MOOD once per stage, treating every stage as a forward-Euler substep. This is the standard construction for SSP Runge–Kutta methods (see [Butcher Tableaus](@ref)). At checkpoint $k$ (stage $k \ge 2$, or the final combination as $k = s + 1$):

```
form Y_k                                             # predictor, implicit solve, BCs
loop
    C ← particles failing the criterion              # first pass: all interior particles
                                                     # later passes: only the dirty set
    if C is empty: done
    lower k_p by one for p ∈ C (particles at order 1 are skipped)
    apply forcing from a snapshot of the lowered orders; add forced particles to C
    dirty ← C ⊕ _order_reach(interp, strategy)       # hops
    recompute gradients of Y_{k-1} on C,
              K_E[k-1] on dirty
    re-form Y_k
end
```

Some properties of this loop:

- **Termination.** Every pass lowers at least one order, and orders are bounded below by 1.
- **Exact locality.** Within a checkpoint the stage state $Y_{k-1}$ is fixed, so an order change can only alter the reconstruction of the dropped particle itself and the fluxes up to `_order_reach` hops away (0 for Central, WENO and Upwind; 1 for MUSCL; 2 for MUSCL with `EPD2`). Outside that set $K_{E,k-1}$ is unchanged and the candidate would pass again, so only the dirty set is re-checked.
- **No cascades.** Forcing reads a snapshot of the orders after the decrements, so a forced particle never forces its own neighbours in the same pass, and a particle is never lowered twice in one pass.
- **Orders persist.** Lowered orders stay lowered for the remaining stages of the step, unless the stepper is built with `reset_orders_per_stage = true`.

After the step, `mood_flags(ts)[i]` is `true` for every particle whose order MOOD lowered at some point during the step (for plotting and diagnostics).

## Writing your own

A criterion is a subtype of `MOODCriterion` plus a functor on the `MOOD` wrapper:

```julia
struct Positivity <: MOODCriterion end

function (m::MOOD{<:MOODStrategy, Positivity})(interp, p, U_prev_p, nb_slice, U_cand_p, pg, f_neighbours)
    return any(<(0), U_cand_p)          # true = inadmissible
end
```

`f_neighbours[k]` for `k ∈ nb_slice` are the neighbour states of the previous stage.

A strategy subtypes `MOODStrategy` (or `Halo{0}` / `Halo{1}`) and may specialise

```julia
_get_interface_orders(::MyStrategy, pg, i, j) -> (d_ij, d_ji)
trigger_halo!(::MyStrategy, p, pg, orders, snapshot, changed)   # mark every lowered particle in `changed`
_order_reach(::MyScheme, ::MyStrategy) -> Int                   # if the strategy widens the dependency
```

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["MOOD.jl"]
Private = false
```
