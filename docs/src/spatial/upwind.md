# Upwind Divergence

First-order upwind schemes on a particle cloud. Three algorithms are available, selected with a symbol at construction:

| Algorithm | Systems | Dimension | Form |
|---|---|---|---|
| `:Classic` | any $M$ | any $D$ | Riemann fluxes at particle midpoints (conservative, supports non-conservative jumps) |
| `:Tiwari` | $M = 1$ | any $D$ | One-sided MLS gradient per direction, times the advection speed |
| `:Praveen` | $M = 1$ | $D = 2$ | Normal/tangential splitting of edge velocities (Praveen's meshless upwind) |

## Construction

```julia
UpwindDivergence(T, D, M, order, alg::Symbol, flux::NumericalFluxFunction; div_order = 0)
```

`order` must be 1. `:Tiwari` and `:Praveen` assert `M == 1`. `:Praveen` is only defined for `D == 2`; other dimensions give a `MethodError` on the first call.

## Algorithms

**Classic.** For each neighbour, the numerical flux between $U_i$ and $U_j$ is evaluated (no reconstruction) and

```math
\nabla\cdot F_i \approx 2\, \mathcal D^{(1)}_i\big[\, F_{\text{num},ij} - F(U_i) + \text{nc\_jump}_{ij} \,\big].
```

See [Numerical Flux Functions](@ref) for the factor 2. The divergence is zero if the particle has fewer than $D$ neighbours.

**Tiwari.** For each direction $d$, the stencil is restricted to upwind neighbours, $v_d\, x_{ij,d} \le 0$ with $v_d = {\tt velocity}(eq, U_i, d)$, and

```math
\nabla\cdot F_i \approx \sum_d v_d\, \big(\partial_d u\big)_{\text{upwind}} .
```

This is the advective (quasi-linear) form $v\cdot\nabla u$, which is conservative only for constant $v$. A direction whose upwind stencil is too small contributes zero.

**Praveen.** Uses a 2×2 first-order moment matrix; each neighbour offset is projected onto normal $n$ and tangential $s$ directions, and only the upwind parts $\min(v_n, 0)$ and $\min(\bar\beta\, v_s, 0)$ contribute. The result is scaled by 2. The divergence is zero for fewer than three neighbours or a singular moment matrix.

## Behaviour

- **Stateless:** no pre-gather pass.
- **Order:** always 1, so MOOD has nothing to lower; the scheme is a natural fallback reference.
- **Moving grids:** all three algorithms read only the PDE API, so with an `ALEFrame` they upwind with respect to the relative velocity $v - w$.

## Reference

```@docs
UpwindDivergence
```
