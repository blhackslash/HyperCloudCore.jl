# MUSCL Reconstruction and Limiting

MUSCL reconstructs the solution around every particle with an MLS Taylor polynomial, evaluates both sides' polynomials at the midpoint of each particle pair, resolves the jump with a numerical flux, and differentiates the resulting interface fluxes. It is the scheme with the most knobs: arbitrary order, *a priori* slope limiting, and the Edge Polynomial Degree (EPD) strategies for MOOD.

## From finite volumes to particles

In finite-volume MUSCL each cell carries a polynomial that is evaluated at the cell faces. On a particle cloud:

- The "cell" is a particle and its polynomial is an MLS fit of state differences over its neighbourhood (the cached `gradients`).
- The "face" between $i$ and $j$ is the midpoint $x_i + x_{ij}/2$. The two reconstructed states are

```math
U_{ij} = U_i + \sum_{\beta} c_{i,\beta}\, p_\beta\!\left(\tfrac12 x_{ij}\right), \qquad
U_{ji} = U_j + \sum_{\beta} c_{j,\beta}\, p_\beta\!\left(-\tfrac12 x_{ij}\right).
```

- The divergence is $2\,\mathcal D_i\big[F_{\text{num}}(U_{ij}, U_{ji}) - F(U_i) + \text{nc\_jump}\big]$.

## Construction

```julia
MUSCL(T, D, M, order, limiter, flux; div_order = 0)
```

- `order ≥ 1` is the **consistency order**. The reconstruction has degree `order - 1`: `order = 1` is piecewise constant (first-order upwind with the given flux), `order = 2` linear, `order = 3` quadratic, and so on.
- `limiter`: `NoLimiter()`, `BarthJespersenLimiter(mode)`, `VenkatakrishnanLimiter(mode)`, and (1-D only) `MinmodLimiter(mode)`, `SuperbeeLimiter(mode)`. `mode = :hard` limits all coefficients, `:soft` only the first-order ones.
- `flux`: a `NumericalFluxFunction`.
- `div_order`: with `0` the divergence fit follows the particle's current order $k_i$; with a positive value it uses `div_order` for every particle, independent of order drops. For an order $o$ (either one) the divergence is fitted with degree $o - 1$ for even $o$ and degree $o$ for odd $o$ ($\max(1, \cdot)$): on (nearly) symmetric stencils the derivative of an even-degree least-squares fit is only $O(h^q)$, because the next, odd monomial aliases into the gradient, while an odd degree gains one order. With degree $o - 1$, MUSCL3 and MUSCL5 were only second and fourth order. The higher degree is used where the stencil has at least 1.5× as many neighbours as the fit has unknowns (1-D MUSCL3: $R \ge 3.5\Delta x$) and up to degree 3 in 3-D; every fit lowers its degree where there are fewer neighbours than unknowns.
- The divergence fit has a **constant term** and runs over the neighbours only (the reconstruction fits go through the particle). Its weights then sum to zero, $\sum_j c_{ij} = 0$, which removes one of the two conservation defects of $2\sum_j c_{ij}\cdot(F^*_{ij} - F_i)$ on irregular point clouds.

## The two passes

1. **Pre-gather (`update_content!`).** Fits the reconstruction at the particle's current order and stores the (limited) coefficients in `gradients[i]`. If the neighbourhood has fewer than $B(D, k-1)$ neighbours, the particle's order is set to 1 **in `particle_orders`**, so MOOD and the EPD rules see the starved particle as first order.
2. **Flux.** For every neighbour, it chooses the interface degrees from the MOOD strategy, masks both reconstructions to those degrees (`mask_basis`), evaluates the numerical flux and non-conservative jump, and differentiates the interface differences.

Because the flux at $i$ reads `gradients[j]` of its neighbours, all gradients of a stage must be computed before any flux; that is why the pre-gather pass is separate.

## Slope limiting (*a priori*)

- **Barth–Jespersen / Venkatakrishnan** (any dimension). The reconstruction is evaluated at every interface. If an interface value leaves the range $[\min_{\nu(i)} U, \max_{\nu(i)} U]$, the coefficients are scaled by $\phi = \min_j \phi(r_j)$ with $\phi_{BJ}(r) = \min(1, r)$ and the smooth Venkatakrishnan function. This is done per state component.
- **Minmod / Superbee** (1-D only). They compare the left and right one-sided slopes from the nearest neighbours on each side.

Limiting and MOOD can be combined: the limiter acts on every reconstruction, and MOOD reduces the order where the limited update is still inadmissible.

## Interface degrees and EPD strategies

When MOOD lowers some particles' orders, the two sides of an edge can have different degrees. The `MOODStrategy` decides which degrees are used at the interface, via `_get_interface_orders(strategy, pg, i, j)`:

| Strategy | Interface degrees $(d_{i\to j}, d_{j\to i})$ | Neighbour forcing |
|---|---|---|
| `EPD0` | $(k_i, k_j)$: each side uses its own order (mismatched reconstruction allowed) | none |
| `StrictEPD0` | $(k_i, k_j)$ | when $i$ drops, its neighbours are lowered to $k_i$ |
| `EPD1` | $(m, m)$ with $m = \min(k_i, k_j)$ | none |
| `EPD2` | $(m, m)$ with $m = \min_{l \in \nu(i)\cup\nu(j)\cup\{i,j\}} k_l$ | none |

(Interface degree $k$ means a degree-$(k-1)$ polynomial.) These map the finite-volume EPD ideas onto edges of the neighbour graph. Moving down the table, the transition between high and low order becomes wider and smoother, and more particles have to be recomputed after a drop: an order change at $p$ alters fluxes up to `_order_reach(::MUSCL, strategy)` hops away, which is 1 for `EPD0`, `StrictEPD0` and `EPD1`, and 2 for `EPD2`. The stepper uses this to limit recomputation to the affected particles (see [Multi-Dimensional Optimal Order Detection (MOOD)](@ref)).

## Stepper hooks

The end of `MUSCL.jl` defines the hooks that tell the time stepper and MOOD about MUSCL:

```julia
_order_reach(::MUSCL, ::MOODStrategy) = 1     # 2 for EPD2
_divergence_degree(::MUSCL)                     # degree of the full-order divergence fit, for the CFL estimate
has_cached_gradients(::MUSCL) = true          # enables MOODu2's curvature test
cached_gradients(m::MUSCL) = m.gradients
```

They have to come after the `MUSCL` type definition. The time-stepping files never name `MUSCL` themselves.

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["MUSCL.jl", "Limiter.jl"]
```
