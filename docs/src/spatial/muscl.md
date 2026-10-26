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
- `div_order`: with `0` the divergence fit follows the particle's current order $k_i$; with a positive value it uses `div_order` for every particle, independent of order drops (keeps the divergence weights of neighbouring particles consistent when MOOD lowers orders). The degree that order maps to is described in [Divergence degree](#Divergence-degree).
- The divergence fit has a **constant term** and runs over the neighbours only, so its weights sum to zero; see [Divergence fits: weights that sum to zero](@ref).

## Divergence degree

For a divergence order $o$ (the particle's order, or `div_order`) the divergence is fitted with degree

| order $o$ | 1 | 2 | 3 | 4 | 5 | 6 | … |
|---|---|---|---|---|---|---|---|
| reconstruction degree | 0 | 1 | 2 | 3 | 4 | 5 | $o - 1$ |
| divergence degree | 1 | 1 | **3** | 3 | **5** | 5 | $o - 1$ (even $o$), $o$ (odd $o$) |

**Why odd orders go one degree up.** On a (nearly) symmetric stencil, the first derivative of a
least-squares fit of degree $q$ is accurate to $O(h^q)$ for even $q$ but to $O(h^{q+1})$ for odd $q$: the
first monomial the fit misses, $x^{q+1}$, is odd for even $q$ and aliases into the gradient (it projects
onto $x$), while for odd $q$ it is even and orthogonal to the gradient. Degree $o - 1$ is therefore enough
for even orders, but MUSCL3 and MUSCL5 were only second and fourth order with it. Fitting the divergence
of an odd order with degree $o$ restores the order (smooth advection on 20 % jittered grids: MUSCL3 EOC
2.0 → 3.0, errors 7× smaller). It does not depend on the reconstruction degree: reconstruction degree 2
with divergence degree 2 is second order, with divergence degree 3 third order.

The higher degree has two conditions, checked per particle (`_divergence_index`):

- the stencil has at least 1.5× as many neighbours as the fit has unknowns ($B(D, o) + 1$, with the
  constant term). An exactly determined fit is fragile on moving grids; in 1-D, MUSCL3 needs
  $R \ge 3.5\Delta x$ (6 neighbours for 4 unknowns), otherwise it keeps degree 2 and second order;
- in 3-D only up to degree 3 (basis size).

Independently, every divergence fit lowers its degree where a particle has fewer neighbours than unknowns.

## Supported orders

The fits are solved inside the kernels with `StaticArrays` matrices. Up to 24 unknowns the Cholesky
factorisation is unrolled and runs on every backend; larger fits fall back to a heap-allocating LAPACK call,
which still works on the CPU backends (slower) but does not compile for GPUs. The largest fit of a scheme is
the reconstruction ($B(D, o-1)$ unknowns) or the divergence ($B(D, q) + 1$):

| $D$ | highest order on GPUs | largest fit there | first CPU-only order | minimum radius for the full order |
|---|---|---|---|---|
| 1 | ≥ 8 (tested; the fit sizes would allow ~23) | 8 unknowns at order 8 | – | MUSCL2: 1.5Δx, MUSCL3: 3.5Δx, MUSCL4: 2.5Δx (3.5Δx on moving grids), MUSCL5: 5.5Δx |
| 2 | **6** | 21 (divergence of degree 5) | 7 (reconstruction: 27 unknowns) | MUSCL3/4: 2.5Δx, MUSCL5/6: 3.5Δx |
| 3 | **4** | 20 (divergence of degree 3) | 5 (reconstruction: 34 unknowns) | MUSCL3/4: 2.5Δx |

(Verified with a few steps of 2-D MUSCL7 and 3-D MUSCL5 on `KABackend(ROCBackend())`, which fail to compile,
and on `PolyesterBackend()`, which run.) Two more limits apply to high orders: the CFL estimate evaluates the
derivative stencil with at most degree 4 (degree 3 in 3-D), so for higher divergence degrees the time step
is estimated from the degree-4 stencil and slightly too large; and the radius must provide enough
neighbours for the reconstruction, $B(D, o - 1)$, or particles drop to first order (see the pre-gather pass
below).

## The two passes

1. **Pre-gather (`update_content!`).** Fits the reconstruction at the particle's current order and stores the (limited) coefficients in `gradients[i]`. If the neighbourhood has fewer than $B(D, k-1)$ neighbours, the particle's order is set to 1 **in `particle_orders`**, so MOOD and the EPD rules see the starved particle as first order.
2. **Flux.** For every neighbour, it chooses the interface degrees from the MOOD strategy, masks both reconstructions to those degrees (`mask_basis`), evaluates the numerical flux and non-conservative jump, and differentiates the interface differences.

Because the flux at $i$ reads `gradients[j]` of its neighbours, all gradients of a stage must be computed before any flux; that is why the pre-gather pass is separate.

## Aligned dissipation and a uniform divergence degree

MOOD drops the divergence degree together with the reconstruction order, and a pair of particles with
different divergence degrees is never conservative (its face vectors differ in magnitude, not only in sign,
see the conservation section of the MLS page): a front of dropped particles loses mass at a rate that does
not shrink with resolution. The degree drop is not needed for conservation, it compensates a stability
flaw of the geometric dissipation: with the states sorted by the sign of the *distance*, the dissipation
of neighbour `j` enters particle `i` with the coefficient `½ Σ_d s_d c_ij,d sign(d_ij,d)`, which is negative
on the slots where a high-degree weight changes sign, so the first-order fallback is not monotone and
high-degree weights overshoot (12 % on a box profile with `div_order = 5`).

[`AlignedRusanovFlux`](@ref) sorts the states by the sign of the weight instead (`dsigns`, kept with the
divergence cache) on every pair where one of the two particles is below the full order; the rule is
symmetric in the pair, and elsewhere the flux is the plain one. The coefficient becomes
`½ Σ_d s_d |c_ij,d| ≥ ½ |c_ij·f'|`, so the first-order update is local-extremum-diminishing for any zero-sum
weights. With it the divergence degree can stay uniform (`div_order = max_order`) while MOOD lowers the
reconstruction only: no mixed-degree pairs, and the mass error is the integrated truncation error,
`O(h^q)` where the flux is smooth and `O(h)` across shocks on irregular clouds (exactly zero on a lattice).
Without drops the scheme is identical to the plain flux. The price is a more diffusive fallback where MOOD
drops (the full sum `Σ_d s_d |c_ij,d|` of a high-degree weight set is larger than that of degree 1).

Measured on a 2-D cloud (`R = 5.5 Δx`, jitter 0.2, MUSCL5 with MOOD EPD1/u2, `div_order = 5`), relative mass
drift at 50² / 100² / 200²: box advection to `t = 1` goes from `+8.0e-3 / +6.2e-3 / +2.0e-3` with the adaptive
degree to `−1.1e-3 / −5.8e-4 / +5.9e-5` (lattice: `1e-14`), with the same extrema; 2-D Burgers to `t = 3`
(shock) from `−6.1e-2 / −5.4e-2 / −5.0e-2` to `+7e-5 / −2.4e-4 / −2.2e-4` with no overshoot. The fixed
degree with the geometric dissipation keeps the mass but overshoots by 11–13 % (box) and 6 % (Burgers).

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
