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

[`AlignedRusanovFlux`](@ref) sorts the states by the sign of the weight instead (the per-slot weights
`dweights`, kept with the divergence cache) on every pair where one of the two particles is below the full order, and on
every pair of a first-order scheme (order 1, where no particle can be below the full order: with the geometric sorting its
degree-1 weights on a jittered cloud with `R = 2.5Δx` give about 2 % negative first-order coefficients, down to −0.57). The
rule is symmetric in the pair, and elsewhere the flux is the plain one. The signs are those of the weights the particle
multiplies the slot's flux difference with: `dweights` on the cached full-degree fit, and the weights of the dynamic fit
for a particle off the cached path (a degree lowered by the neighbour-count safeguard, e.g. in 1-D with `R = 2.5Δx` and
`div_order = 3`, a lower particle order with `div_order = 0`, or no cached factorisation). With the aligned flux such a
particle always takes the per-slot path of the streamed divergence below, also with `streamed = false`, so each slot's
weight is solved once and used for both the sorting and the divergence. The coefficient becomes
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

## Moment-corrected first-order dissipation

With constant reconstructions the Rusanov dissipation enters particle `i` as
``\sum_j \sum_d s_{ij,d}\, g^d_{ij} (U_j - U_i)``, with ``g^d_{ij} = |c^d_{ij}|`` for
[`AlignedRusanovFlux`](@ref) (``\pm c^d_{ij}`` by the sign of the distance for the geometric sorting). For a
smooth state that is ``\sum_d s_d\, \beta^d_i \cdot \nabla U + O(h)`` with the first moment
``\beta^d_i = \sum_j g^d_{ij} x_{ij}``. Since ``|c| \sim 1/h`` and ``x \sim h`` the moment is ``O(1)``; it vanishes
only on symmetric stencils, so on an irregular cloud the first-order scheme is not consistent. For linear
advection the effective local speed ``-\sum_j a_{ij} x_{ij}`` scatters by 7–13 % (1-D) and 5–8 % (2-D) around the
exact one at jitter 0.2–0.4, independently of ``h``, and waves lag by a constant fraction: ``-4.6 \cdot 10^{-3}``
at jitter 0.4 and ``-9 \cdot 10^{-4}`` at 0.2 in 1-D (`R = 2.5Δx`), so the error stops converging.

With `moment_correction = true` (builder key `:moment_correction` in the `Scheme` namespace) a first-order
`MUSCL` adds the dissipation ``\sum_d s_{ij,d}\, e^d_{ij} (U_j - U_i)`` with static weights ``e^d_{ij} \ge 0`` that
cancel the moment, ``\sum_j e^d_{ij} x_{ij} = -\beta^d_i`` (`mweights`, computed with `dweights` once per geometry,
`update_moment_weights!`). Per particle and axis, ``e^d_{ij} = (\tilde x_{ij} \cdot \mu)_+`` with the minimiser
``\mu`` of the convex ``\tfrac12 \sum_j (\tilde x_{ij}\cdot\mu)_+^2 + \beta^d_i \cdot \mu`` (semismooth Newton in
`D` unknowns). It exists whenever ``-\beta^d_i`` lies in the cone of the offsets, i.e. for every particle
surrounded by its neighbours, and favours the far neighbours, which keeps the added dissipation small.

- The coefficients only grow (``s(g + e) \ge s g``), so the positivity argument of the aligned flux, the
  maximum principle and the LED property are kept, under the time-step bound with the larger sum
  ``\tau \sum_{j,d} s_{ij,d}(g^d_{ij} + e^d_{ij}) \le 1``. On periodic clouds with `R = 2.5Δx` the correction adds on
  average 6 % (jitter 0.2) and 11 % (jitter 0.4) to ``\sum_j \sum_d g^d_{ij}``; the largest sum over the cloud,
  which sets that bound, grows by a factor 1.13 / 1.12 (1-D / 2-D, jitter 0.2) and 1.28 / 1.44 (jitter 0.4).
- The weights use the pair speeds ``s_{ij,d}`` of the flux, so the moment left is
  ``\sum_{j,d} (s_{ij,d} - s_{i,d}) (g + e)\, x_{ij} = O(h)`` where the speeds are smooth (exactly zero for linear
  advection), without the extra dissipation of a common speed ``s_{i,d} = \max_j s_{ij,d}``.
- Near a non-periodic boundary the stencil is one-sided and ``-\beta`` can lie on or near the edge of the
  cone: the weights of a particle are scaled down to at most its dissipation sum (`_MOMENT_CAP`), and a
  particle without a solution (or without a cached factorisation) keeps the uncorrected dissipation.
- Higher orders are unchanged (the keyword is ignored for `max_order > 1`): with reconstructions the
  dissipation acts on jumps of size ``O(h^k)``, and a particle that MOOD drops to order 1 sits next to a
  discontinuity, where the ``O(1)`` consistency error is of the size the fallback accepts anyway.

Measured (1-D periodic sine, linear advection, MUSCL1/RK2, AlignedRusanov, CFL 0.4, `t = 10`), L1 error for
`N = 400 / 1600 / 6400 / 25600` at jitter 0.4: `0.375 / 0.203 / 0.181 / 0.180` without and
`0.386 / 0.133 / 0.036 / 0.0092` with the correction (first order); the phase lag drops from `0.046` periods to
`1.2e-3 / 7.7e-5 / 4.9e-6 / 3.0e-7`. The gain needs the diffusion error to fall below the lag: at
`N = 400`, and in 2-D up to `200²` (sine, `t = 2`, jitter 0.2), the first-order diffusion dominates and the
added dissipation raises the error by 1–4 %. On a lattice the weights vanish to round-off and the scheme is
unchanged.

## Streamed divergence

By default the flux pass writes the flux difference $\Delta F_{ij} = F^*_{ij} - F_i$ of every neighbour to the
per-slot buffer `df_flux` and fits them afterwards (`apply_weights_divergence`, or the dynamic fit for a
reduced order). Per neighbour that costs $B \cdot D \cdot M$ flops for the right-hand side of the fit and
$D \cdot M$ numbers of buffer traffic, where $B$ is the number of fit terms (21 for degree 5 in 2-D): for
systems with many components (kinetic models) the fit dominates the step and the buffer the memory.

With `streamed = true` the scheme keeps the derivative weights $c_{ij}$ of the full-degree fit per slot
(`dweights`, $D$ numbers, computed with the divergence cache at $D\cdot B$ flops per slot, once per geometry)
and the flux pass accumulates $\sum_j c_{ij}\cdot\Delta F_{ij}$ directly: $D\cdot M$ flops per neighbour, no
buffer, no static $B\times D M$ matrices. A particle whose fit is not the cached one (a MOOD drop, or no
factorisation) assembles and factorises its own moment matrix once and solves each slot's weights inside the
loop ($B^2$ per slot). The numbers are those of the fit up to round-off (the summation order differs). The
stepper does not allocate `df_flux`/`df_scratch` for a streamed scheme (`uses_slot_buffers`), and the memory
budget counts `dweights` instead. The toggle is the constructor keyword (`:streamed` in the builders'
`Scheme` namespace); the default is off, the buffered path, so existing runs are unchanged bit for bit (with
[`AlignedRusanovFlux`](@ref), particles off the cached path take the per-slot path in either mode, see above).

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
