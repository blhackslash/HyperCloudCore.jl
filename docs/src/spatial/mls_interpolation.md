# Moving Least Squares (MLS) Interpolation

All spatial schemes compute derivatives with one MLS kernel: a weighted least-squares Taylor fit of neighbour differences around the target particle. The implementation uses static arrays sized at compile time, so it allocates nothing.

## Weighted normal equations

For particle $i$ with neighbours $j$, distance vectors $x_{ij} = x_j - x_i$, weights $w_j$ and data differences $\Delta f_j$ (state, or flux, differences), we seek Taylor coefficients $c$ minimising

```math
J(c) = \sum_j w_j \left( p(x_{ij})^{\mathsf T} c - \Delta f_j \right)^2 ,
```

where $p$ is the monomial basis of degree $1 \dots p_{\max}$ with Taylor prefactors, e.g. in 2-D for degree 2

```math
p(x) = \left( x,\; y,\; \tfrac12 x^2,\; \tfrac12 y^2,\; xy \right)^{\mathsf T} .
```

There is no constant term, because the fit is to differences ($\Delta f_i = 0$). Setting $\nabla_c J = 0$ gives the normal equations

```math
\underbrace{\Big(\sum_j w_j\, p(x_{ij})\, p(x_{ij})^{\mathsf T}\Big)}_{N}\, c
= \underbrace{\sum_j w_j\, p(x_{ij})\, \Delta f_j}_{b}.
```

The coefficients are the derivatives at $x_i$, in basis order. The first $D$ entries are the gradient, and the next $D$ are the pure second derivatives $\partial_{dd}$; MUSCL and MOOD's $U_2$ criterion rely on this ordering.

The number of basis terms is $B(D, p) = \binom{p+D}{D} - 1$, available at compile time as `basis_length(Val(D), Val(p))`. A fit is only well-posed with at least $B$ neighbours, and schemes lower the local degree until this holds.

## Implementation

`Interpolator{D, IO, DO}` is a zero-size type: `D` is the dimension and `IO` the polynomial degree. Calling it on a neighbour slice runs `_mls_solve`:

- **Scaling.** Distances are multiplied by `1/scale` (the particle spacing) before the basis is evaluated, so $N$ is $\mathcal O(1)$ regardless of resolution. `build_scale_factors` undoes the scaling on the coefficients afterwards.
- **Assembly.** `N` and `b` are accumulated as `SMatrix` types in one pass over the neighbours.
- **Solve.** A Cholesky factorization of `Symmetric(N + ε I)`, with Tikhonov regularization `ε = min(10⁻¹⁰, 100 eps(T)) · tr(N)/B` (a fixed 10⁻¹⁰ capped the high-order fits at errors of ~10⁻⁸ in Float64). The normal equations square the condition number, and the regularization keeps nearly degenerate neighbourhoods solvable. If the factorization still fails, the fit returns zero.

Variants of the same kernel:

| Call | Used by | Notes |
|---|---|---|
| `interp(nb, dists, w, df; scale)` | MUSCL reconstruction | State differences → `SVector{B, State}` |
| `interp(nb, dists, w, df, mask; scale)` | WENO, Upwind (Tiwari) | Only neighbours with `mask[k] == true`; **no regularization** |
| `interp(nb, dists, w, dF, scratch; scale::Space)` | All divergence evaluations | Flux differences: one moment matrix for all directions, **with a constant term** (see below); returns $\sum_d \partial_d F_d$ |

## Divergence fits: weights that sum to zero

The schemes write the divergence as a weighted sum over neighbour pairs. With $c_{ij} \in \mathbb R^D$ the
first-derivative weights of the MLS fit at particle $i$ and $F^*_{ij} = F^*_{ji}$ the interface flux,

```math
\frac{\mathrm d U_i}{\mathrm d t} = -2 \sum_{j \in \nu(i)} c_{ij} \cdot \big(F^*_{ij} - F_i\big).
```

On a lattice this is conservative. On an irregular point cloud the total $\sum_i V_i U_i$ changes by

```math
\frac{\mathrm d}{\mathrm d t} \sum_i V_i U_i
= -\sum_{\text{pairs } \{i,j\}} \big(V_i c_{ij} + V_j c_{ji}\big) \cdot F^*_{ij}
\;+\; 2 \sum_i V_i \Big(\sum_{j} c_{ij}\Big) \cdot F_i ,
```

so there are two defects: the **pair term** (the weights of a pair are not antisymmetric) and the
**row-sum term** (the weights of a particle do not sum to zero). Both vanish on a lattice by symmetry and
grow linearly with the irregularity of the cloud; for shocks they cause a systematic mass error that does
not shrink under refinement.

**Why the weights of the Taylor fit do not sum to zero.** The reconstruction fits differences
$\Delta f_j = f_j - f_i$ with a basis $p(x)$ of the monomials of degree $1 \dots q$ (no constant), i.e. the
fitted polynomial passes through the particle's own value. Its derivative weights are the first $D$ rows of

```math
c_{ij} = N^{-1}\, w_{ij}\, p(x_{ij}), \qquad N = \sum_{k} w_{ik}\, p(x_{ik})\, p(x_{ik})^{\mathsf T},
```

and their sum $\sum_j c_{ij} = N^{-1} \sum_j w_{ij}\, p(x_{ij})$ (first $D$ rows) is the derivative the fit
assigns to *constant* data $\Delta f_j = 1$. Without a constant in the basis, the fit has to explain the
constant with the monomials, and on a non-symmetric stencil $\sum_j w_{ij}\, p(x_{ij}) \ne 0$ leaks into the
gradient.

**The constant term.** The divergence fits use the extended basis $\tilde p(x) = (1, p(x))$, i.e. the model
$a_0 + p(x)^{\mathsf T} a$ over the neighbours only (the particle itself is *not* a data point):

```math
\tilde N = \sum_{j} w_{ij}\, \tilde p(x_{ij})\, \tilde p(x_{ij})^{\mathsf T}, \qquad
\tilde c_{ij} = \tilde N^{-1}\, w_{ij}\, \tilde p(x_{ij}),
```

and the derivative weights $c_{ij}$ are rows $2 \dots D+1$ of $\tilde c_{ij}$. Constant data is now
reproduced exactly by the intercept ($a_0 = 1$, $a = 0$), so

```math
\sum_{j \in \nu(i)} c_{ij} = 0 \qquad \text{exactly, on every stencil.}
```

Then $2\sum_j c_{ij}\cdot(F^*_{ij} - F_i) = 2\sum_j c_{ij}\cdot F^*_{ij}$: the particle's own flux drops out,
the row-sum defect is gone, a constant flux has zero divergence on any point cloud (free-stream
preservation), and only the pair term is left. Including the particle as a data point would break this:
its own weight $c_{ii}$ is part of the sum, and the neighbour weights then sum to $-c_{ii} \ne 0$.

What it costs and what it does not do:

- The fit reproduces the same polynomials as before, so the consistency order does not change; it has one
  more unknown, $B(D, q) + 1$, and needs one more neighbour.
- The divergence cache (`MLSWeights` with an intercept flag) is separate from the reconstruction cache: one
  $(B + 1)^2$ matrix more per particle.
- The pair term remains, so the scheme is still not exactly conservative on irregular clouds. That needs
  antisymmetric face coefficients (a global correction).

Measured on periodic Burgers with 20 % jitter (mass drift after the shock, before → after): 1-D MUSCL1
$6.6\cdot10^{-5} \to 4.7\cdot10^{-6}$, 1-D MUSCL2 with Venkatakrishnan limiter
$1.0\cdot10^{-3} \to 1.1\cdot10^{-4}$ ($R = 2.5\Delta x$) and $3.4\cdot10^{-4} \to 7.5\cdot10^{-5}$
($R = 5.5\Delta x$), 2-D MUSCL2 $1.7\cdot10^{-4} \to 7.0\cdot10^{-5}$ ($R = 2.5\Delta x$); smooth accuracy
unchanged.

## Dispatch on a runtime order

Schemes store a tuple of interpolators, one per order, and choose an entry at run time from the particle's current order:

- `dispatch_interpolator(interps, order, ...)` returns the padded coefficient vector of `interps[order]`.
- `compute_dynamic_divergence(interps, div_idx, ...)` returns the divergence from `interps[div_idx]`.

Both are `@generated` into an `if`-chain over the tuple, so each branch is type-stable and the runtime order costs one integer comparison per level.

`mask_basis(basis, order, Val(D))` zeroes all basis terms above degree `order - 1`. MUSCL uses it to evaluate a high-degree reconstruction at a lower interface degree without a separate fit.
