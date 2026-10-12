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
- **Solve.** A Cholesky factorization of `Symmetric(N + ε I)`, with Tikhonov regularization `ε = 1e-10 · tr(N)/B`. The normal equations square the condition number, and the regularization keeps nearly degenerate neighbourhoods solvable. If the factorization still fails, the fit returns zero.

Variants of the same kernel:

| Call | Used by | Notes |
|---|---|---|
| `interp(nb, dists, w, df; scale)` | MUSCL reconstruction | State differences → `SVector{B, State}` |
| `interp(nb, dists, w, df, mask; scale)` | WENO, Upwind (Tiwari) | Only neighbours with `mask[k] == true`; **no regularization** |
| `interp(nb, dists, w, dF, scratch; scale::Space)` | All divergence evaluations | Flux differences: fits each direction's flux separately and returns $\sum_d \partial_d F_d$ |

## Dispatch on a runtime order

Schemes store a tuple of interpolators, one per order, and choose an entry at run time from the particle's current order:

- `dispatch_interpolator(interps, order, ...)` returns the padded coefficient vector of `interps[order]`.
- `compute_dynamic_divergence(interps, div_idx, ...)` returns the divergence from `interps[div_idx]`.

Both are `@generated` into an `if`-chain over the tuple, so each branch is type-stable and the runtime order costs one integer comparison per level.

`mask_basis(basis, order, Val(D))` zeroes all basis terms above degree `order - 1`. MUSCL uses it to evaluate a high-degree reconstruction at a lower interface degree without a separate fit.
