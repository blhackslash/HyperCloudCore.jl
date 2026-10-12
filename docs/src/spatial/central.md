# Central Divergence

The simplest scheme: fit the neighbour flux differences with MLS and take the divergence of the fit,

```math
\nabla\cdot F(U_i) \approx \mathcal{D}_i^{(k)}\big[\, F(U_j) - F(U_i) \,\big]_{j \in \nu(i)},
```

where $\mathcal D^{(k)}_i$ is the degree-$k$ MLS divergence operator at particle $i$. The scheme has no numerical dissipation and is meant for smooth problems, convergence studies, and as a baseline. It works for systems of any size.

## Construction

```julia
CentralDivergence(T, D, M, order; div_order = 0)
```

- `order ≥ 1` is the consistency order. The divergence uses `Interpolator{D, order}`.
- `div_order > 0` fixes the divergence degree to `min(div_order, current order)`. With `0`, it follows the particle's current order.

## Behaviour

- **Stateless.** `update_size!` and `update_content!` are no-ops; there is no pre-gather pass.
- **Starved neighbourhoods.** The degree is lowered locally while the particle has fewer than $B(D, k)$ neighbours. If even degree 1 is impossible, the divergence is zero. This local lowering does not change `particle_orders`.
- **MOOD.** The scheme reads `particle_orders[i]`, so MOOD can lower its order. Its flux at $i$ does not depend on neighbours' orders (`_order_reach = 0`), so an order drop only requires recomputing the dropped particle itself.
- **Non-conservative terms.** None are evaluated; use a Riemann-based scheme for PDEs with `evaluate_nc_jump`.

## Reference

```@docs
CentralDivergence
```
