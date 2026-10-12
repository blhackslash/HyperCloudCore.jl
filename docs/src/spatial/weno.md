# Meshfree WENO

A weighted essentially non-oscillatory scheme for **scalar** equations. In each direction it blends a central MLS gradient with an upwind-biased one; the weights favour whichever stencil is smoother.

## From finite volumes to particles

Grid-based WENO combines several candidate stencils with non-linear weights that vanish for stencils crossing a discontinuity. On a particle cloud the candidates are subsets of the neighbourhood:

- **Central stencil $C$:** all neighbours.
- **Directional stencil $S_d$:** for each direction $d$, only the neighbours on the upwind side, i.e. $x_{ij,d} < 0$ if $v_d > 0$ and $x_{ij,d} \ge 0$ otherwise.

Each stencil gets an MLS fit of the state differences, and the derivatives of that fit serve as smoothness indicators.

## Formulation

For a stencil with fitted derivatives $D^\beta u$ (basis index $\beta$), the smoothness indicator is

```math
IS = \sum_{\beta \le D} \big(D^\beta u\big)^2 \Delta x^2 + \sum_{\beta > D} \big(D^\beta u\big)^2 \Delta x^4 ,
```

so first derivatives are weighted by $\Delta x^2$ and all higher-order coefficients by $\Delta x^4$, with $\Delta x = \min_d \Delta x_d$. The weights are

```math
\beta = \frac{1/2}{(IS + \epsilon)^2}, \qquad
w_S = \frac{\beta_S}{\beta_S + \beta_C}, \quad w_C = \frac{\beta_C}{\beta_S + \beta_C},
```

with $\epsilon = 10^{-12}$, applied per state component. The divergence is

```math
\nabla\cdot F_i \approx \sum_d v_d \left( w_S\, (\partial_d u)_{S_d} + w_C\, (\partial_d u)_C \right),
\qquad v_d = {\tt velocity}(eq, U_i, d).
```

This is the advective form $v\cdot\nabla u$ with the speed frozen at $U_i$. It is exact for linear advection; for non-linear scalar laws it is a quasi-linear approximation and not conservative.

## Construction

```julia
WENO(T, D, M, order; div_order = 0)
```

- `order ≥ 2` is required, because the smoothness indicators need second-order coefficients.
- `M == 1` is asserted. The upwind side of a stencil needs a single wave speed per direction; systems would need a characteristic decomposition first.
- `div_order > 0` caps the fit degree at `min(div_order, current order)`.

## Behaviour

- **Stateless:** no pre-gather pass; the masked fits run inside the flux call and use `ib.mask`.
- **Starved stencils.** The central degree is lowered locally until the neighbourhood suffices. A directional stencil that is too small for degree 1 is skipped, and that direction uses the central gradient alone.
- **MOOD.** The scheme reads `particle_orders[i]`; `_order_reach = 0`.
- **Moving grids.** With an `ALEFrame`, `velocity` returns $v_d - w_d$, so both the upwind side and the transport speed are relative to the moving particle.

## Reference

```@docs
WENO
```
