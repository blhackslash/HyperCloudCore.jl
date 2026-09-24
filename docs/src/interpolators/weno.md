# Meshfree WENO Reconstruction

This module implements a Weighted Essentially Non-Oscillatory (WENO) scheme adapted for meshfree particle methods. WENO schemes are designed to achieve high-order accuracy in smooth regions while robustly capturing shocks and discontinuities without spurious oscillations.

## 1. WENO: From Finite Volume to Meshfree

In traditional grid-based Finite Volume (FV) methods, the standard ENO (Essentially Non-Oscillatory) scheme evaluates multiple candidate stencils (e.g., central, left-biased, right-biased) and strictly chooses the single "smoothest" one to perform the reconstruction. WENO improves upon this by taking a convex combination of *all* candidate stencils. Each stencil is assigned a non-linear weight inversely proportional to its smoothness indicator. If a stencil crosses a shock, its smoothness indicator spikes, and its resulting weight drops to zero, effectively masking out the discontinuity while retaining high-order accuracy from the smooth stencils.

In a **meshfree context**, the concept of discrete neighboring cells is replaced by a continuous neighbor cloud. 

*   **Stencil Partitioning:** Instead of combining discrete cell averages, meshfree WENO dynamically partitions the particle's interaction neighborhood into subsets: a central stencil containing all neighbors, and directional (upwind-biased) stencils masked by geometry and characteristic velocities.
*   **Smoothness Indicators:** Instead of using Newton divided differences, the smoothness indicators are formulated directly from the high-order spatial derivatives computed via Moving Least Squares (MLS).
*   **Gradient Blending:** The final spatial derivative is a non-linear weighted sum of the gradients computed from the central and directional neighbor clouds.

---

## 2. Theoretical Formulation

For a given particle $i$, the scheme computes a central gradient using the full neighbor stencil ($C$) and a directional, upwind-biased gradient ($S$).

The smoothness of a given stencil is evaluated by summing the squares of its reconstructed spatial derivatives, scaled by the local particle spacing $\Delta x$ to maintain dimensional consistency:

$$ IS_{stencil} = \sum_{k} \left( D^k u \right)^2 \Delta x^{2k} $$

Where $D^k u$ represents the $k$-th order derivatives (e.g., first derivatives are scaled by $\Delta x^2$, second derivatives by $\Delta x^4$). 

The non-linear weights for the central ($w_C$) and directional ($w_S$) stencils are computed using their respective smoothness indicators:

$$ \beta_{stencil} = \frac{0.5}{(IS_{stencil} + \epsilon)^2} $$
$$ w_S = \frac{\beta_S}{\beta_S + \beta_C}, \quad w_C = \frac{\beta_C}{\beta_S + \beta_C} $$

The final spatial divergence for dimension $d$ is the weighted sum of the stencil derivatives:

$$ \frac{\partial F_d}{\partial x_d} = \left( w_S \left( \frac{\partial u}{\partial x_d} \right)_S + w_C \left( \frac{\partial u}{\partial x_d} \right)_C \right) v_d $$

---

## 3. Implementation Details (`WENO`)

The `WENO` struct provides a completely stateless API (`update_size!` and `update_content!` are no-ops), ensuring zero allocations during the explicit time-stepping loop. Because the smoothness indicators require higher-order derivatives to detect oscillations, the constructor strictly asserts that the interpolation order is at least 2 (`order >= 2`).

### Execution Flow

1.  **Central Stencil Evaluation:**
    The functor first unmasks all neighbors (`ib.mask[global_idx] = true`) and evaluates the full MLS interpolator to obtain the central derivatives `resC`. It then calculates the central smoothness indicator `smoothC` by summing the squared derivatives. First derivatives ($k \le D$) are weighted by `dx2`, while higher-order terms are weighted by `dx4`. 
2.  **Directional Stencil Partitioning:**
    The algorithm loops over each spatial dimension $d$. It checks the local advection velocity `vel[d]` to determine the upwind direction (`use_left`). It iterates over the neighbor slice, masking out downwind particles by checking if the distance vector component `dist_k[d]` aligns with the upwind direction.
3.  **Fallback Mechanism:**
    If the masked upwind stencil contains fewer neighbors than the required interpolation order (`stencil_size < weno.order`), the algorithm cannot safely perform an MLS inversion. It bypasses the WENO blending for that dimension and safely falls back to purely using the central gradient `resC[d] * vel[d]`.
4.  **Directional Evaluation and Blending:**
    If the upwind stencil is sufficiently populated, the MLS interpolator is called again using the masked interaction buffer to yield the directional derivatives `resS`. The directional smoothness indicator `smoothS` and the non-linear weights (`wS`, `wC`) are calculated. A small tolerance `e_tol = 1e-12` is added to the denominators to prevent division by zero in perfectly uniform regions. Finally, the gradients are blended and accumulated into `div_total`.