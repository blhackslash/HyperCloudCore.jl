# Meshfree MUSCL Reconstruction & Limiting

This module implements a high-order Monotonic Upstream-centered Scheme for Conservation Laws (MUSCL) adapted for meshfree particle methods. It includes *a priori* slope limiters and seamlessly integrates with the global Multi-dimensional Optimal Order Detection (MOOD) framework through specialized Edge Polynomial Degree (EPD) strategies.

## 1. MUSCL: From Finite Volume to Meshfree

In traditional grid-based Finite Volume (FV) methods, the MUSCL approach achieves higher-order spatial accuracy by reconstructing the solution within each cell using piecewise polynomials, rather than assuming a constant state. These reconstructed polynomials are then evaluated at the cell interfaces to compute high-order numerical fluxes.

In a **meshfree context**, the concept of a "cell" translates to a "particle," and "interfaces" are conceptualized symmetrically midway along the distance vector connecting two interacting neighbor particles. 

*   **Reconstruction:** Instead of cell-averaged gradients, we use Moving Least Squares (MLS) to compute high-order spatial derivatives directly at the particle locations. 
*   **Interface Evaluation:** The left and right states at the conceptual interface between particle $i$ and neighbor $j$ are reconstructed by evaluating the respective Taylor polynomials along the distance vector $\vec{x}_{ij}$.

To avoid non-physical oscillations (Gibbs phenomenon) when reconstructing across shocks or steep gradients, the scheme must be constrained. This is achieved either *a priori* (Slope Limiters) or *a posteriori* (MOOD).

---

## 2. Slope Limiting (*A Priori*)

Slope limiters operate by directly suppressing the reconstructed gradients before interface states are evaluated. 

For robust arbitrary N-dimensional simulations, this implementation primarily relies on the **Barth-Jespersen (BJ)** and **Venkatakrishnan (VK)** limiters. 
*   Both methods enforce a monotonicity principle by ensuring that the reconstructed states at the conceptual interfaces do not exceed the maximum or minimum bounds defined by the particle's immediate neighbors.
*   While BJ strictly enforces these bounds (often hindering convergence to steady state), VK applies a smoother, differentiable limiting function that allows slight overshoots to improve numerical stability.

For strictly 1D configurations, directional limiters such as **Minmod** and **Superbee** are also available.

---

## 3. MUSCL & MOOD: Edge Polynomial Degree (EPD)

While MOOD order-dropping is managed globally by the `TimeStepper`, MUSCL is uniquely equipped to leverage the symmetric Edge Polynomial Degree ($EPD$) concepts defined in the original FV MOOD literature. 

### Decoupling Reconstruction and Divergence
In meshfree MUSCL, we uniquely split the polynomial logic into two independent operations to guarantee mathematical mass conservation:
1. **Reconstruction Order:** Varies dynamically based on the active local particle order ($d_i$) and the requested $EPD$ strategy. This controls the interface dissipation.
2. **Divergence Order:** Remains locked to a predefined `div_order` globally across all particles. By fixing the integration weights, exact mass conservation is maintained even when local particle orders degrade to resolve shocks.

### Meshfree EPD Strategies
The interface polynomial degrees used to evaluate the states between particle $i$ and neighbor $j$ are conceptualized as $(d_{i \to j}, d_{j \to i})$. The strategies map as follows:

*   **$EPD_0$:** The interface uses the exact local order: $(d_{i \to j}, d_{j \to i}) = (d_i, d_j)$. (Requires a 0-hop Halo strategy).
*   **Strict $EPD_0$:** Interface evaluation is identical to $EPD_0$. However, to preserve strict stability, if particle $i$ drops its order, it violently limits its neighborhood: it forcefully sets $d_j = \min(d_j, d_i)$ for all $j \in \nu(i)$, and flags both immediate and next-nearest neighbors for recalculation.
*   **$EPD_1$:** The interface uses a symmetric minimum of the interacting pair: $(d_{i \to j}, d_{j \to i}) = (\min(d_i, d_j), \min(d_i, d_j))$. (Requires a 1-hop Halo strategy to ensure neighbor states are re-evaluated).
*   **$EPD_2$:** Each particle precomputes an effective degree based on its neighborhood: $d_i^* = \min_{k \in \{i \cup \nu(i)\}}(d_k)$. The interface then uses these effective degrees: $(\min(d_i^*, d_j^*), \min(d_i^*, d_j^*))$. (Requires a 2-hop Halo strategy).

---

## 4. Implementation Details

The MUSCL architecture relies on a smart outer-constructor pattern and zero-allocation execution.

### MUSCL Construction (`MUSCL`)
The `MUSCL` constructor dynamically builds a dense, universally addressable tuple of `Interpolator` instances ranging from 1st-order (`ConstantReconstruction`) up to the requested `MAX_ORDER`. This allows the functor to hot-swap evaluation paths at runtime based on the locally degraded particle order without suffering from type-instability or dynamic dispatch overhead. 

### MUSCL Execution (The Functor)
*   **Pre-Gather Pass (`update_content!`):** Computes the raw unbounded gradients using the dynamic interpolator tuple. If the local neighborhood is starved (too few neighbors to support the active polynomial), it dynamically walks down the tuple until a stable order is found. If a traditional limiter is active, it limits the slopes here and caches them in the `gradients` buffer.
*   **Flux Pass:** Reconstructs the left and right states at the interface using the active $EPD$ strategy logic, evaluates the numerical flux (`F_num`), and calculates the physical non-conservative jump (`nc_jump`). The resulting raw flux differences are pushed directly through the locked `div_order` interpolator to obtain the final numerical divergence.

## 5. Documentation

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["Interpolation/MUSCL.jl","Interpolation/Limiter.jl"]
```
