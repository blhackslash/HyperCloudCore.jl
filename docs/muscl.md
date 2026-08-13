# Meshfree MUSCL Reconstruction & Limiting

This module implements a high-order Monotonic Upstream-centered Scheme for Conservation Laws (MUSCL) adapted for meshfree particle methods. It includes slope limiters and a robust Multi-dimensional Optimal Order Detection (MOOD) framework to ensure physical admissibility and prevent spurious oscillations near discontinuities.

## 1. MUSCL: From Finite Volume to Meshfree

In traditional grid-based Finite Volume (FV) methods, the MUSCL approach achieves higher-order spatial accuracy by reconstructing the solution within each cell using piecewise polynomials, rather than assuming a constant state. These reconstructed polynomials are then evaluated at the cell interfaces to compute high-order numerical fluxes.

In a **meshfree context**, the concept of a "cell" translates to a "particle," and "interfaces" are conceptualized symmetrically midway along the distance vector connecting two interacting neighbor particles. 

*   **Reconstruction:** Instead of cell-averaged gradients, we use Moving Least Squares (MLS) to compute high-order spatial derivatives directly at the particle locations. 
*   **Interface Evaluation:** The left and right states at the conceptual interface between particle $i$ and neighbor $j$ are reconstructed by evaluating the respective Taylor polynomials along the distance vector $\vec{x}_{ij}$.

To avoid non-physical oscillations (Gibbs phenomenon) when reconstructing across shocks or steep gradients, the scheme must be constrained. This is achieved either a priori (Slope Limiters) or a posteriori (MOOD).

---

## 2. Slope Limiting

Slope limiters operate *a priori* by directly suppressing the reconstructed gradients before interface states are evaluated. 

For robust arbitrary N-dimensional simulations, this implementation primarily relies on the **Barth-Jespersen (BJ)** and **Venkatakrishnan (VK)** limiters. 
*   Both methods enforce a monotonicity principle by ensuring that the reconstructed states at the conceptual interfaces do not exceed the maximum or minimum bounds defined by the particle's immediate neighbors.
*   While BJ strictly enforces these bounds (often hindering convergence to steady state), VK applies a smoother, differentiable limiting function that allows slight overshoots to improve numerical stability.

For strictly 1D configurations, directional limiters such as **Minmod** and **Superbee** are also available.

---

## 3. MOOD: Multi-dimensional Optimal Order Detection

Unlike traditional limiters, MOOD operates *a posteriori*. It computes candidate updates using the highest available polynomial order, checks if the resulting state is physically admissible, and locally reduces the polynomial order if it is not. 

### The Original Finite Volume Concept
According to the original literature, the MOOD method assigns a Cell Polynomial Degree (CellPD) to each finite volume cell, initialized to a maximum degree $d_{max}$. To evaluate fluxes at the cell edges, an Edge Polynomial Degree (EdgePD), denoted as $d_{ij}$, is determined based on the CellPDs of the adjacent cells. 

The original paper defines three primary strategies for determining the EdgePD:

*   **$EPD_0$:** The edge degree uses the cell's degree directly ($d_i$).
*   **$EPD_1$:** The edge degree is the minimum of the two sharing cells: $\min(d_i, d_j)$.
*   **$EPD_2$:** The edge degree is the minimum of all cells in the neighborhood of cell $i$: $\min_{j \in \nu(i)}(d_i, d_j)$.

Once candidate values $u_h^*$ are computed, they are checked against a Discrete Maximum Principle (DMP). If a cell violates the DMP, its CellPD is decremented: $d_i := \max(0, d_i - 1)$. This process iterates until all cells satisfy the criteria.

### MOOD Criteria: Theory and Admissibility
The admissibility of a candidate state is determined by a selected MOOD criterion, which evaluates whether the updated state violates physical bounds or exhibits spurious oscillations. The foundational rule of physical admissibility is the **Discrete Maximum Principle (DMP)**. It states that, in the absence of source terms, the updated state at a particle should not exceed the local minimum and maximum of the states in its immediate neighborhood from the previous time step.$$ \min_{j \in \nu(i) \cup \{i\}} u_j^n - \delta \le u_i^{*, n+1} \le \max_{j \in \nu(i) \cup \{i\}} u_j^n + \delta $$

where $u_i^{*, n+1}$ is the candidate updated state for particle $i$, $\nu(i)$ denotes the set of immediate spatial neighbors of particle $i$ at time level $n$, and $\delta \ge 0$ is a small relaxation parameter (typically set to a threshold like $\delta = 10^{-3}$ or scaled with local mesh size) to prevent unnecessary order reduction due to minor numerical noise. In our implementation we have two possible criterions:

*   **$U_1$ Criterion (Standard DMP):** This criterion enforces a strict DMP. It checks if the candidate state falls outside the local neighborhood extrema. A small relaxation parameter $\delta$ is applied so that minute numerical perturbations are ignored, but significant violations flag the particle for an order reduction.
*   **$U_2$ Criterion (MUSCL-Optimized):** A strict DMP often falsely flags smooth, physical extrema (like the peak of a smooth wave) as non-physical, degrading accuracy unnecessarily. The $U_2$ criterion first performs the $U_1$ DMP check; if it fails, it then analyzes the local magnitude and ratios of the spatial gradients (`findLocalExtremaAbs`). If the gradients behave smoothly and represent a physical peak, it permits the state, preserving high-order accuracy in smooth regions.

### Transition to the Meshfree Framework
In this meshfree implementation, the MOOD logic maps elegantly to particle interactions, forming a core component of the meshfree MUSCL scheme. Let $\nu(i)$ be the set of immediate neighbors of particle $i$, and $d_i$ be the polynomial order (`particle_orders`) assigned to particle $i$.

The interface polynomial degrees used to evaluate the states between particle $i$ and neighbor $j$ are conceptualized as $(d_{i \to j}, d_{j \to i})$. The strategies are mapped as follows:

*   **$EPD_0$:** The interface uses the particle's exact local order: $(d_{i \to j}, d_{j \to i}) = (d_i, d_j)$. If a particle drops its order, no halo updates are triggered for its neighbors (`trigger_halo!` is a no-op).
*   **Strict $EPD_0$:** Interface evaluation is identical to $EPD_0$. However, to preserve strict stability, if particle $i$ drops its order, it violently limits its neighborhood: it forcefully sets $d_j = \min(d_j, d_i)$ for all $j \in \nu(i)$, and flags both immediate neighbors and next-nearest neighbors for recalculation.
*   **$EPD_1$:** The interface uses a symmetric minimum of the interacting pair: $(d_{i \to j}, d_{j \to i}) = (\min(d_i, d_j), \min(d_i, d_j))$. If particle $i$ drops its order, all immediate neighbors $j \in \nu(i)$ are flagged for recalculation to ensure the symmetric interface is properly updated.
*   **$EPD_2$:** Each particle first computes a lowered effective degree based on its neighborhood: $d_i^* = \min_{k \in \{i \cup \nu(i)\}}(d_k)$. The interface then uses these effective degrees: $(\min(d_i^*, d_j^*), \min(d_i^*, d_j^*))$. Because the effective degree depends on a wider stencil, an order drop at particle $i$ triggers recalculations for both immediate and next-nearest neighbors.

---

## 4. Implementation details

The architecture is divided into three major components: the MUSCL constructor/functor, the Limiting logic, and the MOOD evaluation loop.

### MUSCL (`MUSCL`)
The `MUSCL` struct acts as a stateful divergence interpolator. 

*   **State Management:** It manages dynamic arrays such as `gradients`, `particle_orders`, and boolean flags for MOOD triggers (`mood_triggered`).
*   **Pre-Gather Pass:** The `update_content!` function computes the raw unbounded gradients using dynamic dispatch to the appropriate order's interpolator. If a traditional limiter is active, it limits the slopes here.
*   **Flux Pass:** The functor reconstructs the left and right states at the interface, evaluates the numerical flux (`F_num`), calculates the physical non-conservative jump (`nc_jump`), and feeds the resulting differences back into the interpolator to obtain the final numerical divergence.

### Slope Limiters (`AbstractSlopeLimiter`)
Limiters are implemented via the `_limit_slopes` function, which iterates over the immediate neighbor slice. 

*   It computes local extrema (`u_max`, `u_min`) and checks the unbounded reconstructed states.
*   It evaluates the limiter function `limiter_phi` (dispatched on the specific strategy like `BarthJespersenLimiter`) to find a scaling factor $\phi_i \in [0, 1]$.
*   The raw MLS gradients are then multiplied by this scaling factor to ensure they do not produce non-physical extrema at the interfaces.

### MOOD Evaluation (`evaluate_mood_and_halo!`)
The MOOD evaluation is handled asynchronously between RK stages. 

*   **Candidate Evaluation:** The function temporarily advances the particle states by applying the current RK (or IMEX) stage weights and the computed numerical divergence.
*   **Criteria Check:** It passes the candidate state to the specified `MOODCriterion` (e.g., `MOODu1` or `MOODu2`).
*   **Halo Propagation:** If a particle is flagged as invalid, its order is reduced, and `trigger_halo!` is dispatched based on the `MOODStrategy` to flag adjacent particles for recalculation in the next iteration of the divergence evaluation loop.
