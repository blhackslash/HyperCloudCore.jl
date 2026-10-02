# Explicit Runge-Kutta Time Integration

This module implements a generalized, explicit, space-time Runge-Kutta (RK) time integrator for meshfree particle methods.

## 1. Mathematical Formulation (Semi-Discrete)

The governing hyperbolic partial differential equations are discretized in space, yielding a system of ordinary differential equations for each particle $i$. Let $U_i$ represent the state vector of particle $i$. The semi-discrete form is given by:

$$\frac{d U_i}{d t} = - K_i(U)$$

where $K_i(U)$ represents the spatial divergence operator $\nabla \cdot F(U_i)$ evaluated by the underlying interpolator, alongside any relevant source terms.

Given an $s$-stage Runge-Kutta method defined by its Butcher tableau coefficients ($A$, $b$, $c$), the integration from time level $n$ to $n+1$ is performed in discrete stages.

---

## 2. Algorithm Breakdown (`GeneralRKTimeStepper`)

The `GeneralRKTimeStepper` functor executes the explicit integration loop over $s$ stages.

### The Stage Loop
For each stage $k \in \{1, \dots, s\}$:

1. **Incremental Grid Movement (ALE):**
   If the grid is moving, the particles are advected for the incremental time $\Delta t_k = (c_k - c_{k-1}) \Delta t$. For the first stage, this is $c_1 \Delta t$.
2. **Intermediate State Calculation:**
   The intermediate state $U_i^{(k)}$ is assembled using the divergence values $K_j$ computed in the previous stages:

   $$U_i^{(k)} = U_i^n - \Delta t \sum_{j=1}^{k-1} A_{k, j} K_j$$
   
   *(Note: The negative sign arises because $K$ represents the numerical divergence $\nabla \cdot F$)*.
3. **Boundary Conditions:**
   Physical boundary conditions are immediately applied to the intermediate state $U_i^{(k)}$.
4. **Spatial Derivative Evaluation:**
   The spatial divergence $K_k = \nabla \cdot F(U^{(k)})$ is computed by dispatching to `evaluate_stage_derivatives!`.

### Final Assembly
After all $s$ stages are evaluated:

1. **Final Grid Movement:**
   The grid is moved for the remaining fractional timestep $(1 - c_s) \Delta t$, and the neighbor lists are updated to reflect the final particle topology.
2. **State Update:**
   The final state at $t^{n+1}$ is constructed using the stage weights $b$:

   $$U_i^{n+1} = U_i^n - \Delta t \sum_{j=1}^s b_j K_j$$

3. **Final Boundary Conditions:**
   Boundary conditions are applied to the finalized state vector.

---

## 3. Spatial Derivative Evaluation (`evaluate_stage_derivatives!`)

The calculation of the spatial divergence $K$ uses multiple dispatch to seamlessly handle standard methods (like Upwind or WENO) as well as dynamic, iterative methods (like MOOD).

### Standard Evaluation (Generic / Non-MOOD)
For fixed-stencil methods, the evaluation requires exactly two parallel passes over the particle grid:

*   **Phase 1 (Pre-Gather):** Iterates over all interior particles to compute and cache localized data, such as slopes or moving-least-squares gradients.
*   **Phase 2 (Divergence):** Iterates over the grid a second time to construct interface fluxes and compute the final divergence $K_k$.

### MOOD Evaluation (Multi-Dimensional Optimal Order Detection)
For high-order `MUSCL` schemes equipped with a `MOOD` strategy, the derivative evaluation is wrapped in a `while true` loop (capped at 20 iterations) to allow for dynamic order-dropping:

1.  **Initialization:** At the very first RK stage ($stage = 1$), all particles are reset to the maximum polynomial order (`MAX_ORDER`). A boolean buffer (`needs_recalc`) flags all particles as requiring calculation.
2.  **Phase 1 (Pre-Gather):** Gradients are computed only for particles flagged in `needs_recalc`.
3.  **Phase 1.5 (Effective Order Precomputation):** To guarantee symmetric stencils (the Halo effect), every particle determines its effective order by querying the minimum polynomial order among its direct neighbors.
4.  **Phase 2 (Divergence):** Interface fluxes and the resulting divergence $K_k$ are calculated for the flagged particles.
5.  **Phase 3 (MOOD & Halo Reduction):** The updated candidate states are passed to the MOOD evaluator. If a particle violates admissibility criteria (e.g., negative density or oscillations), its polynomial order is decremented, and it—along with its halo neighbors—is flagged for recalculation. The loop continues until all particles are admissible or the iteration cap is reached.
