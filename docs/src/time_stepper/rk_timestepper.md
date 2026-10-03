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
   The spatial divergence $K_k = \nabla \cdot F(U^{(k)})$ is computed by dispatching to the `evaluate_stage_derivatives!` routine.

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

The calculation of the spatial divergence $K$ handles both standard stateless evaluations and dynamic, iterative methods globally orchestrated by the `TimeStepper`.

### Standard Evaluation (Generic / Non-MOOD)
For methods executing without an active MOOD configuration, the evaluation requires exactly two parallel passes over the particle grid:

*   **Phase 1 (Pre-Gather):** Iterates over all interior particles to compute and cache localized data (e.g., slopes, basis functions, or moving-least-squares gradients) via `update_content!`.
*   **Phase 2 (Divergence):** Iterates over the grid a second time to construct interface fluxes and compute the final divergence $K_k$.

### MOOD Evaluation (Multi-Dimensional Optimal Order Detection)
When configured with an active `MOOD` strategy, the derivative evaluation is wrapped in a dynamic loop to support per-particle order degradation. Polynomial limits are tracked globally within `ParticleGridCore`, decoupling the MOOD logic from any specific spatial scheme:

1.  **Initialization:** At the first RK stage ($stage = 1$), all particle polynomial orders are reset to the scheme's maximum order (`MAX_ORDER`). A boolean buffer (`needs_recalc`) flags all particles for calculation.
2.  **Phase 1 (Pre-Gather):** Gradients and internal states are computed only for particles flagged in `needs_recalc`.
3.  **Phase 2 (Divergence):** Interface fluxes and the resulting divergence $K_k$ are calculated for the flagged particles using their active polynomial order.
4.  **Phase 3 (Candidate Evaluation & Halo Propagation):** A full prospective candidate state is assembled and passed to the `MOODCriterion`. If a particle violates admissibility (e.g., negative density or non-physical oscillations) **and** its current order is $> 1$, its polynomial order is decremented. The particle and its spatial neighbors (dictated by the `MOODStrategy`) are flagged for recalculation.
5.  **Termination:** The loop continues until no new order drops occur—either because all particles satisfy the physical criteria, or the problematic particles have gracefully bottomed out at order 1 (piecewise constant).

## 4. Documentation

### General Timestepper

```@docs
GeneralRKTimeStepper
```

### Butcher Tableaus

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["TimeStepping/RKButcherTableaus.jl"]
```