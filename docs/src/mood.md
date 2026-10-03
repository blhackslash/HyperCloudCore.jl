# Multi-Dimensional Optimal Order Detection (MOOD)

This module implements the Multi-dimensional Optimal Order Detection (MOOD) framework. MOOD is an *a posteriori* limiting strategy designed to stabilize high-order meshfree particle methods without the excessive numerical dissipation often introduced by traditional *a priori* slope limiters.

## 1. The MOOD Concept

Unlike traditional limiters that blindly suppress gradients before calculating fluxes, MOOD operates on a "calculate, check, and fallback" philosophy:
1.  **Calculate:** Compute a full candidate update for the particle state using the highest available polynomial order.
2.  **Check:** Evaluate the candidate state against strict physical admissibility criteria (e.g., positivity, Discrete Maximum Principle).
3.  **Fallback:** If a particle's candidate state violates the physical bounds, its local polynomial order is decremented, and its update is recalculated.

This guarantees that smooth regions retain their maximum high-order accuracy, while shocks or steep gradients gracefully degrade to stable, lower-order (or piecewise constant) evaluations.

---

## 2. Admissibility Criteria (`MOODCriterion`)

The admissibility of a candidate state is evaluated by a selected `MOODCriterion`. The foundational rule of physical admissibility is the **Discrete Maximum Principle (DMP)**. It states that, in the absence of source terms, the updated state at a particle should not exceed the local minimum and maximum of the states in its immediate neighborhood from the previous time step.

$$\min_{j \in \nu(i) \cup \{i\}} u_j^n - \delta \le u_i^{*, n+1} \le \max_{j \in \nu(i) \cup \{i\}} u_j^n + \delta$$

where $u_i^{*, n+1}$ is the candidate updated state for particle $i$, $\nu(i)$ denotes the set of immediate spatial neighbors, and $\delta \ge 0$ is a small relaxation parameter to prevent unnecessary order reduction due to minor numerical noise. 

We provide two primary criteria:
*   **$U_1$ Criterion (Standard DMP):** Enforces a strict DMP. A small relaxation parameter $\delta$ is applied so that minute numerical perturbations are ignored, but significant violations flag the particle for an order reduction. Ideal for stateless interpolators like `Upwind`.
*   **$U_2$ Criterion (Relaxed/Optimized DMP):** A strict DMP often falsely flags smooth, physical extrema (like the peak of a smooth wave) as non-physical, degrading accuracy unnecessarily. The $U_2$ criterion first performs the $U_1$ check; if it fails, it analyzes the local magnitude and ratios of the spatial gradients. If the gradients behave smoothly and represent a physical peak, it permits the state, preserving high-order accuracy.

---

## 3. Meshfree Architecture and Halo Strategies (`MOODStrategy`)

In this meshfree architecture, the MOOD logic is managed globally by the `TimeStepper` and tracks polynomial limits centrally within `ParticleGridCore`. This completely decouples the MOOD loop from any specific spatial scheme.

### The MOOD Evaluation Loop
During each explicit Runge-Kutta or IMEX stage, the `TimeStepper` orchestrates the following iterative loop:
1.  **Initialization:** At the first stage, all particle polynomial orders are reset to the scheme's `MAX_ORDER`. A boolean buffer (`needs_recalc`) flags all particles for calculation.
2.  **Evaluation:** The divergence interpolator (and explicit sources) evaluates the spatial updates for the flagged particles using their active polynomial order.
3.  **Candidate Synthesis:** A full prospective candidate state is assembled locally.
4.  **Criteria Check & Halo Reduction:** The `MOODCriterion` evaluates this candidate state. If physical bounds are violated, the local particle drops its polynomial order (provided it is $> 1$).
5.  **Halo Triggering:** When a particle drops its order, it often invalidates the evaluations of its neighbors. A `MOODStrategy` dictates how far this invalidation spreads (the "Halo" effect), flagging those specific neighbors in `needs_recalc`.
6.  **Termination:** The loop repeats until all particles are admissible or safely constrained to 1st order.

### Halo Strategies (`Halo{Force_N, Recalc_N}`)
The `MOODStrategy` defines the topological spread of order drops and recalculations using a two-parameter Halo abstraction:
*   **`Force_N` (Order Forcing Depth):** The topological depth (in hops) to which a particle dropping its order aggressively forces its neighbors to also drop their structural order.
*   **`Recalc_N` (Recalculation Depth):** The topological depth to which particles are flagged for recalculation in the next phase.

This directly maps the classical Edge Polynomial Degree (EPD) concepts to a meshfree topology:
*   **`EPD0` (`Halo{0, 0}`):** Strictly local. If a particle drops its order, no neighbors are affected or flagged.
*   **`EPD1` (`Halo{0, 1}`):** 1-Hop Recalculation. If a particle drops its order, it doesn't force its neighbors' structural orders down, but its immediate neighbors are flagged to recalculate (essential for symmetric interfaces).
*   **`EPD2` (`Halo{0, 2}`):** 2-Hop Recalculation. Recalculation flags cascade to immediate and next-nearest neighbors.
*   **`StrictEPD0` (`Halo{1, 2}`):** Aggressive local limiting. If a particle drops its order, it forcefully pulls all immediate neighbors down to match. Because the neighbors' orders changed, recalculation flags cascade out to 2 hops.

## 4. Documentation

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["TimeStepping/MOOD.jl"]
Private = false
```