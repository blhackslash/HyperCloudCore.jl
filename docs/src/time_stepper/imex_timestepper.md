# Implicit-Explicit (IMEX) Time Integration

This module implements a generalized, implicit-explicit (IMEX) space-time Runge-Kutta integrator for meshfree particle methods. It is designed to handle stiff equations, such as kinetic relaxation systems, by treating the non-stiff spatial divergence explicitly and the stiff source terms implicitly.

## 1. Mathematical Formulation (Semi-Discrete)

The governing system of equations for each particle $i$ is partitioned into two components. Let $U_i$ represent the state vector (e.g., the kinetic variables). The semi-discrete IMEX formulation is:

$$\frac{d U_i}{d t} = K_{E}(U_i) + K_{I}(U_i)$$

where $K_{E}$ represents the explicit spatial operator (the negative numerical divergence, $-\nabla \cdot F$), and $K_{I}$ represents the implicitly integrated source term, such as a kinetic relaxation operator. 

The IMEX integration relies on a double Butcher tableau: $\tilde{A}, \tilde{b}, \tilde{c}$ (explicit) and $A, b, c$ (implicit).

---

## 2. Algorithm Breakdown (`GeneralIMEXTimeStepper`)

The `GeneralIMEXTimeStepper` functor executes the integration over $s$ discrete stages.

### The Stage Loop
For each stage $k \in \{1, \dots, s\}$:

1. **Incremental Grid Movement (ALE):** 
   The grid is advected for an incremental time $\Delta t_k$. This step is defined by the explicit time nodes as $(c_{t, k+1} - c_{t, k}) \Delta t$, or $(1 - c_{t, k}) \Delta t$ for the final stage.
2. **Phase 1: Accumulate Stages (Explicit Prediction):** 
   An intermediate state $Y_i^{(k)}$ is assembled using the explicit and implicit evaluations from previous stages:
   $$ Y_i^{(k)} = U_i^n + \Delta t \sum_{j=1}^{k-1} \tilde{a}_{k, j} K_{E, j} + \Delta t \sum_{j=1}^{k-1} a_{k, j} K_{I, j} $$
3. **Phase 2: Non-Local Potential Update:** 
   If the source term is a `NonLocalRelaxationSourceTerm`, a macroscopic non-local potential is updated using `update_nonlocal_potential!` evaluated on the current intermediate states. This aggregates path integrals of the macroscopic states across the domain.
4. **Phase 3: Implicit Solve & Source Term Evaluation:** 
   If the diagonal implicit coefficient is non-zero ($a_{k,k} > 10^{-14}$), the intermediate state is updated by solving the implicit relation:
   $$ Y_i^{(k)} = \text{solve}(Y_i^{(k)}, \Delta t \cdot a_{k,k}, \dots) $$
   Once $Y_i^{(k)}$ is finalized, the stiff source term $K_{I, k}$ is evaluated.
5. **Phase 4: Explicit Spatial Derivatives:** 
   Physical boundary conditions are applied to $Y_i^{(k)}$, and the explicit spatial divergence $K_{E, k}$ is evaluated via `evaluate_stage_derivatives_imex!`. 

### Final Assembly
After all $s$ stages are evaluated:

1. **Final Step Update:** 
   The final state at $t^{n+1}$ is constructed using the stage weights $\tilde{b}$ and $b$:
   $$U_i^{n+1} = U_i^n + \Delta t \sum_{j=1}^s \tilde{b}_j K_{E, j} + \Delta t \sum_{j=1}^s b_j K_{I, j}$$
2. **Final Boundary Conditions:** 
   Boundary conditions are enforced on the final updated state.

---

## 3. Explicit Spatial Derivative Evaluation (`evaluate_stage_derivatives_imex!`)

The explicit divergence operator perfectly mirrors the direct RK solver architecture, including full support for multi-dimensional optimal order detection (MOOD).

For high-order `MUSCL` schemes, the function wraps the spatial evaluations in an iterative loop:

1.  **Phase 1 (Pre-Gather):** Evaluates local gradients only for particles flagged for recalculation (`needs_recalc`).
2.  **Phase 1.5:** Precomputes the effective minimal polynomial order within the neighborhood to preserve symmetric stencils.
3.  **Phase 2:** Reconstructs interface fluxes to yield the final explicit divergence $K_{E, k}$.
4.  **Phase 3 (MOOD & Halo Reduction):** Evaluates admissibility criteria. Particles failing the criteria drop their polynomial order and re-trigger calculations for themselves and their local halo neighbors until stability is achieved.