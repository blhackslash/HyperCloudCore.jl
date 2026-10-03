# Meshfree Upwind Algorithms

This module implements various upwind divergence algorithms adapted for meshfree particle methods. Upwind schemes are essential for stabilizing hyperbolic equations by ensuring that the numerical discretization respects the physical direction of wave propagation.

## 1. Upwind Schemes in a Meshfree Context

Unlike grid-based methods that evaluate fluxes at discrete cell interfaces, meshfree upwinding operates over a continuous neighbor cloud. To accommodate different PDE types and dimensionality, the framework provides three distinct algorithmic approaches:

*   **Classic Algorithm:** Natively supports general systems of equations across 1D, 2D, and 3D geometries.
*   **Tiwari Algorithm:** Exclusively restricted to scalar PDEs. It utilizes directional masking to dynamically construct an upwind stencil based on local velocity.
*   **Praveen Algorithm:** Restricted to 1st-order scalar PDEs and specifically tailored for 2D spatial domains.

---

## 2. Theoretical Formulation

The formulations vary depending on the chosen algorithm type:

**Classic Algorithm**
The central particle evaluates pairwise numerical fluxes $F_{num}$ and non-conservative jumps with each neighbor in its interaction radius. The spatial divergence is obtained by passing these flux differences to the underlying interpolator and applying a global scaling factor:

$$\nabla \cdot F = 2.0 \times \text{Interpolator}(F_{num} - F_i + \text{nc\_jump})$$


**Tiwari Algorithm**
The scheme identifies the upwind direction dimension-by-dimension. A neighbor particle is included in the masked upwind stencil only if its spatial offset $\Delta x_d$ opposes the direction of the local velocity $v_d$:

$$v_d \Delta x_d \le 0$$

The final divergence for each dimension is the interpolated spatial derivative multiplied by the local scalar velocity.

**Praveen Algorithm**
In 2D space, the algorithm projects the interaction vectors into local normal ($n_x, n_y$) and shear ($s_x, s_y$) components. It limits downstream influence using thresholding minimum functions:

$$\min(v_n, 0)$$

Interactions are weighted by evaluating a local moment matrix $N_s$ and scaled by a factor of 2.0.

---

## 3. Implementation Details (`UpwindDivergence`)

The `UpwindDivergence` struct provides a strictly stateless execution API where the `update_size!` and `update_content!` methods act as no-ops. During construction, the factory asserts that the chosen interpolation order is at least 1. 

### Execution Flows

*   **Classic Algorithm (`ClassicAlgorithm`):**
    *   Evaluates the analytical flux $F_i$ for the primary central particle.
    *   Loops over the active neighbor slice, sorting states to compute the interface numerical flux and any non-conservative jumps.
    *   Returns a zero state if the number of available neighbors is strictly less than the requested interpolation order.
    *   Passes the accumulated flux differences to the Moving Least Squares (MLS) interpolator and multiplies the final result by 2.0.
*   **Tiwari Algorithm (`TiwariAlgorithm`):**
    *   Asserts during initialization that the PDE is purely scalar (`M == 1`).
    *   Loops over each spatial dimension independently, dynamically masking out particles that lie in the downwind direction.
    *   If the resulting populated upwind stencil has fewer neighbors than the required interpolation order, it immediately returns a zero state for that dimension.
    *   Executes the interpolator on the masked interaction buffer to compute the spatial derivative and multiplies it by the scalar velocity.
*   **Praveen Algorithm (`PraveenAlgorithm`):**
    *   Enforces strict requirements during initialization: the spatial dimension must be exactly 2, the PDE must be scalar (`M == 1`), and the interpolation order must be exactly 1.
    *   Assembles a 2D velocity vector and constructs an inverse moment matrix $N_s$ scaled by the local particle spacing.
    *   Automatically falls back to returning a zero state if the neighbor count drops below 3, if the local spacing scale is below `1e-14`, or if the determinant of $N_s$ is less than `1e-14`.
    *   Evaluates downwind-rejecting minimum bounds on the normal and shear velocities, accumulates the stabilized divergence, and scales the final tensor by 2.0.


 ## 4. Documentation

```@docs
UpwindDivergence
```