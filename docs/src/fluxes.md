# Numerical Flux Functions

This module provides the numerical flux functions used to resolve the Riemann problem at the conceptual interfaces between interacting particles. It currently supports the robust Rusanov (Local Lax-Friedrichs) flux for general N-dimensional systems, as well as an Upwind flux optimized for scalar equations.

## 1. Theoretical Background

In finite volume and meshfree MUSCL schemes, reconstructing the left and right states ($U_L$ and $U_R$) at an interface creates a discontinuous jump. Numerical flux functions compute a stable, single flux value $F_{num}$ across this discontinuity by introducing necessary upwinding or numerical dissipation based on the characteristic wave speeds.

### Rusanov (Local Lax-Friedrichs) Flux
The Rusanov flux is a universally applicable approximate Riemann solver. It does not require a full eigendecomposition of the system's Jacobian, making it highly robust and easily extensible to any N-dimensional system. It stabilizes the central flux average by adding a diffusive term scaled by the maximum local wave speed.

$$F_{num} = \frac{1}{2} \left( F_L + F_R - s (U_R - U_L) \right)$$

Where $s = \max(|\lambda_L|, |\lambda_R|)$ is the maximum characteristic wave speed evaluated from the left and right states.

### Upwind Flux
The Upwind flux traces the flow of information along characteristics. For scalar conservation laws, it perfectly captures the correct physical direction without adding excessive numerical dissipation. The wave speed is determined by the Rankine-Hugoniot jump condition:

$$s = \left| \frac{F_R - F_L}{U_R - U_L} \right|$$

If the denominator approaches zero, the speed defaults to the local advection velocity.

---

## 2. Implementation Details

*   **`RusanovFlux`:** Computes the numerical flux for every dimension $d$ by applying the maximum localized wave speed $s^d$ as the dissipation coefficient.
*   **`UpwindFlux`:** 
    *   **Scalar Execution:** Computes the Rankine-Hugoniot wave speed $s$. To avoid division by zero, if the state jump $\Delta u < 10^{-14}$, it falls back to the absolute advection velocity.
    *   **System Fallback:** Because a pure Upwind scheme requires a full Roe matrix or characteristic decomposition for systems, calling `UpwindFlux` on any system ($M > 1$) gracefully and automatically falls back to executing the `RusanovFlux`.