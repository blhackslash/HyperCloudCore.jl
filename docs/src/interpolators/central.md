# Central Divergence Scheme

This module implements a stateless, universal central difference scheme for meshfree particle methods. It is designed to compute the spatial divergence of the flux for hyperbolic partial differential equations across 1D, 2D, and 3D domains.

## 1. Mathematical Formulation (Discrete Form)

The primary goal of the central divergence scheme is to approximate the spatial divergence $\nabla \cdot F(U_i)$ at a given particle $i$. The discrete formulation evaluates the interaction between particle $i$ and its surrounding neighbors $j$ within a specified radius.

For each neighbor $j$ in the interaction stencil, the raw central flux difference is computed as:

$$\Delta F_{ij} = F(U_j) - F(U_i) + \Delta_{nc}(U_i, U_j, \vec{x}_{ij})$$

Where:
*   $F(U_i)$ and $F(U_j)$ are the conservative fluxes evaluated at particles $i$ and $j$, respectively.
*   $\Delta_{nc}$ represents the non-conservative jump term, which is evaluated based on the states and the distance vector between the particles.

Once the flux differences $\Delta F_{ij}$ are calculated for all neighbors in the stencil, they are passed directly as a vector to a stateless matrix-vectorized interpolator. The interpolator uses the local particle topology—specifically the distances and kernel weights—to reconstruct the full spatial divergence. 

## 2. Implementation Details (`CentralDivergence`)

The `CentralDivergence` functor is built to natively support systems of equations via the `State{M, T}` type. 

### API and State Management
*   **Stateless API:** The central divergence method is entirely stateless. Consequently, the `update_size!` and `update_content!` functions are implemented as no-ops (`nothing`), requiring no memory allocations or caching between Runge-Kutta stages.
*   **Order Requirements:** The scheme requires a minimum interpolation order of 1 (`order >= 1`). 
*   **Neighbor Checking:** Before computing the flux differences, the functor checks the number of available neighbors. If the number of neighbors is strictly less than the required interpolation order, the divergence gracefully falls back to zero.

### Execution Flow
1.  **Local Flux Evaluation:** The conservative flux $F_i$ for the target particle is evaluated once.
2.  **Neighbor Loop:** The algorithm iterates over a provided `nb_slice`, calculating the physical jump and accumulating the raw flux difference directly into a mutually-exclusive slot in the interaction buffer (`ib.df_flux`).
3.  **Interpolation:** The accumulated flux differences, along with pre-calculated distances and weights, are fed into the interpolator (scaled by `dx`) to yield the final numerical divergence.