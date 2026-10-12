# Numerical Flux Functions

Riemann-based schemes (Upwind Classic and MUSCL) treat the midpoint between particle $i$ and neighbour $j$ as an interface. They resolve the jump between the left and right states there with a numerical flux.

## Interface fluxes and the factor 2

For each neighbour, the scheme stores the difference $F_{\text{num},ij} - F(U_i)$ plus a non-conservative jump, and the MLS kernel differentiates it with respect to the neighbour offset. The interface sits at $x_i + x_{ij}/2$, so to first order

```math
F_{\text{num},ij} - F(U_i) \approx \tfrac12\, x_{ij} \cdot \nabla F ,
```

and the scheme multiplies the fitted derivative by 2. The central scheme differentiates $F(U_j) - F(U_i)$ directly and has no such factor.

Left and right are assigned per direction $d$ by the sign of the offset: if $x_{ij,d} > 0$, then $i$ is the left state and $j$ the right one for direction $d$ (`sort_flux`).

## Rusanov (local Lax–Friedrichs)

```math
F_{\text{num},d} = \tfrac12 \left( F_d(U_L) + F_d(U_R) - s_d\, (U_R - U_L) \right),
\qquad s_d = \max\big(\lambda_d(U_L), \lambda_d(U_R)\big),
```

with $\lambda_d$ from `max_eigenvalue(eq, U, d)`. It needs no eigen-decomposition and works for any system.

## Upwind

For scalar equations ($M = 1$) the dissipation coefficient is the Rankine–Hugoniot speed

```math
s_d = \left| \frac{F_d(U_R) - F_d(U_L)}{U_R - U_L} \right| ,
```

falling back to $|{\tt velocity}(eq, U_L, d)|$ when $|U_R - U_L| < 10^{-14}$. For systems ($M > 1$) a proper upwind flux would need a characteristic decomposition, so `UpwindFlux` dispatches to `RusanovFlux` instead.

## On moving grids

On a moving grid the stepper passes the PDE wrapped in an `ALEFrame`, whose flux is $F - wU$, whose `max_eigenvalue` is the shifted spectrum and whose `velocity` is the shifted Jacobian. Both fluxes therefore dissipate with the *relative* wave speed without any change to the flux code. For the full reduction in dissipation, the PDE should implement the four-argument `max_eigenvalue(eq, U, d, shift)` (see [Moving Grids (ALE)](@ref)).

## Writing your own

Subtype `NumericalFluxFunction` and implement

```julia
(::MyFlux)(f_L::Flux{D,M,T}, f_R::Flux{D,M,T}, F_L::Flux{D,M,T}, F_R::Flux{D,M,T}, eq::HyperbolicPDE{D,M,T}) -> Flux{D,M,T}
```

The arguments hold one left/right state and flux per direction `d`. Call only the PDE API on `eq` (no field access), so the flux keeps working inside an `ALEFrame`.

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["FluxFunctions.jl"]
Private = false
```
