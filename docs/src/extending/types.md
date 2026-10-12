```@meta
CurrentModule = HyperCloudCore
```

# Abstract Types

## Static array aliases

Physical quantities are static arrays with fixed sizes, which keeps every particle loop allocation-free:

```@docs
Space
State
Flux
Velocity
```

| Alias | Shape | Meaning |
|---|---|---|
| `Space{D,T}` | `SVector{D,T}` | Position, distance or velocity vector |
| `State{M,T}` | `SVector{M,T}` | State of one particle |
| `Flux{D,M,T}` | `SVector{D, State{M,T}}` | One flux (or one state) per spatial direction |
| `Velocity{D,M,T,L}` | `SVector{D, SMatrix{M,M,T,L}}` | Flux Jacobian per direction |

## Extension points

| Abstract type | Implement it for | Contract |
|---|---|---|
| `HyperbolicPDE{D,M,T}` | a new equation system | [API Contracts](@ref) |
| `AbstractExplicitSourceTerm`, `AbstractImplicitSourceTerm` | source terms | [API Contracts](@ref) |
| `DivergenceInterpolator` | a new spatial scheme | [API Contracts](@ref) |
| `NumericalFluxFunction` | a new Riemann solver | [Numerical Flux Functions](@ref) |
| `AbstractSlopeLimiter` | a new MUSCL limiter | `_limit_slopes` (see `Limiter.jl`) |
| `MOODCriterion`, `MOODStrategy` | admissibility tests, interface-degree rules | [Multi-Dimensional Optimal Order Detection (MOOD)](@ref) |
| `GridMover` | grid velocity models | [Moving Grids (ALE)](@ref) |
| `AbstractBoundaryCondition` | boundary conditions | [Boundary Conditions](@ref) |
| `MLSWeightFunction` | MLS kernels | [MLS Weight Functions](@ref) |
| `TimeStepper` | a different time integrator | functor `(ts)(eq, pg, t, dt)` |

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["Types.jl"]
```
