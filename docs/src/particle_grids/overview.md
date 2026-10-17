# Particle Grid Overview

A `ParticleGrid` holds everything that describes *where* the solution lives: the particles and their flags, the neighbour lists with MLS weights and distances, the per-particle consistency order, the state vector `rhos`, and an optional grid mover. Solver state that only matters during a time step (stage buffers, MOOD bookkeeping) lives in the time stepper instead.

## Construction

```julia
pg = ParticleGrid(geom, nominal_dx, weight_func, M;
                  randomness = ntuple(_ -> 0.0, D),
                  rng        = Random.default_rng(),
                  mover      = NoGridMover())
```

| Argument | Meaning |
|---|---|
| `geom::GeometricDomain` | Shape, periodicity, boundary tags and boundary-condition map (see [Geometric Domains](@ref)). |
| `nominal_dx::NTuple{D,T}` | Spacing of the Cartesian seed lattice per axis. |
| `weight_func::MLSWeightFunction` | Kernel for the MLS fits; its cutoff `get_cutoff(weight_func)` is the neighbour radius `R`. |
| `M::Int` | Number of state components. |
| `randomness` | Per-axis amplitude of a uniform jitter added to every lattice point. |
| `mover::GridMover` | Grid velocity model; `NoGridMover()` keeps the particles fixed (see [Moving Grids (ALE)](@ref)). |
| `volume_estimator::VolumeEstimator` | How `pg.core.volumes` is computed; `ShepardVolumes()` by default (see [Particle Volumes and Adaptivity](@ref)). |

The constructor generates the point cloud, runs a neighbour search, reorders the particles along a space-filling curve, searches again on the reordered cloud, and computes the particle volumes.

## Internal structure

| Field | Type | Contents |
|---|---|---|
| `pg.meta` | `GridMetadata` | `N`, `N_interior`, `N_ghost`, cutoff `R`, `dx`, largest neighbourhood `max_nb` |
| `pg.core` | `ParticleGridCore` | `positions`, `is_boundary`, `tags`, `volumes` (partition-of-unity weights, see [Particle Volumes and Adaptivity](@ref)), `particle_orders` |
| `pg.neighbor` | `NeighborData` | Weight function and CSR-style lists: `ranges[i]` indexes `indices`, `weights`, `distances` |
| `pg.bins` | `GlobalBins` | Coarse spatial hash (bin size ≥ `R`) and linked lists for the search |
| `pg.reorder` | `ReorderData` | Permutation buffers for the Morton reordering |
| `pg.shared` | `SharedBuffers` | Scratch vectors for grid-level routines (BCs, neighbour search, CFL) |
| `pg.mover` | `GridMover` | Grid velocity model |
| `pg.rhos` | `Vector{State{M,T}}` | The solution |

`particle_orders` is the only solver-related field on the core. Every spatial scheme reads it, and the time stepper's MOOD loop writes it. All other MOOD state (trigger flags, recompute masks, order snapshots) lives in the time stepper's `MOODWork`.

The buffers grow by a factor 1.25 when more capacity is needed (`ensure_capacity!`) and never shrink. Only the first `pg.meta.N` entries of the particle arrays, and the first `last(pg.neighbor.ranges[N])` entries of the neighbour arrays, are meaningful.

## Neighbour search

`pg.neighbor(pg)` rebuilds the topology:

1. Wraps positions back into the primary cell on periodic axes.
2. Hashes particles into coarse bins of edge length ≥ `R`.
3. **Counting pass** (parallel): counts neighbours within `R` in the $3^D$ surrounding bins.
4. Prefix sum to obtain `ranges`.
5. **Writing pass** (parallel): stores neighbour index, weight `w(|x_{ij}|^2)` and the shortest (minimum-image) distance vector $x_j - x_i$.

A particle is never its own neighbour. For moving grids, `refresh_geometry!` recomputes distances and weights for the *existing* lists without changing the topology; pairs that have drifted beyond `R` get weight zero.

## Reordering

`pg.reorder(pg)` sorts particles along a Morton Z-curve (2-D/3-D) or by coordinate (1-D) to improve cache locality, then permutes `positions`, `is_boundary`, `tags`, `particle_orders`, `volumes`, `rhos` and `curvatures` in place. It is skipped when the permutation is already the identity. Reordering invalidates the neighbour lists and every per-particle buffer held by a time stepper, so call it **between** time steps and run `pg.neighbor(pg)` afterwards.

## CFL time step

`get_time_step(pg, eq, interp)` returns the time step for CFL number 1:

```math
\Delta t = \min_i \frac{1}{D \sum_d \lambda_d(U_i)\, \sum_{j} \lvert c^{(d)}_{ij} \rvert },
```

where $\lambda_d$ is `max_eigenvalue(eq, U_i, d)` and $c^{(d)}_{ij}$ are the weights of the first-derivative operator of the MLS fit at particle $i$. The fit degree is `_divergence_degree(interp)`: the consistency order for Central/WENO/Upwind, and order − 1 for MUSCL (or `MLS_order` − 1 if that is set). Degrees above 4 (3 in 3-D) are clamped. Multiply the result by your CFL number.

### What the CFL number means on a point cloud

The number is **not** a classical Courant number $\lambda \Delta t / \Delta x$. On a mesh the update of a cell involves its faces, and $\Delta x$ is the distance to them. Here the update of particle $i$ involves every neighbour $j$ in the support radius, each through its own MLS weight, and the role of $1/\Delta x$ is played by the $\ell^1$ norm $\sum_j \lvert c^{(d)}_{ij} \rvert$ of the derivative stencil. That norm depends on the support radius, the weight function, the local point distribution and the fit degree, so the time step adapts to all of them per particle.

Where the bound comes from: the schemes evaluate the divergence as $2 \sum_j c_{ij} \cdot (F^*_{ij} - F_i)$ with a numerical flux $F^*$. For the first-order scheme with the Rusanov flux, one forward Euler step can be written as a convex combination of $U_i$ and states of the form $U_j \mp F_d(U_j)/\lambda_d$ (which are admissible for $\lambda_d$ at least the wave speed) as long as

```math
\Delta t \sum_d \lambda_d \sum_j \lvert c^{(d)}_{ij} \rvert \le 1 ,
```

i.e. **CFL ≤ D**. Then the step preserves invariant domains (positivity of density and pressure, the maximum principle for scalar laws), and SSP Runge-Kutta methods inherit the bound. The argument assumes $\sum_j c_{ij} = 0$ and $\operatorname{sign} c^{(d)}_{ij} = \operatorname{sign}(x_j - x_i)_d$ (true on regular, nearly symmetric stencils) and wave speeds of the neighbours not larger than that of particle $i$; it is a sufficient condition, not a sharp stability limit. The extra factor $1/D$ in the formula above keeps CFL 1 inside the bound in every dimension.

Consequences:

- **The scale differs from a Courant number.** On a 2-D jittered grid with support radius 2.5 Δx and a linear fit, CFL 1 corresponds to $\lambda_d \Delta t / \Delta x \approx 0.3$ per direction (measured on the Euler quadrant and double Mach reflection problems).
- **Higher fit degrees give smaller steps at the same CFL number**: the derivative stencil of a cubic fit has a larger $\ell^1$ norm than that of a linear fit (about 2.7× in 2-D with support radius 5.5 Δx), so MUSCL4 takes about 2.7× as many steps as MUSCL2 at the same CFL number.
- **Larger support radii give larger steps**: the stencil averages over more distant neighbours (Euler quadrant problem, 400², linear fit: 6474 steps with radius 2.5 Δx, 2870 with 5.5 Δx at the same CFL number).

### Measured limits

RK2 (SSP) and RK4, Float32/Float64, from the HyperCloud.jl example problems (2026-10):

| Problem | Schemes | Runs at | Fails at |
|---|---|---|---|
| 2-D Euler quadrant problem, 400², radius 2.5 Δx | RK2 Upwind1, MUSCL2 (VK, MOOD EPD1/EPD2) | 3.0 | 4.0 |
| 2-D double Mach reflection, 960×300 | same | 2.0 (highest tried) | – |
| 2-D forward-facing step, 600×200 | RK2 MUSCL2 (VK, MOOD EPD1) | 0.8 (highest tried) | – |
| 1-D Sod, 330 particles, radius 5.5 Δx | RK2 Upwind1, MUSCL2 (MOOD, VK, minmod); RK4 MUSCL4 MOOD | 1.2 | 1.5 (VK and minmod still run) |
| 1-D linear advection, smooth | RK4 MUSCL2 … MUSCL5 | 1.5 (MUSCL2) … 3.0 (MUSCL5) | 2.0 … 4.0 |

On the 2-D shock problems, the solutions at CFL 1 differ from those at CFL 0.2 by 1e-5 (limiters, first order) to a few 1e-3 (MOOD, whose order decisions shift) in relative L1 norm of the density, far below the discretisation error. CFL 0.5 – 1 is a reasonable default for shock problems in 2-D; in 1-D stay at or below 1.

On moving grids this estimate uses the absolute wave speed $\lambda$, not the relative speed $\lvert\lambda - w\rvert$ seen by the scheme. See [Moving Grids (ALE)](@ref) for the consequences.

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["ParticleGrids.jl"]
Private = false
```
