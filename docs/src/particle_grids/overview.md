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

`get_time_step(pg, eq, interp)` returns the largest stable explicit step for CFL number 1:

```math
\Delta t = \min_i \frac{1}{D \sum_d \lambda_d(U_i)\, \sum_{j} \lvert c^{(d)}_{ij} \rvert },
```

where $\lambda_d$ is `max_eigenvalue(eq, U_i, d)` and $c^{(d)}_{ij}$ are the weights of the first-derivative operator of the MLS fit at particle $i$. The fit degree is `_divergence_degree(interp)`: the consistency order for Central/WENO/Upwind, and order − 1 for MUSCL. Degrees above 4 are clamped to 4. Multiply the result by your CFL number.

On moving grids this estimate uses the absolute wave speed $\lambda$, not the relative speed $\lvert\lambda - w\rvert$ seen by the scheme. See [Moving Grids (ALE)](@ref) for the consequences.

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["ParticleGrids.jl"]
Private = false
```
