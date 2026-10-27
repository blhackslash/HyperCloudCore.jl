# Particle Volumes and Adaptivity

Collocation particles carry no mass. To define a discrete conserved quantity, and to make inserting and merging particles exactly conservative, every particle gets a positive weight $V_i$ (stored in `pg.core.volumes`) with

```math
\sum_i V_i = |\Omega|, \qquad \sum_i V_i\, f(x_i) \approx \int_\Omega f\,\mathrm dx .
```

No Voronoi tessellation is built or stored. $V_i = \int_\Omega \phi_i\,\mathrm dx$ for shape functions $\phi_i$ that form a partition of unity ($\sum_i \phi_i \equiv 1$), evaluated with a quadrature on the point cloud itself or on a throwaway lattice. Only the scalar $V_i$ is kept.

## Volume estimators

| Estimator | Shape functions | Evaluation | Quadrature order | Cost |
|---|---|---|---|---|
| `ShepardVolumes(; sweeps = 1)` | $\phi_i(x) = w(\lvert x - x_i\rvert) / \sum_k w(\lvert x - x_k\rvert)$ | particle-based, no lattice | 2 on quasi-uniform clouds; biased where the spacing varies | one neighbour sweep per `sweep` (kernel) |
| `MLSVolumes(P; samples_per_dx = 2)` | MLS reproducing-kernel functions of degree $P$ | background lattice over $\Omega$ | $P + 1$ on any cloud | lattice points × neighbours (serial, host) |

**Shepard** starts from the kernel-density volume $V_i \propto 1/S_i$, $S_i = w(0) + \sum_j w_{ij}$, applies `sweeps` partition-of-unity sweeps $V_i \leftarrow \sum_j (w_{ij}/S_j)\,V_j$ (which preserve $\sum V$), and renormalises to $|\Omega|$. On a uniform periodic cloud it reproduces $V_i = \Delta x^D$ exactly. It needs a kernel that is bounded at zero distance; with `InverseWeightFunction` the volumes fall back to $|\Omega|/N$ with a warning.

$S_i$ is a lattice sum over a fixed number of neighbours, which is not proportional to the local particle density when the spacing varies. On a moving (stretched) cloud the Shepard quadrature therefore has a bias that depends on the local stretch and does **not** shrink with the resolution: for a smooth profile on a 1-D grid moved by a sinusoidal velocity of amplitude 0.5, $\sum_i V_i u(x_i)$ of the *exact* solution is off by $-2\cdot 10^{-3}$ at every $N$ from 200 to 1600, while MLS volumes of degree 2 converge ($-2\cdot 10^{-5} \to 5\cdot 10^{-9}$). Mass errors measured with Shepard volumes on a moving grid contain that bias.

**MLS** evaluates, at every lattice point $y$, the shape functions $\phi_i(y) = w_i\, p_i^{\mathsf T} a$ with $\big(\sum_k w_k p_k p_k^{\mathsf T}\big) a = e_1$ and the full basis $p_k = (1, \text{monomials}(x_k - y))$. Because $\sum_i \phi_i(y)\, q(x_i) = q(y)$ for polynomials $q$ of degree $\le P$, the volumes integrate such polynomials exactly (up to the lattice's midpoint error). `P = 0` is the lattice-integrated Shepard partition. Shape functions of degree $\ge 1$ can be negative; a volume that comes out negative is clamped before the final renormalisation. The lattice sweep is a serial host loop: as the per-step re-estimation of a moving grid it costs about 3 (1-D, 1600 particles) to 21 (2-D, $141^2$) explicit MUSCL2/RK2 steps on 8 threads, so it is not the default. On a device grid it runs on a host copy (only the volumes are copied back).

```julia
pg = ParticleGrid(geom, dx, wf, M; volume_estimator = MLSVolumes(2))   # at construction
update_volumes!(pg, ShepardVolumes())                                   # any later time
total_mass(pg)                                                          # Σ V_i U_i
domain_volume(pg)                                                       # |Ω|
```

`domain_volume` is analytic for the built-in rectangular and spherical domains; define a method for a custom geometry label.

## Volumes on moving grids

After every step of a moving grid, the time stepper recomputes the volumes from the new geometry with its `volume_estimator` (default `ShepardVolumes()`; `nothing` leaves them untouched). With the config-driven builders the estimator of the `Grid` namespace (`:volumes`) is used for the initial volumes *and* for every re-estimation ([`build_particle_grid`](@ref) stores it in `context[:VolumeEstimator]`, which [`build_timestepper`](@ref) and [`build_adaptivity`](@ref) read). Recomputation keeps the volumes consistent with the current cloud at the cost of one neighbour sweep, and never drifts. The alternative of integrating $\dot V_i = V_i\,(\nabla\cdot w)_i$ along with the state (the geometric conservation law) is smoother in time but accumulates error and needs periodic re-anchoring; it is not implemented.

The collocation scheme itself is not exactly conservative, with or without motion: $\sum_i V_i U_i$ changes by the scheme's truncation error per step. What the volumes buy is a *well-defined* discrete mass, so that topology changes can preserve it exactly.

## Particle insertion and merging

On Lagrangian grids particles cluster in compressions and thin out in expansions. `ParticleAdaptivity` repairs that between steps:

```julia
ad = ParticleAdaptivity(Float64; max_gap = 1.6, min_gap = 0.5, clearance = 0.7,
                        samples_per_dx = 2, exclude_reduced_order = true,
                        max_events = typemax(Int),
                        volume_estimator = ts.volume_estimator)   # moving grid: the stepper's estimator
report = adapt_particles!(pg, ad, _extract_order(interp))   # between time steps
report.inserted, report.removed, report.mass_residual         # residual at round-off
```

### Detection

Lengths are in units of $\Delta x = \min_d \Delta x_d$.

- **Insertion:** a neighbour pair farther apart than `max_gap` whose midpoint has no particle within `clearance` gets a new particle at the midpoint. The clearance test is what prevents insertions on a regular lattice, where second neighbours are $2\Delta x$ apart but the midpoint is occupied.
- **Merge:** a neighbour pair closer than `min_gap` is replaced by one particle at its midpoint.
- At most one event per particle neighbourhood and per call, and at most `max_events` per call. Event centres are kept $R\,(2 + 2\sqrt D) + \Delta x$ apart, so that no event's remap region (the support of its quadrature box) contains a particle that another event removes or adds: the events of one call are remapped one after the other on the old topology, and a particle removed by one event would otherwise take its share of another event's swept content with it. (With the former spacing of $4R$ the regions overlapped in 2-D and 3-D, and calls with several events lost up to $10^{-7}$ of the mass.)
- **Shock band:** with `exclude_reduced_order`, no event happens within $4R$ of a particle whose MOOD order was lowered during the last step. Interpolation of any order is unreliable across a discontinuity, and a Lagrangian grid in a shock region is better left alone (or driven towards Eulerian motion by the mover).

### The conservative remap

A topology change replaces the partition of unity $\{\phi_i^{\text{old}}\}$ by $\{\phi_i^{\text{new}}\}$. Each particle's volume and content change by the *swept* amounts

```math
\delta V_i = \int \big(\phi_i^{\text{new}} - \phi_i^{\text{old}}\big)\,\mathrm dx, \qquad
\delta m_i = \int \big(\phi_i^{\text{new}} - \phi_i^{\text{old}}\big)\,\tilde U\,\mathrm dx ,
```

where $\tilde U$ is the limited (bound-clipped) linear reconstruction of the *old* solution around the nearest old particle. New particles have $\phi^{\text{old}} = 0$, removed particles $\phi^{\text{new}} = 0$. Since both families sum to one pointwise, $\sum_i \delta V_i = 0$ and $\sum_i \delta m_i = 0$ for *any* quadrature: conservation is exact by construction, which is the particle analogue of swept-region remapping in ALE finite volumes. The integrals are evaluated on a lattice of spacing $\Delta x/\texttt{samples\_per\_dx}$ over the box of half-width $2R$ around the event, using only the old neighbour structure. The new state is $U_i = (V_i U_i + \delta m_i)/(V_i + \delta V_i)$.

Two corrections follow:

1. **Residual.** The stored content of a removed particle, $V_r U_r$, is not exactly $\int \phi_r^{\text{old}} \tilde U$. The difference is spread over the affected particles in proportion to their new volumes, so $\sum V U$ is preserved to round-off.
2. **Repair.** Each affected particle is clipped to the bounds of its old neighbourhood, and the clipped content is redistributed to affected particles that still have room. This enforces the discrete maximum principle while keeping the sum exact; if no room is left, the remainder is spread by volume and the bounds are violated by that remainder.

Then the particle arrays are compacted/extended, the neighbour lists are rebuilt, and new particles start at the scheme's maximum order. The remap is second-order accurate in smooth regions (linear reconstruction, averaged over the new shape function's support).

### Volumes after the call

The remap's volumes $V_i + \delta V_i$ are integrals of the Shepard partition. A moving grid re-estimates the volumes after the next step with its own estimator, which defines volumes differently (per particle, the two differ by several percent near an event), and that re-estimation would change $\sum V U$ by about as much as an event moves. With `volume_estimator` set (the builders pass the stepper's estimator on moving grids), `adapt_particles!` therefore re-estimates the volumes right after the topology change and puts the resulting change of $\sum V U$ back onto the particles the remap changed, with the bounded, exactly conservative redistribution of the repair above. The call is then exact for the volumes the following steps use, and the states differ from the remap's only by that small correction. (Keeping each particle's content $V_i U_i$ across the re-estimation would also be exact, but it puts the per-particle volume differences, up to 18 % next to a merge, into the states.)

### What the remap does not do

- It does not change states outside the event's $3R$–$4R$ neighbourhood, so far-field solutions are untouched (with `volume_estimator`, far-field *volumes* change by the estimator's global renormalisation, typically $O(1/N)$).
- It does not conserve across the time step itself; see above.
- It does not move particles for mesh quality. A repulsion or smoothing term belongs in the `GridMover`, and keeps the particle set well-spaced *before* adaptivity has to act.

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["Volumes.jl", "Adaptivity.jl"]
Private = false
```
