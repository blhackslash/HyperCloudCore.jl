# Particle Volumes and Adaptivity

Collocation particles carry no mass. To define a discrete conserved quantity, and to make inserting and merging particles exactly conservative, every particle gets a positive weight $V_i$ (stored in `pg.core.volumes`) with

```math
\sum_i V_i = |\Omega|, \qquad \sum_i V_i\, f(x_i) \approx \int_\Omega f\,\mathrm dx .
```

No Voronoi tessellation is built or stored. $V_i = \int_\Omega \phi_i\,\mathrm dx$ for shape functions $\phi_i$ that form a partition of unity ($\sum_i \phi_i \equiv 1$), evaluated with a quadrature on the point cloud itself or on a throwaway lattice. Only the scalar $V_i$ is kept.

## Volume estimators

| Estimator | Shape functions | Evaluation | Quadrature order | Cost |
|---|---|---|---|---|
| `ShepardVolumes(; sweeps = 1)` | $\phi_i(x) = w(\lvert x - x_i\rvert) / \sum_k w(\lvert x - x_k\rvert)$ | particle-based, no lattice | 2 (quasi-uniform clouds) | one neighbour sweep per `sweep` |
| `MLSVolumes(P; samples_per_dx = 2)` | MLS reproducing-kernel functions of degree $P$ | background lattice over $\Omega$ | $P + 1$ | lattice points × neighbours |

**Shepard** starts from the kernel-density volume $V_i \propto 1/S_i$, $S_i = w(0) + \sum_j w_{ij}$, applies `sweeps` partition-of-unity sweeps $V_i \leftarrow \sum_j (w_{ij}/S_j)\,V_j$ (which preserve $\sum V$), and renormalises to $|\Omega|$. On a uniform periodic cloud it reproduces $V_i = \Delta x^D$ exactly. It needs a kernel that is bounded at zero distance; with `InverseWeightFunction` the volumes fall back to $|\Omega|/N$ with a warning.

**MLS** evaluates, at every lattice point $y$, the shape functions $\phi_i(y) = w_i\, p_i^{\mathsf T} a$ with $\big(\sum_k w_k p_k p_k^{\mathsf T}\big) a = e_1$ and the full basis $p_k = (1, \text{monomials}(x_k - y))$. Because $\sum_i \phi_i(y)\, q(x_i) = q(y)$ for polynomials $q$ of degree $\le P$, the volumes integrate such polynomials exactly (up to the lattice's midpoint error). `P = 0` is the lattice-integrated Shepard partition. Shape functions of degree $\ge 1$ can be negative; a volume that comes out negative is clamped before the final renormalisation.

```julia
pg = ParticleGrid(geom, dx, wf, M; volume_estimator = MLSVolumes(2))   # at construction
update_volumes!(pg, ShepardVolumes())                                   # any later time
total_mass(pg)                                                          # Σ V_i U_i
domain_volume(pg)                                                       # |Ω|
```

`domain_volume` is analytic for the built-in rectangular and spherical domains; define a method for a custom geometry label.

## Volumes on moving grids

After every step of a moving grid, the time stepper recomputes the volumes from the new geometry with its `volume_estimator` (default `ShepardVolumes()`; `nothing` leaves them untouched). Recomputation keeps the volumes consistent with the current cloud at the cost of one neighbour sweep, and never drifts. The alternative of integrating $\dot V_i = V_i\,(\nabla\cdot w)_i$ along with the state (the geometric conservation law) is smoother in time but accumulates error and needs periodic re-anchoring; it is not implemented.

The collocation scheme itself is not exactly conservative, with or without motion: $\sum_i V_i U_i$ changes by the scheme's truncation error per step. What the volumes buy is a *well-defined* discrete mass, so that topology changes can preserve it exactly.

## Particle insertion and merging

On Lagrangian grids particles cluster in compressions and thin out in expansions. `ParticleAdaptivity` repairs that between steps:

```julia
ad = ParticleAdaptivity(Float64; max_gap = 1.6, min_gap = 0.5, clearance = 0.7,
                        samples_per_dx = 2, exclude_reduced_order = true)
report = adapt_particles!(pg, ad, _extract_order(interp))   # between time steps
report.inserted, report.removed, report.mass_residual         # residual at round-off
```

### Detection

Lengths are in units of $\Delta x = \min_d \Delta x_d$.

- **Insertion:** a neighbour pair farther apart than `max_gap` whose midpoint has no particle within `clearance` gets a new particle at the midpoint. The clearance test is what prevents insertions on a regular lattice, where second neighbours are $2\Delta x$ apart but the midpoint is occupied.
- **Merge:** a neighbour pair closer than `min_gap` is replaced by one particle at its midpoint.
- At most one event per particle neighbourhood and per call. Events are kept $4R$ apart so their remaps do not interact.
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

### What the remap does not do

- It does not change particles outside the event's $3R$–$4R$ neighbourhood, so far-field solutions are untouched.
- It does not conserve across the time step itself; see above.
- It does not move particles for mesh quality. A repulsion or smoothing term belongs in the `GridMover`, and keeps the particle set well-spaced *before* adaptivity has to act.

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["Volumes.jl", "Adaptivity.jl"]
Private = false
```
