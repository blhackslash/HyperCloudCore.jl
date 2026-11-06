# Geometric Domains

A `GeometricDomain` describes the physical problem region and nothing else: its bounding box, which points are inside, how boundary points are tagged, which axes are periodic, and which boundary condition belongs to which tag. Numerical padding (ghost layers) and point generation are handled separately when the `ParticleGrid` is built.

## Fields

| Field | Meaning |
|---|---|
| `mins`, `maxs` | Bounding box of the physical region |
| `is_periodic::SVector{D,Bool}` | Periodicity per axis |
| `is_interior(x)` | `true` if `x` lies in the physical region |
| `get_tag(x)` | `0` for interior points, a positive integer boundary tag otherwise |
| `bc_map::Dict{Int,AbstractBoundaryCondition}` | Boundary condition per tag |

## Built-in shapes

### Rectangular

```julia
get_rectangular_domain(T, mins, maxs; bc_map = Dict(), is_periodic = false)
```

`is_periodic` is either one `Bool` for all axes or an `NTuple{D,Bool}`. Boundary points get the tag of the nearest face: `2d - 1` for the lower and `2d` for the upper face of axis `d`, the lower tag on ties (edges and corners); the faces of a periodic axis are ignored.

| D | tags |
|:--|:-----|
| 1 | 1 left (`x = min`), 2 right |
| 2 | 1 left, 2 right, 3 bottom, 4 top (a channel periodic in x has only 3 and 4) |
| 3 | 1/2 at `x = min/max`, 3/4 at `y`, 5/6 at `z` |

Up to version 0.7.9, 1-D and 3-D boxes tagged every boundary point 1, so `Dict(1 => bc)` covered both walls of an interval (and every face of a box) and a tag-2 entry was dead. Such a map now covers only the left wall (the x-min face), and the constructor warns that the other tags have no condition: write `Dict(1 => bc, 2 => bc)` (`Dict(t => bc for t in 1:6)` in 3-D).

Boundary particles whose tag has no entry in `bc_map` keep their state (like a Dirichlet condition), and an entry whose tag no particle carries does nothing. When a non-empty `bc_map` and the tags of the boundary particles do not match, the `ParticleGrid` constructor warns (once per combination and session). An empty `bc_map` freezes every boundary particle without a warning.

### Spherical

```julia
get_spherical_domain(T, center, radius; bc_map = Dict(), is_periodic = false)
```

A disc (2-D) or ball (3-D); every exterior point gets tag 1.

## Point generation

When a `ParticleGrid` is constructed, the domain is wrapped in a `ComputationalDomain`, and points are generated as follows:

1. **Canvas.** On non-periodic axes the bounding box is extended by $\lceil R/\Delta x_d\rceil$ spacings of that axis on each side to host ghost particles, i.e. at least $R$ on every axis; periodic axes get no padding. (Up to version 0.7.9 every axis got $\lceil R/\Delta x_{\max}\rceil$ spacings, which left the band thinner than $R$ on the finer axis of an anisotropic grid.)
2. **Lattice.** A Cartesian lattice with spacing `nominal_dx` (plus optional jitter `randomness`) covers the canvas. A non-periodic axis gets `round(L/dx) + 1` points, endpoints included: if a wall lies on the lattice (its index `(w - canvas_min)/dx` is an integer up to the round-off of the wall, the canvas minimum and the division), that lattice point is put on the wall exactly before the jitter is added, so both walls are interior. A wall between two lattice points (an extent that is not a multiple of `dx`) is not snapped, however close. (Up to version 0.7.9 a distance tolerance did the snapping; far from the origin in Float32 it grew to almost `dx` and could put two points on a wall.) A periodic axis gets `round(L/dx)` points, because the endpoint coincides with the start point after wrapping.
3. **Narrow band.** Lattice points outside the physical region are kept only if they lie within $R + \Delta x_{\max}$ of an interior point. These become the boundary (ghost) particles.
4. **Tags.** `get_tag` sets `tags` and `is_boundary = (tag != 0)`.

Boundary particles carry the boundary condition and are never updated by the spatial scheme; see [Boundary Conditions](@ref).

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["Domains.jl"]
Private = false
```
