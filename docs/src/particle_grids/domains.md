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

`is_periodic` is either one `Bool` for all axes or an `NTuple{D,Bool}`. Tags in 2-D are assigned by the nearest edge: 1 left, 2 right, 3 bottom, 4 top. In 1-D and 3-D every boundary point has tag 1.

### Spherical

```julia
get_spherical_domain(T, center, radius; bc_map = Dict(), is_periodic = false)
```

A disc (2-D) or ball (3-D); every exterior point gets tag 1.

## Point generation

When a `ParticleGrid` is constructed, the domain is wrapped in a `ComputationalDomain`, and points are generated as follows:

1. **Canvas.** On non-periodic axes the bounding box is extended by $\lceil R/\Delta x_{\max}\rceil$ spacings on each side to host ghost particles. As soon as *any* axis is periodic, no padding is added on any axis.
2. **Lattice.** A Cartesian lattice with spacing `nominal_dx` (plus optional jitter `randomness`) covers the canvas. A non-periodic axis gets `round(L/dx) + 1` points, endpoints included. A periodic axis gets `round(L/dx)` points, because the endpoint coincides with the start point after wrapping.
3. **Narrow band.** Lattice points outside the physical region are kept only if they lie within $R + \Delta x_{\max}$ of an interior point. These become the boundary (ghost) particles.
4. **Tags.** `get_tag` sets `tags` and `is_boundary = (tag != 0)`.

Boundary particles carry the boundary condition and are never updated by the spatial scheme; see [Boundary Conditions](@ref).

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["Domains.jl"]
Private = false
```
