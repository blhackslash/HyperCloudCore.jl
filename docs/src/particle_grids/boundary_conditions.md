# Boundary Conditions

Boundary conditions act on the boundary (ghost) particles of the narrow band around the physical region. They are registered per boundary tag in the domain's `bc_map` and applied by `apply_boundary_conditions!`, which iterates over the map and calls each condition's functor.

The time stepper applies them to **every state it builds**: each stage state $Y_i$ (after the implicit solve) and the final state. MOOD re-forms a candidate after an order drop, and the boundary conditions are re-applied to that candidate as well. Interior particles are never touched by a boundary condition, and boundary particles are never updated by the spatial scheme; their right-hand side is zero.

On moving grids, boundary particles keep their positions (their grid velocity is zero).

## Built-in conditions

- **`FixedDirichlet`** sets every boundary particle with the given tag back to its value in `pg.rhos`. Because boundary particles are never advanced, this is their initial state.
- **`OutflowBC`** is a zero-gradient (transmissive) condition using a nearest-donor sweep. Interior particles act as donors; in up to five passes, each unresolved boundary particle with the tag copies the state of its nearest already-resolved neighbour. A boundary particle that finds no donor keeps its initial state.

## Writing your own

Subtype `AbstractBoundaryCondition` and implement the functor:

```julia
struct MyBC <: AbstractBoundaryCondition end

function (::MyBC)(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq, t)
    @inbounds for i in 1:pg.meta.N
        if pg.core.is_boundary[i] && pg.core.tags[i] == tag
            rhos_buffer[i] = ...   # write in place
        end
    end
    return nothing
end
```

| Argument | Meaning |
|---|---|
| `pg` | The grid; positions, tags, flags and neighbour lists are available |
| `rhos_buffer` | The state vector being built (a stage buffer, the MOOD candidate, or the final state). **Modify this one**, not `pg.rhos`. |
| `tag` | The boundary tag this call is responsible for |
| `ts` | The time stepper (useful for type-based dispatch) |
| `eq` | The PDE |
| `t` | Time of the state being built (stage time, or `t + dt`) |

Register the condition with `get_rectangular_domain(...; bc_map = Dict(1 => MyBC()))`. Use `pg.shared` buffers for scratch space (as `OutflowBC` does with `pg.shared.int_buffer`) rather than allocating.

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["BoundaryConditions.jl"]
Private = false
```
