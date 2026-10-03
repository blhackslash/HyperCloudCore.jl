# Boundary Conditions

Boundary conditions in the meshfree solver are managed via a flexible `AbstractBoundaryCondition` strategy pattern. During time integration, the `apply_boundary_conditions!` function iterates over the active boundary condition map stored in the grid's geometry configuration (`pg.geometry.bc_map`) and dispatches the corresponding strategy for each registered domain tag.

## Built-In Strategies

The solver provides two primary concrete boundary condition strategies out of the box:

*   **`FixedDirichlet`**: A strict, unchanging boundary condition strategy. When called, it iterates over the grid and enforces a static Dirichlet condition by resetting the state of any boundary particle matching the specified tag back to its initial original state stored in `pg.rhos`.
*   **`OutflowBC`**: A zero-gradient, transmissive boundary condition strategy. It enforces outflow utilizing a multi-pass nearest-donor algorithm. It initiates all active interior particles as valid state donors, then executes up to 5 symmetrical passes outward, dynamically locating the nearest resolved neighbor using squared distances and copying its state to the target particle. It safely falls back to the original initial state for any completely orphaned boundary particles that fail to resolve a donor.

## Custom Boundary Conditions (API Contract)

You can easily extend the solver with custom boundary behaviors. To do this, you must define a new struct that subtypes `AbstractBoundaryCondition` and implement its functor (callable struct) method.

The mandatory API contract requires your functor to match the following signature:

```julia
function (::YourCustomBC)(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq, t)
    # Custom logic here
    return nothing
end
```

### Input Parameters
*   **`pg` (`ParticleGrid`)**: The main execution context, providing access to the underlying topology, metadata, and core arrays like `pg.core.tags` and `pg.core.is_boundary`.
*   **`rhos_buffer` (`AbstractVector{State}`)**: The active state vector buffer that the timestepper is currently writing to. This is the array your function **must modify in place**.
*   **`tag` (`Int`)**: The specific integer tag associated with the boundary region you are evaluating.
*   **`ts`**: The active time stepping functor (e.g., Runge-Kutta or IMEX).
*   **`eq` (`HyperbolicPDE`)**: The current equation system being solved.
*   **`t` (`Real`)**: The current continuous simulation time.

### Expected Output and Behavior
*   The functor must iterate over the grid and conditionally evaluate particles (typically checking if `pg.core.tags[i] == tag` and `pg.core.is_boundary[i]`).
*   The functor must modify the target states inside `rhos_buffer` directly (in-place mutation).
*   The function must return `nothing` upon completion.

## Documentation

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["Grid/BoundaryConditions.jl"]
Private = false
```