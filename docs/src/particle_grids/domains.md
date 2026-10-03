# Geometric Domains

The `GeometricDomain` serves as a pure mathematical abstraction representing the physical shape and boundary regions of the problem space. It isolates the exact physical boundaries from the broader computational canvas used for numerical padding and ghost particle generation.

## Core Logic and Boundary Tagging

The `GeometricDomain` operates by encapsulating functional closures that continuously evaluate spatial coordinates. The internal logic relies on two primary functions:
*   **Interior Evaluation (`is_interior`)**: A function evaluating whether a given coordinate resides strictly within the mathematical geometry.
*   **Tag Generation (`get_tag`)**: A function generating integer boundary tags based on spatial location for any point that falls outside the interior.

These integer tags are stored within a `bc_map` dictionary, which maps specific geometric edges to user-defined physical boundary condition evaluators.

## Rectangular Domains

Constructed using `get_rectangular_domain`, this geometry models Cartesian boxes. 

*   **Inputs**: Requires the minimal bounding box coordinates (`mins`) and maximal bounding box coordinates (`maxs`) defined as spatial tuples. It also accepts keyword arguments for periodicity (`is_periodic`) and the boundary condition dictionary (`bc_map`).
*   **Logic**: The interior evaluation function enforces strict Cartesian bounding box constraints across all spatial dimensions.
*   **Tagging**: In 2D space, the domain automatically evaluates the nearest distance to the bounding edges, assigning a tag of 1 to the Left edge, 2 to the Right, 3 to the Bottom, and 4 to the Top. For other dimensions, it defaults to a uniform tag of 1.

## Spherical Domains

Constructed using `get_spherical_domain`, this geometry models circles in 2D or spheres in 3D.

*   **Inputs**: Requires a spatial tuple defining the `center` point and a singular `radius` value. Like the rectangular domain, it accepts `is_periodic` and `bc_map` keyword arguments.
*   **Logic**: Employs a squared-distance interior evaluation against the defined center point and radius to determine if a point resides inside the volume.
*   **Tagging**: The boundary tag generation is uniform, assigning a boundary tag of 1 to all exterior points.

## Documentation

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["Grid/Domains.jl"]
Private = false
```