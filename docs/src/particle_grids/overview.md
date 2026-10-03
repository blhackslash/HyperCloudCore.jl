# Particle Grid Overview

The `ParticleGrid` serves as the primary orchestrator for the mesh-free solver, representing the active computational domain by tying together the physical geometry, state vectors, and mesh-free topology. It manages the internal state and execution context through a series of pre-allocated components and memory workspaces.

## Instantiation and Required Inputs

Constructing a `ParticleGrid` requires defining the fundamental physical and mathematical properties of the simulation. The core mandatory arguments are:

*   **`geom` (`GeometricDomain`)**: A mathematical abstraction defining the problem's physical shape and boundary regions (e.g., rectangular or spherical). It controls whether spatial coordinates reside inside the interior and dictates integer boundary tagging.
*   **`nominal_dx` (`NTuple`)**: Defines the baseline Cartesian distance spacing for the universal point generator. This dictates the spatial resolution of the resulting particle cloud.
*   **`weight_func` (`MLSWeightFunction`)**: A moving least squares weighting function, such as an `ExponentialWeightFunction` or `InverseWeightFunction`. This function provides the spatial cutoff radius (`R`) required to configure the interpolation range and the global spatial hashing bins.
*   **`M` (`Int`)**: The dimension of the state vector, which allocates the memory capacity for the particle state variables (`rhos`) and curvatures.

## Core Grid Structures

The particle grid execution context is divided into distinct internal structures:

*   **ParticleGridCore**: Encapsulates the fundamental particle geometry, including positions, volumes, boundary flags, and arrays for universal adaptive order tracking.
*   **GridMetadata**: Stores critical solver variables such as grid capacities, physical resolutions, and the active interaction radius.
*   **SharedBuffers**: Provides thread-safe, pre-allocated workspaces for floating-point, integer, and boolean operations to prevent allocations during execution.

## Spatial Hashing and Neighbor Search

The mesh-free topology relies on a fast spatial hashing and branchless neighbor search methodology.

*   **GlobalBins**: Defines the coarse spatial hashing bins and a linked list system for the neighbor search algorithm. It converts 1D, 2D, or 3D physical coordinates into a flattened 1D array index corresponding to a specific spatial coarse cell.
*   **NeighborData**: Maintains the ranges, indices, computed weights, and strictly shortest distance vectors for all active particle neighborhoods. 
*   The neighbor search is executed in two branchless passes: a counting pass that scans adjacent bins for valid neighbors within the interaction radius, and a writing pass that accurately records indices, weights, and distances directly into the target buffers.
*   Distance calculations and topology updates natively enforce physical periodic wrapping limits per axis.

## Memory Optimization and Reordering

To maximize computational efficiency, the grid manages spatial sorting using a `ReorderData` component. 

*   The grid performs a proximity-optimized spatial reordering of the entire particle configuration to drastically improve CPU cache locality and memory access patterns.
*   Spatial sorting is achieved using Morton Z-order curves for 2D and 3D grids, which generate indices by interleaving the bits of normalized spatial coordinates.
*   Once the permutation buffer is sorted, fast native Julia in-place permutations (`Base.permute!`) update the underlying positions, state vectors, and physical flags.
*   The memory mutation is bypassed entirely if the permutation buffer is determined to be already sorted.

## Dynamic Time Stepping

The grid acts as the foundation for dynamically calculating the exact geometric CFL-restricted time step across the active domain. 

*   The grid evaluates the maximum eigenvalues from the equation system against the localized geometric coefficients of the moving least squares (MLS) formulation.
*   The system extracts the effective spatial derivative scales by employing a Cholesky factorization of the localized MLS moment matrix.

## Documentation

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["Grid/ParticleGrids.jl"]
Private = false
```