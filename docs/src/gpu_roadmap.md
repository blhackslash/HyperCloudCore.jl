# Execution backends and the GPU roadmap

HyperCloudCore runs every parallel per-particle loop as a *kernel body* `body(i, args...)` launched with
`foreach_particle(body, backend, N, args)` (see `src/Backend.jl`). The backend is a global setting:

```julia
set_backend!(PolyesterBackend())            # default: Polyester `@batch per=core`
set_backend!(ThreadsBackend())              # Base.Threads `@threads :dynamic`
set_backend!(KernelAbstractions.CPU())      # KernelAbstractions kernels, compiled through PoCL on the host
set_backend!(CUDABackend())                 # from CUDA.jl (likewise ROCBackend(), MetalBackend(), oneAPIBackend())
```

`set_threads!(true/false)` still selects the Threads / Polyester backend.

## What runs as a kernel today

| pass | kernel body | file |
|---|---|---|
| neighbour gather + scheme caches | `_rhs_gather!` | `TimeStepping/TimeSteppers.jl` |
| flux divergence + explicit sources | `_rhs_flux!` | `TimeStepping/TimeSteppers.jl` |
| stage assembly, implicit solve, implicit RHS | `_stage_predictor!`, `_stage_implicit_solve!`, `_stage_implicit_rhs!` | `TimeStepping/TimeSteppers.jl` |
| MOOD candidate check | `_mood_flag!` | `TimeStepping/MOOD.jl` |
| geometry refresh (moving grids) | `_refresh_slots!` | `TimeStepping/GridMovers.jl` |
| inverse MLS moment matrices | `_moment_body!` | `Interpolation/Interpolators.jl` |
| Shepard volumes | `_shepard_sums!`, `_shepard_sweep!` | `Grid/Volumes.jl` |
| CFL estimate | `_cfl_body!` | `Grid/ParticleGrids.jl` |
| periodic wrap, neighbour counting and writing | `_wrap_position!`, `_count_neighbours!`, `_write_neighbours!` | `Grid/NeighborLogic.jl` |

With the KernelAbstractions backend these bodies are compiled as device kernels. The 0.10 `CPU()`
backend compiles through PoCL (OpenCL on the host), so running the test suite with it validates the
GPU kernel rules (bits-type arguments, no dynamic dispatch, no allocation) without a GPU. Results agree
with the Polyester path to round-off (the OpenCL compiler contracts floating-point operations differently).

## Data layout

The storage containers (`ParticleGridCore`, `NeighborData`, `SharedBuffers`, `GlobalBins`,
`InteractionBuffer`, `MUSCL.gradients`, `MLSWeights`, `IMEXButcherTableau`) are parametric in their array
types. `Adapt.adapt_structure` methods in `src/DeviceAdapt.jl` turn the host objects into bits types for a
kernel launch: arrays become device arrays, the mutable `GridMetadata` an immutable `GridMetadataView`,
`Ref`s plain values, and the host-only parts of a `ParticleGrid` (geometry closures and boundary map, scratch
buffers, reordering state) become `nothing`. `device_grid(CuArray, pg)` builds a grid whose arrays live on a
device.

## Rules for kernel bodies

- Only `args` as input; no closures over the enclosing method's variables or static parameters.
- No allocation, no `Dict` access, no `@warn`/`@info`, no exceptions with interpolated messages.
- No dynamic dispatch: a runtime integer must not become a `Val` (use `basis_count(D, degree)` for the
  basis size of a runtime degree; the interpolator tables are selected by the generated dispatchers).
- Arrays of arrays are not device-indexable; the stage arrays of the stepper are tuples for that reason.

## Not yet on the device

| part | status | plan |
|---|---|---|
| prefix sum of the neighbour search | host loop between the two kernel passes | exclusive scan on the device (`accumulate!`), or a counting-sort cell list |
| `build_global_bins!` (linked lists) | host | atomic insertion into cell heads, or a sort-based cell list |
| MOOD decrement and forcing halo | host (serial by design: no cascade) | per-particle kernel with the snapshot; forcing with atomics |
| `expand_mask!` | host | one kernel per hop |
| Morton reordering | host (`sort!`) | device sort of the keys + permutation kernels |
| boundary conditions (`FixedDirichlet`, `OutflowBC`) | host (`bc_map::Dict`) | tag-indexed tuple of conditions; `OutflowBC`'s donor search as iterated kernels |
| particle adaptivity | host | stays on the host (rare, inherently serial); transfer the grid around the call |
| WENO, Tiwari upwind | not device-compatible: the stencil order is a runtime `Val` | pad to the maximum basis length and mask |
| reductions (`get_time_step`, `total_mass`) | `minimum`/loop on host arrays | `mapreduce` on device arrays |
| data residency | arrays are copied into kernel views; host arrays on the CPU backends | keep the grid on the device (`device_grid`) and transfer only around host passes |

Float64 throughput on consumer GPUs is a small fraction of Float32; everything is parametric in `T`.
