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
| MOOD candidate check, decrement, forcing, mask expansion | `_mood_flag!`, `_mood_drop!`, `_mood_force!`, `_expand_body!` | `TimeStepping/MOOD.jl` |
| geometry refresh (moving grids) | `_refresh_slots!` | `TimeStepping/GridMovers.jl` |
| inverse MLS moment matrices | `_moment_body!` | `Interpolation/Interpolators.jl` |
| Shepard volumes | `_shepard_sums!`, `_shepard_sweep!` | `Grid/Volumes.jl` |
| CFL estimate | `_cfl_body!` | `Grid/ParticleGrids.jl` |
| cell list (bin histogram, scatter, per-bin sort) | `_bin_count!`, `_bin_scatter!`, `_bin_sort!` | `Grid/NeighborLogic.jl` |
| periodic wrap, neighbour counting, slot ranges, writing | `_wrap_position!`, `_count_neighbours!`, `_neighbour_ranges!`, `_write_neighbours!` | `Grid/NeighborLogic.jl` |
| boundary conditions (Dirichlet; outflow donor search, one pass per hop) | `_dirichlet_body!`, `_outflow_init!`, `_outflow_pass!`, `_outflow_fallback!` | `Grid/BoundaryConditions.jl` |
| adaptivity pre-check (any pair violating the spacing rules) | `_adapt_candidate!` | `Grid/Adaptivity.jl` |

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

The per-particle scratch of the time stepper and the interpolators (stage states, neighbour gathers, MOOD
masks, cached gradients and moment matrices) grows with the particle count. Those buffers are `Growable`s
(`src/Growable.jl`): a mutable holder of an array of the `storage` type given to the constructor, reallocated
with `similar` when the capacity is exceeded (`resize!` only exists for `Vector`). `Adapt` unwraps a
`Growable` to the array it holds, and *every* `foreach_particle` launch passes its arguments through
`kernel_view` (`adapt(KernelView(), args)`), so kernel bodies see the same argument types on the CPU
backends as on a device. A step on a device therefore reads

```julia
dg     = device_grid(ROCArray, pg)
interp = MUSCL(Float64, D, M, 3, NoLimiter(), RusanovFlux(); storage = ROCArray)
ts     = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64); storage = ROCArray)
set_backend!(ROCBackend())
ts(eq, dg, t, dt)
```

With the config-driven builders, `context[:Storage] = ROCArray` does the same: `build_particle_grid`
returns the device grid, `build_scheme` and `build_timestepper` allocate on it.

Before a `KABackend` launch, `check_resident` walks the arguments and throws if an array does not live on
that backend. Without it AMDGPU.jl would copy a host array to the device for the launch and silently drop
the kernel's writes to it.

## Rules for kernel bodies

- Only `args` as input; no closures over the enclosing method's variables or static parameters.
- No allocation, no `Dict` access, no `@warn`/`@info`, no exceptions with interpolated messages.
- No dynamic dispatch: a runtime integer must not become a `Val` (use `basis_count(D, degree)` for the
  basis size of a runtime degree; the interpolator tables are selected by the generated dispatchers).
- Arrays of arrays are not device-indexable; the stage arrays of the stepper are tuples for that reason.
- No two particles write the same entry: neighbour-to-neighbour effects are written in the *pull* form
  (every particle gathers from its neighbours, as the MOOD forcing and `expand_mask!` do), which the
  symmetric neighbour relation allows and which keeps the result independent of the launch order.

## Not yet on the device

| part | status | plan |
|---|---|---|
| Morton reordering | host (`sort!`); only runs when a grid is constructed | device sort of the keys + permutation kernels, if ever needed between steps |
| particle adaptivity (events) | host: detection, remap and compaction are serial per event | stays there; a kernel pre-check (`_adapt_candidate!`) skips the round trip when the cloud is well spaced, otherwise the grid is copied to the host, adapted, and copied back in place |
| WENO, Tiwari upwind | not device-compatible: the stencil order is a runtime `Val` | pad to the maximum basis length and mask |
| data residency | the grid (`device_grid`), the stepper and MUSCL (`storage`) live on the device; static and moving grids step entirely there, neighbour search, MOOD, boundary conditions and the reductions included (the `bc_map` loop itself is on the host, its conditions are kernels); the builders follow `context[:Storage]` | |
| growing a device grid | every per-particle and per-slot array of the grid is a `Growable`, so `ensure_capacity!` works on the device (adaptivity relies on it) | |
| MLS volumes | host (point queries on the cell list) | `ShepardVolumes` is on the device; `MLSVolumes` is a construction-time choice |

## Slot layout

The per-slot arrays (`indices`, `weights`, `distances`, and the stepper's gather buffers) have two
layouts, selected by the element type of `neighbor.ranges`:

- **row-major** (`UnitRange` slots, host grids): a particle's slots are contiguous, which is what the
  CPU's caches want;
- **slot-major** (`StepRange` slots with stride `N`, device grids): slot `k` of all particles is
  contiguous, so the threads of a wavefront, which handle consecutive particles, read consecutive
  addresses. Every particle is padded to `max_nb` slots.

Kernel bodies only iterate `ranges[i]`, so they serve both; `device_grid` converts between the
layouts on host arrays, and the search writes whichever layout the grid has. On the RX 9070 XT the
neighbour gather and flux passes, which are over 90 % of a step, run 2.7 times faster slot-major (a step 2.4 times, 250 000 particles, Float64).

With that, both passes move their bytes at roughly 70–90 % of the card's bandwidth, so further gains
need fewer bytes per slot. The largest remaining item is the gather buffer round trip: the gather
pass writes a neighbour's state and difference per slot and the flux pass reads them back, about a
fifth of the slot traffic. Dropping it means the flux functor gathers the state itself, which
changes the interpolator API (it would receive the state vector), so it is left for a separate step.
The moment matrices of a moving grid (`_moment_body!`, a fifth of a moving step) are recomputed at
every stage because every stage has its own geometry; that is the scheme, not the layout.

## The neighbour search

The cell list is a counting sort: a kernel writes every particle's bin and bumps the bin's count with
an atomic add, a prefix sum over the bins gives the segment starts, a second kernel scatters the
particles into their segments through an atomic cursor, and a kernel per bin sorts its (small) segment
by descending particle index. That last sort removes the only nondeterminism the atomics introduce,
so the neighbour lists are identical on every backend and in every run, and in the order the former
linked list produced. The counting and writing passes then walk the `3^D` bins around a particle's bin
as contiguous segments. The slot ranges come from a prefix sum of the counts (`cumsum!`) and two
reductions (total and maximum), all on the device.

## Testing on the device

`HYPERCLOUD_BACKEND=amdgpu` adds the device tests (`test/gpu_tests.jl`) to the suite. They need AMDGPU.jl,
which is not a test dependency of the package (no release supports KernelAbstractions 0.10 yet, it comes
from the main branch), so they run in the `test/gpu` environment:

```sh
julia --project=test/gpu -e 'using Pkg; Pkg.instantiate()'     # once
HYPERCLOUD_BACKEND=amdgpu julia --project=test/gpu test/runtests.jl
```

Float64 throughput on consumer GPUs is a small fraction of Float32; everything is parametric in `T`.
