# HyperCloudCore.jl

Physics-agnostic meshless solver engine for hyperbolic PDEs: particle grids, MLS
(generalized finite difference) operators, Central/Upwind/MUSCL/WENO schemes, limiters,
MOOD, SSP-RK and IMEX time stepping, moving grids, and CPU/GPU execution backends.
Concrete physics lives in HyperCloud.jl, not here.

## Commands

```bash
julia --project -e 'using Pkg; Pkg.test()'                       # full CPU suite
julia -t 8 --project=test test/runtests.jl                        # same suite, faster to iterate (test env)
julia --project=test -e 'using Pkg; Pkg.instantiate()'            # once, for the line above
julia --project=docs -e 'using Pkg; Pkg.instantiate()'            # docs env, once
julia --project=docs docs/make.jl                                 # build docs into docs/build
HYPERCLOUD_BACKEND=amdgpu julia --project=test/gpu test/runtests.jl   # device tests (AMD GPU only)
julia -t 8 --project benchmark/step.jl                            # step throughput, see BENCH_* in the file
```

`test/runtests.jl` takes no arguments: to run a subset, include the files you need in a
session (the shared helpers `make_pg_1d`, `set_ic!`, … are defined in `runtests.jl`).

## Layout

| Path | Contents |
|:-----|:---------|
| `src/HyperCloudCore.jl` | Module, `Space`/`State`/`Flux`/`Velocity` aliases, include order |
| `src/API.jl`, `src/Types.jl` | API contracts physics packages implement (`flux`, `max_eigenvalue`, `velocity`, sources, …) and abstract types |
| `src/Backend.jl` | Execution backends and `foreach_particle` (every parallel per-particle loop) |
| `src/DeviceAdapt.jl`, `src/Growable.jl`, `src/MemoryBudget.jl` | `Adapt` rules / `KernelView`, growable buffers, memory estimate and `reserve!` |
| `src/Grid/` | Domains, particle grids, neighbour search (cell lists), BCs, volumes, adaptivity, reordering |
| `src/Interpolation/` | MLS interpolators, schemes, numerical fluxes, limiters |
| `src/TimeStepping/` | RK/IMEX time stepper, Butcher tableaus, MOOD, grid movers |
| `src/Builders/` | Config-driven builders (`build_*`) used by HyperCloud |
| `docs/src/gpu_roadmap.md` | Which passes run as kernels, data layout, what is still host-only |

## Rules for kernel code

Every parallel per-particle loop is a kernel body `body(i, args...)` launched with
`foreach_particle(body, backend, N, args)`; never write a raw `@threads`/`@batch` loop or
backend-specific code elsewhere. Kernel bodies must work on every backend:

- No allocation, no dynamic dispatch, no `Dict` access, only bits-type arguments
  (`kernel_view`/`Adapt` turns host structs into their kernel form; add an
  `adapt_structure` method in `src/DeviceAdapt.jl` for new containers).
- No scalar indexing of device arrays outside kernels (device tests run with
  `allowscalar(false)`).
- Polyester rules (from `src/Backend.jl`): a body must not use the static parameters of
  the enclosing method (`T`, `D`, `Val(D)`, `zero(T)`), and every captured variable must
  be assigned exactly once in the enclosing function (reassigned captures get boxed and
  allocate).
- Per-particle scratch goes into `Growable`s with the `storage` array type, and must be
  registered in `src/MemoryBudget.jl` (`particle_buffers`/`slot_buffers`) so
  `memory_estimate` and `reserve!` stay correct.
- Without a GPU, check new or changed kernels with
  `set_backend!(HyperCloudCore.KA.CPU())` (PoCL). Results match the Polyester path to
  round-off, not bit for bit.
- New kernels or host-only passes: update the tables in `docs/src/gpu_roadmap.md`.

## Conventions

- Supports **Julia 1.10** (`[compat] julia = "1.10"`, tested on GitHub). Don't use
  1.11+ features (`public`, `@main`, `Memory`, `ScopedValues`, `:greedy` scheduling, …),
  even though the container's Julia is newer.
- Stay generic in `T` (Float32 and Float64 both work: no `Float64` literals in numerics,
  use `T(…)`/`zero(T)`/`eps(T)`), in `D` (1–3) and in `M`.
- Use the `SVector` aliases (`State{M,T}`, `Flux{D,M,T}`, …) and keep hot paths type
  stable and allocation free; check with `@allocated`/`@code_warntype` when touching them.
- Every exported function needs a docstring and a test, and must appear in the docs:
  `docs/make.jl` uses `checkdocs = :exports`, so an undocumented export fails the build.
  New pages need an entry in `pages` in `docs/make.jl`.
- Docs cross-link to HyperCloud via DocumenterInterLinks, which downloads an inventory
  from github.io (may be blocked here). For a local build point
  `HYPERCLOUD_INVENTORY` at `../HyperCloud.jl/docs/build/objects.inv` after building
  HyperCloud's docs, or report the cross-reference warnings.
- Numerics changes (schemes, weights, MOOD, tableaus) come with a convergence or
  accuracy check: state the observed order or error before and after in the PR.
- Commit subjects follow the repo style: `Area: what changed`
  (e.g. `MOOD: drop orders per particle`, `Docs: zero-sum divergence weights`).
