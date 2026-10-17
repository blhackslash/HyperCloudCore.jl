# HyperCloudCore.jl
**HyperCloudCore.jl** is a high-performance, physics-agnostic meshless solver backend designed for the numerical integration of hyperbolic Partial Differential Equations (PDEs) in Julia. 

## Commands
- Tests (CPU): `julia --project -e 'using Pkg; Pkg.test()'`
- Tests (GPU): `HYPERCLOUD_BACKEND=amdgpu julia --project=test/gpu test/runtests.jl`
  (once: `julia --project=test/gpu -e 'using Pkg; Pkg.instantiate()'`; the registered AMDGPU.jl 2.9, see test/gpu/Project.toml)
- Single test file: `julia --project test/runtests.jl <name>`

## Conventions
- Kernels use KernelAbstractions.jl, never backend-specific code outside `src/backends/`.
- No scalar indexing on GPU arrays; tests run with `GPUArraysCore.allowscalar(false)`.
- Public functions need a docstring and a test.

## Workflow
- Small commits with clear messages. Run the tests before every commit.
- Don't change Project.toml [compat] or add dependencies without asking.