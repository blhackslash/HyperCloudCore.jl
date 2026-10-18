```@meta
CurrentModule = HyperCloudCore
```

# Config-Driven Builders

The `build_*` functions in `src/Builders/` turn flat, namespaced parameter dictionaries into the numerical objects of the package. They exist so that a wrapper (or a test suite) does not have to re-implement the symbol-to-type mapping for every scheme, tableau, limiter and strategy, and so that this mapping is updated together with the types it names.

The physics side is deliberately *not* here. PDEs, initial conditions and source terms are built by the wrapper, which fills `context[:Equation]`, `[:D]`, `[:M]` and the source tuples before calling the builders. In [HyperCloud.jl](@extref HyperCloud Builder-API), the runners such as [`run_direct_simulation`](@extref HyperCloud HyperCloud.run_direct_simulation) call the builders below in this order, and the configuration keys are listed in its [Configuration Reference](@extref HyperCloud Configuration-Reference).

```julia
using HyperCloudCore

context = Dict{Symbol, Any}(:T => Float64, :D => 1, :M => 1, :Equation => eq,
                            :ExplicitSources => (), :ImplicitSources => ())
context[:WeightConf] = Dict(:name => :exponential, :range => 2.5, :alpha => 1.0)

context[:Domain]  = build_domain(Dict(:domain => :rectangular, :mins => (-1.0,), :maxs => (1.0,), :periodic => true), context)
pg                = build_particle_grid(Dict(:Ns => (200,), :mover => :lagrangian), context)
context[:Flux]    = build_flux(Dict(:name => :Rusanov), context)
context[:Limiter] = build_limiter(Dict(:name => :none), context)
context[:MOOD]    = build_mood(Dict(:criterion => :U2, :strategy => :EPD1, :delta_relax => 1e-4), context)
context[:Scheme]  = build_scheme(Dict(:name => :MUSCL, :order => 3), context)
context[:Tableau] = build_tableau(Dict(:stepper => :RK3), context)
ts                = build_timestepper(context)
adaptivity        = build_adaptivity(Dict(:enabled => true), context)
```

Every builder is `build_x(conf::Dict, context::Dict)`, where `conf` holds the keys of one namespace with the prefix stripped (`:Scheme_order` becomes `:order`; in HyperCloud, [`extract_namespace`](@extref HyperCloud HyperCloud.extract_namespace) does the stripping). The builder dispatches on `Val(conf[:name])` (or the namespace's selector key), so adding a variant means defining one more method:

```julia
HyperCloudCore.build_flux(::Val{:HLL}, conf::Dict, context::Dict) = MyHLLFlux()
```

An unknown name raises an error that names the missing method.

## Context keys

The `context::Dict{Symbol, Any}` carries everything that is already built:

| Key | Set by | Read by |
|---|---|---|
| `:T` (legacy `:Type`) | caller | all (via [`real_type`](@ref)) |
| `:Equation`, `:D`, `:M` | caller (PDE builder) | grid, schemes, time stepper, domains |
| `:Domain` | [`build_domain`](@ref) | [`build_particle_grid`](@ref), [`build_mover`](@ref) |
| `:WeightConf` | caller (`Weight` namespace) | [`build_particle_grid`](@ref) → [`build_weights`](@ref) |
| `:max_dx`, `:interp_range` | [`build_particle_grid`](@ref), [`build_weights`](@ref) | |
| `:Grid` | [`build_particle_grid`](@ref) | [`build_mood_criterion`](@ref), memory reservation |
| `:Flux`, `:Limiter`, `:MOOD`, `:Scheme`, `:Tableau` | the respective builders | [`build_scheme`](@ref), [`build_timestepper`](@ref) |
| `:ExplicitSources`, `:ImplicitSources` | caller | [`build_timestepper`](@ref) |
| `:VolumeEstimator` (optional) | caller | [`build_timestepper`](@ref) |
| `:Storage` (array type, default `Array`) | caller | [`build_particle_grid`](@ref) ([`device_grid`](@ref)), [`build_scheme`](@ref), [`build_timestepper`](@ref) |
| `:MemoryLimit` (bytes), `:MemoryTolerance`, `:SlotsPerParticle` | caller | [`build_timestepper`](@ref) ([`reserve!`](@ref); stores the reservation in `:Memory`) |

`:Storage` and `:MemoryLimit` are how a run is put on a GPU and given a memory budget; see [Execution backends and the GPU roadmap](@ref).

## Selectors

| Builder | Namespace | Selector | Values |
|---|---|---|---|
| [`build_domain`](@ref) | `Grid` | `:domain` | `:rectangular`, `:spherical` |
| [`build_weights`](@ref) | `Weight` | `:name` | `:exponential`, `:inverse` |
| [`build_mover`](@ref) | `Grid` | `:mover` | `:none`, `:lagrangian`, `:scaled_lagrangian`, `:sinusoidal` |
| [`build_volume_estimator`](@ref) | `Grid` | `:volumes` | `:shepard`, `:mls` |
| [`build_flux`](@ref) | `Flux` | `:name` | `:Rusanov`, `:Upwind` |
| [`build_limiter`](@ref) | `Limiter` | `:name` | `:none`, `:minmod`, `:superbee`, `:VK`, `:BJ` (+ `:mode`) |
| [`build_mood`](@ref) | `MOOD` | `:criterion`, `:strategy` | `:none`/`:only`/`:U1`/`:U2`; `:EPD0`/`:SEPD0`/`:EPD1`/`:EPD2` |
| [`build_scheme`](@ref) | `Scheme` | `:name` | `:MUSCL`, `:Upwind`, `:Central`, `:WENO` |
| [`build_tableau`](@ref) | `Time` | `:stepper` | `:Euler`, `:RK2`, `:RK2Ralston`, `:RK3`, `:RK4`, `:IMEXEuler`, `:ARS222`, `:ARS233`, `:PRSSP3`, `:SSP332` |
| [`build_adaptivity`](@ref) | `Adapt` | `:enabled` | `true`/`false` (+ the [`ParticleAdaptivity`](@ref) knobs) |

[`build_particle_grid`](@ref) calls [`build_weights`](@ref), [`build_mover`](@ref) and [`build_volume_estimator`](@ref) itself, and [`build_domain`](@ref) resolves the boundary conditions with [`parse_bc`](@ref). The per-builder keys are listed in the docstrings below.

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["Builders.jl", "weights.jl", "domains.jl", "movers.jl", "volumes.jl", "particle_grid.jl", "limiter.jl", "mood.jl", "schemes.jl", "tableaus.jl", "timestepper.jl", "adaptivity.jl"]
```
