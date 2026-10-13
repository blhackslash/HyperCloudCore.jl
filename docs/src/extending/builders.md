```@meta
CurrentModule = HyperCloudCore.Builders
```

# Config-Driven Builders

`HyperCloudCore.Builders` turns flat, namespaced parameter dictionaries into the numerical objects of the package. It exists so that a wrapper (or a test suite) does not have to re-implement the symbol-to-type mapping for every scheme, tableau, limiter and strategy, and so that this mapping is updated together with the types it names.

```julia
using HyperCloudCore, HyperCloudCore.Builders

context = Dict{Symbol, Any}(:Type => Float64, :D => 1, :M => 1, :Equation => eq,
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

Every builder is `build_x(conf::Dict, context::Dict)` and dispatches on `Val(conf[:name])` (or the namespace's selector key). Adding a variant means defining one more method:

```julia
Builders.build_flux(::Val{:HLL}, conf::Dict, context::Dict) = MyHLLFlux()
```

The physics side is deliberately *not* here: PDEs, initial conditions and source terms are built by the wrapper, which then fills `context[:Equation]`, `[:D]`, `[:M]` and the source tuples before calling into this module.

## Selectors

| Builder | Namespace | Selector | Values |
|---|---|---|---|
| `build_domain` | `Grid` | `:domain` | `:rectangular`, `:spherical` |
| `build_weights` | `Weight` | `:name` | `:exponential`, `:inverse` |
| `build_mover` | `Grid` | `:mover` | `:none`, `:lagrangian`, `:scaled_lagrangian`, `:sinusoidal` |
| `build_volume_estimator` | `Grid` | `:volumes` | `:shepard`, `:mls` |
| `build_flux` | `Flux` | `:name` | `:Rusanov`, `:Upwind` |
| `build_limiter` | `Limiter` | `:name` | `:none`, `:minmod`, `:superbee`, `:VK`, `:BJ` (+ `:mode`) |
| `build_mood` | `MOOD` | `:criterion`, `:strategy` | `:none`/`:only`/`:U1`/`:U2`; `:EPD0`/`:SEPD0`/`:EPD1`/`:EPD2` |
| `build_scheme` | `Scheme` | `:name` | `:MUSCL`, `:Upwind`, `:Central`, `:WENO` |
| `build_tableau` | `Time` | `:stepper` | `:Euler`, `:RK2`, `:RK2Ralston`, `:RK3`, `:RK4`, `:IMEXEuler`, `:ARS222`, `:ARS233`, `:PRSSP3`, `:SSP332` |
| `build_adaptivity` | `Adapt` | `:enabled` | `true`/`false` (+ the `ParticleAdaptivity` knobs) |

`parse_bc` resolves a `Grid_bc` dictionary `tag => :fixed_dirichlet | :outflow | <symbol of a BC type>` into the `bc_map` of the domain.

## Reference

```@autodocs
Modules = [HyperCloudCore.Builders]
```
