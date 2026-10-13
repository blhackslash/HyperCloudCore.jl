"""
    HyperCloudCore.Builders

Config-driven constructors for the numerical objects of HyperCloudCore. Every builder has the
signature `build_x(conf::Dict, context::Dict)` and dispatches on `Val(conf[:name])` (or a similar
selector key), so a wrapper package (or a test suite) can add its own variants by defining
another `Val` method. This module owns the *numerics* side of the configuration: schemes,
fluxes, limiters, MOOD, tableaus, time steppers, weight functions, domains, boundary conditions,
grid movers, volume estimators, adaptivity and the particle grid. The *physics* side (PDEs,
initial conditions, source terms) belongs to the wrapper.

# Context keys

The `context::Dict{Symbol, Any}` carries everything that is already built. Builders read:

| Key | Set by | Read by |
|---|---|---|
| `:Type` | caller | all |
| `:Equation`, `:D`, `:M` | caller (PDE builder) | grid, schemes, time stepper, domains |
| `:Domain` | `build_domain` | `build_particle_grid`, `build_mover` |
| `:WeightConf` | caller | `build_particle_grid` → `build_weights` |
| `:max_dx`, `:interp_range` | `build_particle_grid`, `build_weights` | |
| `:Grid` | `build_particle_grid` | `build_mood_criterion` |
| `:Flux`, `:Limiter`, `:MOOD`, `:Scheme`, `:Tableau` | the respective builders | `build_scheme`, `build_timestepper` |
| `:ExplicitSources`, `:ImplicitSources` | caller | `build_timestepper` |

# Namespaces

Configuration dictionaries are flat with namespaced keys (`:Scheme_name`, `:Time_stepper`, …);
`PDEStudioCore.extract_namespace(params, :Scheme)` yields the `conf` dict a builder expects. Keys used:

- `Grid`: `domain`, `mins`, `maxs`, `Ns`, `periodic`, `bc`, `randomness_factor`, `SEED`, `mover`,
  `mover_alpha`, `mover_amplitude`, `volumes`, `volumes_degree`, `volumes_samples_per_dx`
- `Weight`: `name`, `range`, `alpha`
- `Flux`: `name`;  `Limiter`: `name`, `mode`;  `MOOD`: `criterion`, `strategy`, `delta_relax`
- `Scheme`: `name`, `order`, `MLS_order`, `upwind_alg_nd`
- `Time`: `stepper`, `reset_orders_per_stage`, `full_search_per_stage`
- `Adapt`: `enabled`, `max_gap`, `min_gap`, `clearance`, `samples_per_dx`, `exclude_reduced_order`
"""
module Builders

using ..HyperCloudCore
using StaticArrays, Random

export build_weights, build_domain, parse_bc, build_particle_grid, build_mover, build_volume_estimator
export build_flux, build_limiter, build_mood, build_mood_criterion, build_mood_strategy, build_scheme
export build_tableau, build_timestepper, build_adaptivity

include("weights.jl")
include("domains.jl")
include("movers.jl")
include("volumes.jl")
include("particle_grid.jl")
include("limiter.jl")
include("mood.jl")
include("schemes.jl")
include("tableaus.jl")
include("timestepper.jl")
include("adaptivity.jl")

end # module
