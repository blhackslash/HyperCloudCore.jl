"""
    build_particle_grid(grid_conf, context) -> ParticleGrid

`Grid` namespace. Requires `real_type(context)`, `[:D]`, `[:M]`, `[:Domain]` and `[:WeightConf]` (the
`Weight` namespace).

| Key | Meaning |
|:--|:--|
| `:Ns` | Particles per direction (tuple, required); the lattice spacing is `(maxs - mins) ./ Ns` |
| `:randomness_factor` | Jitter per direction as a fraction of the spacing (default 0) |
| `:seed` (legacy `:SEED`) | Seed of the `MersenneTwister` for the jitter (default 42) |
| `:mover`, `:mover_*` | Grid mover, see [`build_mover`](@ref) |
| `:volumes`, `:volumes_*` | Volume estimator, see [`build_volume_estimator`](@ref) |

Stores the largest spacing in `context[:max_dx]`, builds the weight function with
[`build_weights`](@ref)`(context[:WeightConf], context)` and sets `context[:Grid]` and
`context[:VolumeEstimator]` (the estimator of the initial volumes, which [`build_timestepper`](@ref) and
[`build_adaptivity`](@ref) then use for every re-estimation on a moving grid). The grid is built on
the host and moved to [`storage_type`](@ref)`(context)` with [`device_grid`](@ref) if that is not `Array`.
"""
function build_particle_grid end

function build_particle_grid(grid_conf::Dict, context::Dict)
    T = real_type(context)
    D = context[:D]::Int
    M = context[:M]::Int
    geom = context[:Domain]

    Ns  = grid_conf[:Ns]::Tuple
    rf  = get(grid_conf, :randomness_factor, ntuple(_ -> 0.0, D))::Tuple
    dxs = ntuple(d -> (T(geom.maxs[d]) - T(geom.mins[d])) / Int(Ns[d]), Val(D))
    context[:max_dx] = maximum(dxs)
    randomness = ntuple(d -> T(rf[d]) * dxs[d], Val(D))

    weight_func = build_weights(context[:WeightConf]::Dict, context)
    rng   = MersenneTwister(get(grid_conf, :seed, get(grid_conf, :SEED, 42))::Int)
    mover = build_mover(grid_conf, context)
    vol   = build_volume_estimator(grid_conf, context)

    pg = ParticleGrid(geom, dxs, weight_func, M;
                      randomness = randomness, rng = rng, mover = mover, volume_estimator = vol)
    context[:VolumeEstimator] = vol      # the time stepper and the adaptivity re-estimate with the same estimator
    storage = storage_type(context)
    storage === Array || (pg = device_grid(storage, pg))
    context[:Grid] = pg
    return pg
end
