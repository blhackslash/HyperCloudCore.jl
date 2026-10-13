"""
    build_particle_grid(grid_conf, context) -> ParticleGrid

Requires `real_type(context)`, `[:D]`, `[:M]`, `[:Domain]`, `[:WeightConf]`. Stores `:max_dx` in the
context, builds the weight function, the grid mover and the volume estimator from the same
`Grid` namespace, and sets `context[:Grid]`.
"""
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
    context[:Grid] = pg
    return pg
end
