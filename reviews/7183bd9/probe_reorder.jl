# Morton keys in T: does the particle order of constructed grids change (Float32 / Float64, 2-D / 3-D, jittered)?
using HyperCloudCore, Random, Logging
for T in (Float64, Float32), D in (2, 3), jit in (0.0, 0.3)
    n = D == 2 ? 64 : 16; dx = T(1 / n)
    geom = get_rectangular_domain(T, ntuple(_ -> zero(T), D), ntuple(_ -> one(T), D); is_periodic = true)
    pg = ParticleGrid(geom, ntuple(_ -> dx, D), ExponentialWeightFunction(one(T), T(2.5) * dx), 1;
                      randomness = ntuple(_ -> T(jit) * dx, D), rng = MersenneTwister(1))
    println("T=$T D=$D jitter=$jit N=$(pg.meta.N) order hash=", string(hash(collect(view(pg.core.positions, 1:pg.meta.N))), base = 16))
end
