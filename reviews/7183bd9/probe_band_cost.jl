# Cost of the shock-band marking in _detect_events: 2-D (200²) and 3-D (32³) jittered periodic grids, 10 % of the
# particles with a lowered order (a band of them), median of 5 calls.
using HyperCloudCore, Random, Logging, Statistics
const HC = HyperCloudCore
for (D, n) in ((2, 200), (3, 32))
    dx = 1 / n
    geom = get_rectangular_domain(Float64, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D); is_periodic = true)
    pg = ParticleGrid(geom, ntuple(_ -> dx, D), ExponentialWeightFunction(1.0, 2.5dx), 1; randomness = ntuple(_ -> 0.2dx, D), rng = MersenneTwister(1))
    N = pg.meta.N
    for i in 1:N; pg.core.particle_orders[i] = abs(pg.core.positions[i][1] - 0.5) < 0.05 ? 1 : 3; end
    ad = ParticleAdaptivity(Float64)
    HC._detect_events(pg, ad, 3)
    t = median([(@elapsed HC._detect_events(pg, ad, 3)) for _ in 1:5])
    println("D=$D N=$N reduced=$(count(==(1), view(pg.core.particle_orders, 1:N))): _detect_events median $(round(1e3t; digits = 1)) ms")
end
