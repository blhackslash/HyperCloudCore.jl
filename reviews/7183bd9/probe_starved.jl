# _check_starved: how many interior particles have < D + 1 neighbours for usual radii and jitters (walls and periodic).
using HyperCloudCore, Random, Printf, Logging
for D in (1, 2, 3), Rf in (1.2, 1.5, 2.0, 2.5), jit in (0.0, 0.2, 0.3, 0.45), per in (true, false)
    n = D == 1 ? 200 : D == 2 ? 30 : 10; dx = 1 / n
    geom = get_rectangular_domain(Float64, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D); is_periodic = per)
    local pg
    with_logger(NullLogger()) do
        pg = ParticleGrid(geom, ntuple(_ -> dx, D), ExponentialWeightFunction(1.0, Rf * dx), 1;
                          randomness = ntuple(_ -> jit * dx, D), rng = MersenneTwister(1))
    end
    nst = count(i -> !pg.core.is_boundary[i] && length(pg.neighbor.ranges[i]) < D + 1, 1:pg.meta.N)
    nst > 0 && @printf("D=%d R=%.1f dx jitter=%.2f periodic=%d: %d of %d interior particles starved\n", D, Rf, jit, per, nst, pg.meta.N_interior)
end
println("done (configurations not listed: no starved particle)")
