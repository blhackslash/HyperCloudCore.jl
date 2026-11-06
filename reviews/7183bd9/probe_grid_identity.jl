# Are standard grids (divisible extents, Float64/Float32, jittered or not, walls/periodic) identical to the old code?
using HyperCloudCore, Random, Logging
for T in (Float64, Float32), (lo, hi, n) in ((0.0, 1.0, 30), (-1.0, 1.0, 30), (0.0, 2.0, 50), (-1.0, 1.0, 46)), jit in (0.0, 0.2, 0.45), per in (false, true), D in (1, 2)
    dx = (T(hi) - T(lo)) / n
    geom = get_rectangular_domain(T, ntuple(_ -> T(lo), D), ntuple(_ -> T(hi), D); is_periodic = per)
    pg = with_logger(NullLogger()) do
        ParticleGrid(geom, ntuple(_ -> dx, D), ExponentialWeightFunction(one(T), T(2.5) * dx), 1; randomness = ntuple(_ -> T(jit) * dx, D), rng = MersenneTwister(2))
    end
    N = pg.meta.N
    println("T=$T [$lo,$hi] n=$n jit=$jit per=$per D=$D N=$N Nint=$(pg.meta.N_interior) h=", string(hash((collect(view(pg.core.positions, 1:N)), collect(view(pg.core.volumes, 1:N)), collect(view(pg.core.is_boundary, 1:N)))), base = 16))
end
