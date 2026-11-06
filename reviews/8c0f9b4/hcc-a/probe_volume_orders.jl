# Check the documented quadrature orders of the volume estimators (docs/src/particle_grids/volumes.md).
using HyperCloudCore, StaticArrays, Random, Printf
f(x) = 1 + sin(π * x) + x^2
F(a, b) = (b - a) + (-cos(π * b) + cos(π * a)) / π + (b^3 - a^3) / 3
function err(est, N; periodic = false, jitter = 0.0, seed = 1)
    geom = get_rectangular_domain(Float64, (0.0,), (1.0,); is_periodic = periodic)
    dx = 1 / N
    pg = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1; randomness = (jitter * dx,), rng = MersenneTwister(seed))
    update_volumes!(pg, est)
    q = sum(pg.core.volumes[i] * f(pg.core.positions[i][1]) for i in 1:pg.meta.N)
    return abs(q - F(0.0, 1.0))
end
Ns = (40, 80, 160, 320)
for (label, kw) in (("box lattice", (;)), ("periodic 30% jitter", (periodic = true, jitter = 0.3)))
    println("== ", label)
    for (name, est) in (("Shepard", ShepardVolumes()), ("MLS0", MLSVolumes(0)), ("MLS1", MLSVolumes(1)), ("MLS2", MLSVolumes(2)), ("MLS3", MLSVolumes(3)))
        es = [err(est, N; kw...) for N in Ns]
        eocs = [log2(es[k] / es[k + 1]) for k in 1:length(es) - 1]
        @printf("%-8s err %s  EOC %s\n", name, join([@sprintf("%.2e", e) for e in es], " "), join([@sprintf("%5.2f", e) for e in eocs], " "))
    end
end
