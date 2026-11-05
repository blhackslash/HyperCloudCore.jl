# Periodic jittered clouds with a smooth periodic integrand (the box integrand of probe_volume_orders.jl jumps at the seam).
using HyperCloudCore, StaticArrays, Random, Printf
f(x) = 1 + sin(2π * x) + 0.3 * cos(4π * x) + 0.2 * sin(6π * x)^2       # ∫_0^1 f = 1.1
function err(est, N; jitter = 0.3, seed = 1)
    geom = get_rectangular_domain(Float64, (0.0,), (1.0,); is_periodic = true)
    dx = 1 / N
    pg = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1; randomness = (jitter * dx,), rng = MersenneTwister(seed))
    update_volumes!(pg, est)
    return abs(sum(pg.core.volumes[i] * f(pg.core.positions[i][1]) for i in 1:pg.meta.N) - 1.1)
end
Ns = (40, 80, 160, 320, 640)
for jit in (0.1, 0.3)
    println("== periodic, jitter ", jit, " (mean over 3 seeds)")
    for (name, est) in (("Shepard", ShepardVolumes()), ("MLS0", MLSVolumes(0)), ("MLS1", MLSVolumes(1)), ("MLS2", MLSVolumes(2)), ("MLS3", MLSVolumes(3)))
        es = [sum(err(est, N; jitter = jit, seed = s) for s in 1:3) / 3 for N in Ns]
        eocs = [log2(es[k] / es[k + 1]) for k in 1:length(es) - 1]
        @printf("%-8s err %s  EOC %s\n", name, join([@sprintf("%.2e", e) for e in es], " "), join([@sprintf("%5.2f", e) for e in eocs], " "))
    end
end
