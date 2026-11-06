# Shepard quadrature error on a jittered periodic cloud (no ghosts) under refinement: f = 1 + sin(πx) + x², on [-1, 1].
using HyperCloudCore, Random
f = x -> 1 + sin(π * x) + x^2          # ∫ = 2 + 2/3
for jit in (0.1, 0.3)
    errs = Float64[]
    for N in (40, 80, 160, 320)
        dx = 2 / N
        geom = get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true)
        e = 0.0
        for seed in 1:4
            pg = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1; randomness = (jit * dx,), rng = MersenneTwister(seed))
            e += abs(sum(pg.core.volumes[i] * f(pg.core.positions[i][1]) for i in 1:pg.meta.N) - (2 + 2 / 3)) / 4
        end
        push!(errs, e)
    end
    println("jitter $jit: Shepard errors ", round.(errs; sigdigits = 3), " EOC ", round.([log2(errs[j] / errs[j + 1]) for j in 1:3]; digits = 2))
end
