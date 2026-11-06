using HyperCloudCore, Printf, Random
const HC = HyperCloudCore
T = Float32
let rng = MersenneTwister(11), tot = 0, big_err = 0, mx = 0.0
    while tot < 1500
        c  = T(round(40 * rand(rng) - 20; digits = 2)); L = T(round(0.2 + 3 * rand(rng); digits = 2))
        Ns = rand(rng, 10_000:200_000)
        dx = (T(c + L) - c) / Ns
        q = eps(T(abs(c) + L)) / dx
        0.01 < q < 0.1 || continue
        tot += 1
        geom = get_rectangular_domain(T, (c,), (T(c + L),))
        cd = HC.ComputationalDomain(geom, (dx,), T(2.5))
        r = (c - cd.canvas_mins[1]) / dx
        e = abs(r - round(r)); mx = max(mx, e); big_err += e > 0.01
        tot <= 3 && @printf("c=%g L=%g Ns=%d q=%.3f cmin=%.9g r=%.5f\n", c, L, Ns, q, cd.canvas_mins[1], r)
    end
    println("tot $tot, r_lo error > 0.01: $big_err, max $mx")
end
