# Random Float32 1-D domains [c, c + L] with Ns spacings that Float32 resolves (ulp(|c| + L) < dx / 10):
# count lost wall points (interior count < Ns + 1), duplicated ones (min spacing < dx/2), walls without a point on them.
using HyperCloudCore, Printf, Random
const HC = HyperCloudCore
function check(T, lo, hi, Ns; R = 2.5)
    dx = (T(hi) - T(lo)) / Ns
    geom = get_rectangular_domain(T, (T(lo),), (T(hi),))
    cd = HC.ComputationalDomain(geom, (dx,), T(R))
    pos, isb, _, _, _ = HC.get_points(cd, geom; nominal_dx = (dx,), interp_range_factor = T(R))
    xs = sort([p[1] for (p, b) in zip(pos, isb) if !b])
    return (n = length(xs), onlo = xs[1] == T(lo), onhi = xs[end] == T(hi), md = minimum(diff(xs)) / dx)
end
let rng = MersenneTwister(7), lost = 0, dup = 0, off = 0, tot = 0, ex = String[]
    T = Float32
    while tot < 3000
        c  = T(round(200 * rand(rng) - 100; digits = 3)); L = T(round(0.1 + 5 * rand(rng); digits = 2))
        Ns = rand(rng, (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000))
        dx = (T(c + L) - c) / Ns
        eps(T(abs(c) + L)) < dx / 10 || continue
        tot += 1
        r = check(T, c, c + L, Ns)
        l = r.n < Ns + 1; d = r.md < 0.5; o = !(r.onlo && r.onhi)
        lost += l; dup += d; off += o
        (l || d) && length(ex) < 10 && push!(ex, @sprintf("c=%g L=%g Ns=%d ulp/dx=%.3f n=%d onlo=%d onhi=%d md=%.3f", c, L, Ns, eps(T(abs(c) + L)) / dx, r.n, r.onlo, r.onhi, r.md))
        o && !(l || d) && length(ex) < 10 && push!(ex, @sprintf("[off] c=%g L=%g Ns=%d ulp/dx=%.3f n=%d onlo=%d onhi=%d md=%.3f", c, L, Ns, eps(T(abs(c) + L)) / dx, r.n, r.onlo, r.onhi, r.md))
    end
    println("Float32 random: $tot configurations, lost $lost, duplicated $dup, wall not on a point $off")
    foreach(println, ex)
end
