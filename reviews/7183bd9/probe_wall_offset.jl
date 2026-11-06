# Offset 1-D domains [c, c + L] with Ns spacings, Float32/Float64: interior count, wall points on the walls, min
# spacing; only configurations the float type resolves (ulp(|c| + L) < dx / 10).
using HyperCloudCore, Printf
const HC = HyperCloudCore
function check(T, lo, hi, Ns; R = 2.5)
    dx = (T(hi) - T(lo)) / Ns
    geom = get_rectangular_domain(T, (T(lo),), (T(hi),))
    cd = HC.ComputationalDomain(geom, (dx,), T(R))
    pos, isb, _, _, _ = HC.get_points(cd, geom; nominal_dx = (dx,), interp_range_factor = T(R))
    xs = sort([p[1] for (p, b) in zip(pos, isb) if !b])
    return (n = length(xs), onlo = xs[1] == T(lo), onhi = xs[end] == T(hi), md = minimum(diff(xs)) / dx, xmax = xs[end])
end
let bad = 0, tot = 0
    for T in (Float32, Float64), c in (0.0, -1.0, 1.0, 3.7, 10.0, -10.0, 100.0, -100.0, 1000.0, -1000.0, 1.0e4),
        L in (1.0, 2.0, 0.3, 7.0), Ns in (10, 37, 100, 333, 1000, 3000, 10_000, 30_000, 100_000)
        T == Float64 && Ns > 10_000 && continue
        dx = (T(c + L) - T(c)) / Ns
        eps(T(abs(c) + L)) < dx / 10 || continue
        r = check(T, c, c + L, Ns)
        tot += 1
        ok = r.n == Ns + 1 && r.onlo && r.onhi && r.md > 0.5
        ok || (bad += 1; @printf("%-7s c=%-8g L=%-4g Ns=%-6d ulp/dx=%.3f n=%d (exp %d) onlo=%d onhi=%d minspacing=%.3f dx xmax=%.9g\n",
                                 T, c, L, Ns, eps(T(abs(c) + L)) / dx, r.n, Ns + 1, r.onlo, r.onhi, r.md, r.xmax))
    end
    println("offset sweep: $bad / $tot resolvable configurations with a lost/duplicated/off wall point")
end
