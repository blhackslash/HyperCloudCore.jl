# Float32 walls at powers of two (the canvas minimum lies in the next binade, so its rounding error is a full ulp of
# the wall): lost wall points, duplicates, walls without a point on them; Ns with ulp(wall)/dx < 0.1.
using HyperCloudCore, Printf, Random
const HC = HyperCloudCore
function check(T, lo, hi, Ns; R = 2.5)
    dx = (T(hi) - T(lo)) / Ns
    geom = get_rectangular_domain(T, (T(lo),), (T(hi),))
    cd = HC.ComputationalDomain(geom, (dx,), T(R))
    pos, isb, _, _, _ = HC.get_points(cd, geom; nominal_dx = (dx,), interp_range_factor = T(R))
    xs = sort([p[1] for (p, b) in zip(pos, isb) if !b])
    return (n = length(xs), onlo = xs[1] == T(lo), onhi = xs[end] == T(hi), md = minimum(diff(xs)) / dx,
            gaplo = (Float64(xs[1]) - lo) / Float64(dx), gaphi = (hi - Float64(xs[end])) / Float64(dx))
end
let rng = MersenneTwister(5), lost = 0, dup = 0, off = 0, tot = 0, ex = String[]
    T = Float32
    for lo in (-1.0, -2.0, -4.0, -8.0, -0.5), L in (0.3, 0.5, 0.7, 1.0, 1.7), _ in 1:40
        hi = lo + L
        Ns = rand(rng, 20_000:400_000)
        dx = (T(hi) - T(lo)) / Ns
        q = eps(T(abs(lo))) / dx
        q < 0.1 || continue
        tot += 1
        r = check(T, lo, hi, Ns)
        l = r.n < Ns + 1; d = r.md < 0.5; o = !(r.onlo && r.onhi)
        lost += l; dup += d; off += o
        (l || d || o) && length(ex) < 15 && push!(ex, @sprintf("[%g, %g] Ns=%d ulp/dx=%.3f n=%d (exp %d) onlo=%d onhi=%d first point %+.3f dx from lo, last %+.3f dx from hi, md=%.3f",
                                               lo, hi, Ns, q, r.n, Ns + 1, r.onlo, r.onhi, r.gaplo, r.gaphi, r.md))
    end
    println("Float32 power-of-two walls: $tot configurations, lost $lost, duplicated $dup, wall without a point on it $off")
    foreach(println, ex)
end
