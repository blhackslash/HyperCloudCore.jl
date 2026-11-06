# Fine Float32 lattices with 0.01 < ulp(|c| + L)/dx < 0.1 (resolvable, but the wall's ratio carries more than 0.01
# of round-off from canvas_min): lost wall points, duplicates, walls without a point on them.
using HyperCloudCore, Printf, Random
const HC = HyperCloudCore
function check(T, lo, hi, Ns; R = 2.5)
    dx = (T(hi) - T(lo)) / Ns
    geom = get_rectangular_domain(T, (T(lo),), (T(hi),))
    cd = HC.ComputationalDomain(geom, (dx,), T(R))
    pos, isb, _, _, _ = HC.get_points(cd, geom; nominal_dx = (dx,), interp_range_factor = T(R))
    xs = sort([p[1] for (p, b) in zip(pos, isb) if !b])
    wl = HC._wall_index(T(lo), cd.canvas_mins[1], dx); wh = HC._wall_index(T(hi), cd.canvas_mins[1], dx)
    return (n = length(xs), onlo = xs[1] == T(lo), onhi = xs[end] == T(hi), md = minimum(diff(xs)) / dx, wl = wl, wh = wh,
            gaplo = (Float64(xs[1]) - lo) / dx, gaphi = (hi - Float64(xs[end])) / dx)
end
let rng = MersenneTwister(11), lost = 0, dup = 0, off = 0, tot = 0, ex = String[]
    T = Float32
    while tot < 1500
        c  = T(round(40 * rand(rng) - 20; digits = 2)); L = T(round(0.2 + 3 * rand(rng); digits = 2))
        Ns = rand(rng, 10_000:200_000)
        dx = (T(c + L) - c) / Ns
        q = eps(T(abs(c) + L)) / dx
        0.01 < q < 0.1 || continue
        tot += 1
        r = check(T, c, c + L, Ns)
        l = r.n < Ns + 1; d = r.md < 0.5; o = !(r.onlo && r.onhi)
        lost += l; dup += d; off += o
        (l || d || o) && length(ex) < 12 && push!(ex, @sprintf("c=%g L=%g Ns=%d ulp/dx=%.3f n=%d (exp %d) wall idx=(%d,%d) onlo=%d onhi=%d first/last point %.3f/%.3f dx inside md=%.3f",
                                               c, L, Ns, q, r.n, Ns + 1, r.wl, r.wh, r.onlo, r.onhi, r.gaplo, r.gaphi, r.md))
    end
    println("Float32 fine: $tot configurations, lost $lost, duplicated $dup, wall without a point on it $off")
    foreach(println, ex)
end
