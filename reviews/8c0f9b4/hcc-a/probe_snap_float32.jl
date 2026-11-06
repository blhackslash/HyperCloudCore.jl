# Wall snapping (sqrt(eps)·dx) vs the round-off of canvas_mins + (I-1)·dx, which grows with |x|/dx.
using HyperCloudCore, StaticArrays
function lost(T, lo, hi, Ns)
    dx = (T(hi) - T(lo)) / Ns
    geom = get_rectangular_domain(T, (lo,), (hi,))
    cd = HyperCloudCore.ComputationalDomain(geom, (dx,), T(2.5))
    pos, isb, _, _, _ = HyperCloudCore.get_points(cd, geom; nominal_dx = (dx,), interp_range_factor = T(2.5))
    xs = [p[1] for (p, b) in zip(pos, isb) if !b]
    return length(xs) != Ns + 1 || minimum(xs) != T(lo) || maximum(xs) != T(hi)
end
for (T, lo, hi, Nrange) in ((Float32, 0, 1, 100:100:20000), (Float32, -1, 1, 100:100:20000), (Float32, 10, 11, 10:10:2000),
                            (Float64, 1000, 1001, 1000:1000:200000))
    bad = [N for N in Nrange if lost(T, lo, hi, N)]
    println(T, " [", lo, ", ", hi, "]: ", length(bad), " of ", length(Nrange), " resolutions lose a wall point; first: ", first(bad, 6))
end
