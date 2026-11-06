# Frequency of the boundary misclassification: how many Ns (10..400) on a few boxes lose an end point,
# replicating get_points' lattice (canvas_mins + (I-1) dx) via ParticleGrid's ComputationalDomain.
using HyperCloudCore
function count_bad(lo, hi; Nss = 10:400, irf = 2.5, T = Float64)
    bad = Int[]
    for Ns in Nss
        dx = (T(hi) - T(lo)) / Ns
        geom = get_rectangular_domain(T, (lo,), (hi,))
        cd = HyperCloudCore.ComputationalDomain(geom, (dx,), irf)
        pos, isb, _, _, _ = HyperCloudCore.get_points(cd, geom; nominal_dx = (dx,), interp_range_factor = irf)
        count(!, isb) == Ns + 1 || push!(bad, Ns)
    end
    return bad
end
for (lo, hi) in ((0.0, 1.0), (-1.0, 1.0), (-0.5, 0.5)), T in (Float64, Float32)
    b = count_bad(lo, hi; T = T)
    println("[$lo, $hi] $T: $(length(b)) of 391 values of Ns lose a wall point, e.g. ", first(b, 8))
end
