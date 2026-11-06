# Candidate regression configurations for F2 (duplicated wall point with the old distance tolerance)
using HyperCloudCore
const HC = HyperCloudCore
for (lo, hi, Ns) in ((61.6f0, 61.6f0 + 0.24f0, 5000), (-54.181f0, -54.181f0 + 0.39f0, 10_000), (100f0, 101f0, 10_000), (1000f0, 1001f0, 1000))
    dx   = (hi - lo) / Ns
    geom = get_rectangular_domain(Float32, (lo,), (hi,))
    cd   = HC.ComputationalDomain(geom, (dx,), 2.5f0)
    pos, isb, _, _, _ = HC.get_points(cd, geom; nominal_dx = (dx,), interp_range_factor = 2.5f0)
    xs   = sort([p[1] for (p, b) in zip(pos, isb) if !b])
    println("[$lo, $hi] Ns=$Ns: interior $(length(xs)) (expected $(Ns + 1)), on walls $(xs[1] == lo)/$(xs[end] == hi), min spacing $(minimum(diff(xs)) / dx) dx")
end
