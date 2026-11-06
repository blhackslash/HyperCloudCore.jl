# Which points coincide in Float32 [0,1], Ns = 1e6 (tolerance 0.95 dx)?
using HyperCloudCore; const HC = HyperCloudCore
for Ns in (600_000, 800_000, 1_000_000)
    T = Float32; dx = one(T) / Ns
    geom = get_rectangular_domain(T, (0,), (1,))
    cd = HC.ComputationalDomain(geom, (dx,), T(2.5))
    pos, isb, _, _, _ = HC.get_points(cd, geom; nominal_dx = (dx,), interp_range_factor = T(2.5))
    x = [p[1] for p in pos]; p = sortperm(x)
    d = findall(iszero, diff(x[p]))
    println("Ns=$Ns: ", length(d), " coincident pairs: ", [(x[p[k]], isb[p[k]], isb[p[k+1]]) for k in d])
end
