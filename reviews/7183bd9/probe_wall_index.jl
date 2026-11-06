# Sweep get_points on 1-D domains: interior count, wall points exactly on the walls, min spacing.
# Run with the new code (worktree test env) and the old code (/tmp/review_pr1/old/test) and diff the outputs.
using HyperCloudCore, Printf
const HC = HyperCloudCore
function check(T, lo, hi, Ns; R = 2.5, dxv = nothing)
    dx = dxv === nothing ? (T(hi) - T(lo)) / Ns : T(dxv)
    geom = get_rectangular_domain(T, (T(lo),), (T(hi),))
    cd = HC.ComputationalDomain(geom, (dx,), T(R))
    pos, isb, _, _, _ = HC.get_points(cd, geom; nominal_dx = (dx,), interp_range_factor = T(R))
    xs = sort([p[1] for (p, b) in zip(pos, isb) if !b])
    nexp = floor(Int, (big(hi) - big(lo)) / big(dx) + big(1e-6)) + 1   # lattice points in [lo, hi] (exact arithmetic)
    onlo = xs[1] == T(lo); onhi = xs[end] == T(hi)
    md = minimum(diff(xs)) / dx
    return (n = length(xs), nexp = nexp, onlo = onlo, onhi = onhi, md = md, xmax = xs[end])
end
mode = get(ARGS, 1, "offset")
let
if mode == "offset"
    # Float32 offset domains [c, c + L], Ns points
    bad = 0; tot = 0
    for T in (Float32, Float64), c in (0.0, 1.0, 10.0, 100.0, 1000.0, -1000.0, 1.0e4), L in (1.0, 2.0, 0.3), Ns in (10, 100, 1000, 3000, 10_000, 30_000)
        T == Float64 && Ns > 10_000 && continue
        r = check(T, c, c + L, Ns)
        tot += 1
        ok = r.n == Ns + 1 && r.onlo && r.onhi && r.md > 0.5
        ok || (bad += 1; @printf("%-7s c=%-8g L=%-4g Ns=%-6d n=%d (exp %d) onlo=%d onhi=%d minspacing=%.3f dx xmax=%.9g\n", T, c, L, Ns, r.n, Ns + 1, r.onlo, r.onhi, r.md, r.xmax))
    end
    println("offset sweep: $bad / $tot configurations with a lost/duplicated/off wall point")
elseif mode == "nondiv"
    # Non-divisible extents [0, 1] with dx = 1/(n + f): fractional part f of L/dx
    for T in (Float64, Float32), n in (10, 30, 100), f in (-0.02, -0.011, -0.009, -0.005, -0.001, 0.001, 0.005, 0.009, 0.011, 0.02, 0.3, 0.5)
        dx = 1 / (n + f)
        r = check(T, 0.0, 1.0, 0; dxv = dx)
        @printf("%-7s n=%-4d frac=%+.3f  N_interior=%-4d onhi=%d  xmax-1=%+.2e dx  min spacing=%.4f dx\n", T, n, f, r.n, r.onhi, (Float64(r.xmax) - 1) / Float64(T(dx)), r.md)
    end
end
end
