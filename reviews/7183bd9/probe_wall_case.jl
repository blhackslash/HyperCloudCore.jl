# The one off-wall case of the offset sweep: Float32 [-1, -0.7], Ns = 1e5
using HyperCloudCore, Printf
const HC = HyperCloudCore
T = Float32
for (lo, hi, Ns) in ((-1.0, -0.7, 100_000), (-1.0, -0.7, 99_999), (1.0, 1.3, 100_000), (-1.0, 0.0, 100_000))
    dx = (T(hi) - T(lo)) / Ns
    geom = get_rectangular_domain(T, (T(lo),), (T(hi),))
    cd = HC.ComputationalDomain(geom, (dx,), T(2.5))
    r_lo = (T(lo) - cd.canvas_mins[1]) / dx; r_hi = (T(hi) - cd.canvas_mins[1]) / dx
    @printf("[%g, %g] Ns=%d dx=%.6g ulp(lo)/dx=%.3f  r_lo=%.5f (idx %d)  r_hi=%.5f (idx %d)  tolerance lo %.4f hi %.4f\n", lo, hi, Ns, dx, eps(T(abs(lo)))/dx,
            r_lo, HC._wall_index(T(lo), cd.canvas_mins[1], dx), r_hi, HC._wall_index(T(hi), cd.canvas_mins[1], dx),
            min(0.25, 0.01 + 8eps(T) * abs(r_lo)), min(0.25, 0.01 + 8eps(T) * abs(r_hi)))
end
