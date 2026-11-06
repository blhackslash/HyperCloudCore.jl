# Where do 1-D runs break / overshoot as a function of the CFL number (branch vs parent), affected and unaffected radii.
include(joinpath(@__DIR__, "header.jl"))
function run(o, R, cfl, f; lim = NoLimiter(), tab = RK4_Classical_Tableau(Float64), n = 100, tend = 1.0, mood = MOOD())
    pg = grid(1, Float64; n, R)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f(pg.core.positions[i])); end
    eq = LinearAdvection(((1.0,),), Float64); m = MUSCL(Float64, 1, 1, o, lim, RusanovFlux())
    ts = GeneralRKTimeStepper(eq, m, mood, tab)
    t = 0.0; k = 0
    while t < tend - 1e-12 && k < 5000
        dt = min(cfl * get_time_step(pg, eq, m), tend - t); ts(eq, pg, t, dt); t += dt; k += 1
        all(i -> isfinite(pg.rhos[i][1]) && abs(pg.rhos[i][1]) < 1e3, 1:pg.meta.N) || return (k, Inf, NaN, NaN)
    end
    ii = findall(i -> !pg.core.is_boundary[i], 1:pg.meta.N)
    u = [pg.rhos[i][1] for i in ii]
    e = sqrt(sum(i -> (pg.rhos[i][1] - f(mod.(pg.core.positions[i] .- t, 1.0)))^2, ii) / sum(i -> f(pg.core.positions[i])^2, ii))
    return (k, e, minimum(u), maximum(u))
end
sinf(x) = sinpi(2x[1]); sq(x) = abs(x[1] - 0.5) < 0.25 ? 1.0 : 0.0
println("smooth sine, RK4, no limiter: (steps, rel L2 error)")
for (o, R) in ((2, 2.5), (3, 2.5), (3, 3.5), (5, 2.5), (5, 3.5))
    print(rpad("MUSCL$o R$R", 14))
    for cfl in (1.0, 1.5, 2.0, 3.0, 4.0)
        k, e, _, _ = run(o, R, cfl, sinf); print(@sprintf("  cfl %.1f: %4d %s", cfl, k, isfinite(e) ? @sprintf("%.2e", e) : "BLOWUP  "))
    end
    println()
end
println("square wave, VK(:hard), no MOOD: (steps, min, max)")
for (o, R, tab, tn) in ((2, 2.5, RK3_SSP_Tableau(Float64), "RK3"), (3, 2.5, RK3_SSP_Tableau(Float64), "RK3"), (3, 2.5, RK2_SSP_Tableau(Float64), "RK2"), (3, 3.5, RK3_SSP_Tableau(Float64), "RK3"), (5, 3.5, RK3_SSP_Tableau(Float64), "RK3"))
    print(rpad("MUSCL$o R$R $tn", 18))
    for cfl in (0.4, 0.6, 0.8, 1.0)
        k, e, mn, mx = run(o, R, cfl, sq; lim = VenkatakrishnanLimiter(:hard), tab); print(@sprintf("  cfl %.1f: %4d [%.4f, %.4f]", cfl, k, mn, mx))
    end
    println()
end
