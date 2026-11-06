# Starved MUSCL particles: 2-D periodic jittered cloud with a small radius (many particles with 2-8 neighbours);
# smooth advection error and boundedness of a box, old vs new code
include(joinpath(@__DIR__, "common.jl"))
eq = Adv{2, Float64}(SVector(1.0, 0.5))
f0(x) = sin(2pi * x[1]) * sin(2pi * x[2])
box(x) = (abs(x[1] - 0.5) < 0.2 && abs(x[2] - 0.5) < 0.2) ? 1.0 : 0.0
for (order, R) in ((3, 1.6), (4, 1.6), (4, 2.0)), lim in (NoLimiter(), VenkatakrishnanLimiter(:hard))
    for n in (24,)
        pg = grid2d(n; range = R, jitter = 0.25, seed = 3)
        N  = pg.meta.N
        nnb = [length(pg.neighbor.ranges[i]) for i in 1:N]
        setic!(pg, f0)
        m  = MUSCL(Float64, 2, 1, order, lim, RusanovFlux())
        ts = GeneralRKTimeStepper(eq, m, MOOD(), RK3_SSP_Tableau(Float64))
        dt0 = get_time_step(pg, eq, m); nst = 300; dt = 0.3 * dt0; tend = nst * dt
        t = 0.0
        for k in 1:nst; ts(eq, pg, t, dt); t += dt; end
        err = sqrt(sum(abs2(pg.rhos[i][1] - f0(pg.core.positions[i] - SVector(1.0, 0.5) * tend)) for i in 1:N) / N)
        ords = [count(==(o), view(pg.core.particle_orders, 1:N)) for o in 1:order]
        # box: max overshoot after the same steps
        setic!(pg, box); t = 0.0
        for k in 1:nst; ts(eq, pg, t, dt); t += dt; end
        ub = [pg.rhos[i][1] for i in 1:N]
        flush(stdout); println("MUSCL$order R=$(R)dx $(nameof(typeof(lim))) n=$n: nb count min/max $(extrema(nnb)), starved(<basis) $(count(<(HCC.basis_count(2, order - 1)), nnb)); ",
                "CFL dt/dx $(round(dt0 * n; sigdigits = 3)); orders after step $ords; t = $(round(tend; sigdigits = 3)), smooth L2 err $(round(err; sigdigits = 4)); box extrema $(round.(extrema(ub); sigdigits = 4))")
    end
end
