# Probe: do the limited MUSCL reconstructions stay inside the local bounds [min, max] of the neighbourhood, as the
# limiter docstring ("enforce monotonicity") suggests? Degree-2 reconstruction (MUSCL order 3), smoothed step.
include(joinpath(@__DIR__, "_prelude.jl"))

function max_overshoot(limiter; order = 3, N = 60, ic = x -> 0.5 * (1 + tanh(x / 0.04)))
    pg = make_pg_1d(N); eq = LinearAdvection(((1.0,),), Float64)
    for i in 1:N; pg.rhos[i] = State{1, Float64}(ic(pg.core.positions[i][1])); end
    m  = MUSCL(Float64, 1, 1, order, limiter, RusanovFlux())
    ts = GeneralRKTimeStepper(eq, m, MOOD(), RK3_SSP_Tableau(Float64))
    HyperCloudCore.update_size!(ts, N, length(pg.neighbor.indices), false); HyperCloudCore.update_size!(m, N)
    fill!(view(pg.core.particle_orders, 1:N), order)
    HyperCloudCore.compute_explicit_rhs!(ts, eq, pg, ts.K_E_stages[1], ts.K_X_stages[1], pg.rhos, 0.0, true)
    nb = pg.neighbor; worst = 0.0
    for i in 1:N
        u = [pg.rhos[nb.indices[k]][1] for k in nb.ranges[i]]; push!(u, pg.rhos[i][1])
        lo, hi = extrema(u)
        for k in nb.ranges[i]
            p = HyperCloudCore.build_basis(Val(order - 1), 0.5 * nb.distances[k])
            r = pg.rhos[i][1] + sum(m.gradients[i][b][1] * p[b] for b in eachindex(p))
            worst = max(worst, r - hi, lo - r)
        end
    end
    return worst
end

@testset "limited reconstructions stay in the local bounds" begin
    ics = (("tanh step", x -> 0.5 * (1 + tanh(x / 0.04))), ("box", x -> abs(x) < 0.3 ? 1.0 : 0.0),
           ("kink |x|", x -> abs(x)), ("narrow peak", x -> exp(-x^2 / 0.002)), ("box+ramp", x -> x < 0 ? 0.0 : x < 0.2 ? 5x : 0.5))
    for (name, ic) in ics, lim in (BarthJespersenLimiter(:soft), BarthJespersenLimiter(:hard), VenkatakrishnanLimiter(:soft), VenkatakrishnanLimiter(:hard),
                MinmodLimiter(:soft), MinmodLimiter(:hard), NoLimiter())
        w = max_overshoot(lim; ic = ic)
        @printf("%-12s %-32s max overshoot of the midpoint reconstruction: %.3e\n", name, string(typeof(lim)), w)
        lim isa NoLimiter || @test w <= 1e-12
    end
end
