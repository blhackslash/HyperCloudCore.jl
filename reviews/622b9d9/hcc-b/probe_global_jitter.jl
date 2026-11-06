# Probe: global convergence of MUSCL 2-4 on a jittered periodic 1-D cloud (linear advection of a smooth wave,
# RK4 classical, CFL 0.3, t = 0.5). The pipeline EOC tests only use lattices (Grid_randomness_factor = 0).
include(joinpath(@__DIR__, "_prelude.jl"))

function global_error(order, n; jit = 0.25, R = 3.5, tend = 0.5, moment = false)
    dx = 2 / n
    geom = get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true)
    pg = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, R * dx), 1; randomness = (jit * dx,), rng = MersenneTwister(11))
    eq = LinearAdvection(((1.0,),), Float64)
    u0(x) = sin(π * x) + 0.5cos(2π * x)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(u0(pg.core.positions[i][1])); end
    m = MUSCL(Float64, 1, 1, order, NoLimiter(), order == 1 ? AlignedRusanovFlux() : RusanovFlux(); moment_correction = moment)
    ts = GeneralRKTimeStepper(eq, m, MOOD(), RK4_Classical_Tableau(Float64))
    t = 0.0
    while t < tend - 1e-12
        dt = min(0.3 * dx, tend - t); ts(eq, pg, t, dt); t += dt
    end
    xs = [pg.core.positions[i][1] for i in 1:pg.meta.N]
    return maximum(abs(pg.rhos[i][1] - u0(xs[i] - tend)) for i in 1:pg.meta.N)
end

@testset "global EOC on jittered 1-D clouds" begin
    for (order, moment) in ((1, false), (1, true), (2, false), (3, false), (4, false))
        e = [global_error(order, n; moment) for n in (80, 160, 320)]
        @printf("MUSCL %d%s: err %.2e %.2e %.2e  EOC %.2f %.2f\n", order, moment ? " (moment)" : "", e..., log2(e[1] / e[2]), log2(e[2] / e[3]))
    end
end
