# Particles with fewer than D + 1 neighbours (degree 1 is under-determined): fixed-dt states and Δt, both trees.
include(joinpath(@__DIR__, "header.jl"))
res = Dict{String, Any}()
for (D, o, R, jit, n) in ((1, 3, 1.2, 0.45, 40), (2, 3, 1.2, 0.4, 16), (2, 5, 1.5, 0.4, 16))
    pg = grid(D, Float64; n, R, jit, seed = 7)
    cnt = [length(pg.neighbor.ranges[i]) for i in 1:pg.meta.N]
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(sinpi(2pg.core.positions[i][1])); end
    eq = LinearAdvection(ntuple(d -> (d == 1 ? 1.0 : 0.5,), D), Float64)
    m = MUSCL(Float64, D, 1, o, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eq, m, MOOD(), RK3_SSP_Tableau(Float64))
    dtc = get_time_step(pg, eq, m)
    for k in 0:4; ts(eq, pg, k * 0.05 / n, 0.05 / n); end
    name = "D$D MUSCL$o R$R"
    res[name] = [pg.rhos[i][1] for i in 1:pg.meta.N]
    @printf("%-20s particles with < D+1 neighbours: %d (min %d), ok=false: %d, dt·n = %.5f, finite = %s\n", name,
            count(<(D + 1), cnt), minimum(cnt), count(i -> !m.div_weights.ok[i], 1:pg.meta.N), dtc * n, all(isfinite, res[name]))
end
serialize(joinpath(@__DIR__, "starved_$(TAG).jls"), res)
