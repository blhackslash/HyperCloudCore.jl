# Moving grids (non-uniform prescribed velocity, frozen topology and full search per stage): fixed-dt states for
# parent-vs-branch comparison, and the Δt sequence of a CFL-driven run.
include(joinpath(@__DIR__, "header.jl"))
smooth(x) = sinpi(2x[1]) * (length(x) > 1 ? cospi(2x[2]) : 1.0)
wfield(x, t) = typeof(x)(ntuple(d -> 0.3 * sinpi(2x[1]) * (d == 1 ? 1 : 0.5), length(x)))
res = Dict{String, Any}()
for (D, o, R, n) in ((1, 3, 2.5, 40), (2, 3, 2.0, 16), (2, 5, 2.5, 12)), fs in (false, true)
    name = "D$D MUSCL$o R$R full_search=$fs"
    pg = grid(D, Float64; n, R, jit = 0.25, mover = PrescribedMover(wfield))
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(smooth(pg.core.positions[i])); end
    eq = LinearAdvection(ntuple(d -> (d == 1 ? 1.0 : 0.5,), D), Float64)
    m  = MUSCL(Float64, D, 1, o, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eq, m, MOOD(), RK3_SSP_Tableau(Float64); full_search_per_stage = fs)
    dt = 0.2 / n
    for k in 0:4; ts(eq, pg, k * dt, dt); end
    res[name] = [pg.rhos[i][1] for i in 1:pg.meta.N]
    # CFL-driven: Δt sequence
    dts = Float64[]; t = 5dt
    for k in 1:5
        h = 0.6 * get_time_step(pg, eq, m); push!(dts, h * n)
        ts(eq, pg, t, h); t += h
    end
    res[name * " dts"] = dts
    println(rpad(name, 32), " finite=", all(isfinite, res[name]), " dt·n (CFL 0.6) = ", round.(dts, digits = 5),
            "  cache current after step: ", HC.is_current(m.div_weights, pg))
end
serialize(joinpath(@__DIR__, "moving_$(TAG).jls"), res)
