# KA CPU (PoCL) vs Polyester: the padded-inverse kernel (_moment_body! with a degree function), dweights, RHS and
# get_time_step (which now launches prepare_geometry!) on stencils where the safeguard lowers the degree.
include(joinpath(@__DIR__, "header.jl"))
smooth(x) = sinpi(2x[1]) * (length(x) > 1 ? cospi(2x[2]) : 1.0)
function run(backend, D, T, mk; n, R, jit, steps = 3, per = true)
    set_backend!(backend)
    try
        bc = per ? nothing : Dict{Int, AbstractBoundaryCondition}(t => OutflowBC() for t in 1:2D)
        pg = grid(D, T; n, R, jit, per, bc)
        for i in 1:pg.meta.N; pg.rhos[i] = State{1, T}(T(smooth(pg.core.positions[i]))); end
        eq = LinearAdvection(ntuple(d -> (T(d == 1 ? 1 : 0.5),), D), T)
        sc = mk(T)
        dt0 = get_time_step(pg, eq, sc)                    # prepare_geometry! from get_time_step, on this backend
        dw = copy(sc.dweights[1:length(pg.neighbor.indices)]); im = copy(sc.div_weights.inv_moment[1:pg.meta.N])
        ts = GeneralRKTimeStepper(eq, sc, MOOD(), RK3_SSP_Tableau(T))
        dt = T(0.2 / n)
        for k in 0:(steps - 1); ts(eq, pg, T(k * dt), dt); end
        slots = reduce(vcat, [collect(pg.neighbor.ranges[i]) for i in 1:pg.meta.N]); return [pg.rhos[i][1] for i in 1:pg.meta.N], dt0, dw[slots], im
    finally
        set_backend!(PolyesterBackend())
    end
end
ka = KABackend(HC.KA.CPU(); workgroupsize = 32)
cases = Any[
    ("1-D MUSCL3 R2.5 jit", 1, T -> MUSCL(T, 1, 1, 3, NoLimiter(), RusanovFlux()), (; n = 40, R = 2.5, jit = 0.25)),
    ("1-D MUSCL5 R2.5 aligned streamed walls", 1, T -> MUSCL(T, 1, 1, 5, NoLimiter(), AlignedRusanovFlux(); streamed = true), (; n = 40, R = 2.5, jit = 0.25, per = false)),
    ("2-D MUSCL3 R2.0 jit aligned", 2, T -> MUSCL(T, 2, 1, 3, VenkatakrishnanLimiter(:hard), AlignedRusanovFlux()), (; n = 12, R = 2.0, jit = 0.3)),
    ("2-D MUSCL5 R2.5 jit streamed", 2, T -> MUSCL(T, 2, 1, 5, NoLimiter(), RusanovFlux(); streamed = true), (; n = 10, R = 2.5, jit = 0.2)),
    ("3-D MUSCL3 R1.8 jit", 3, T -> MUSCL(T, 3, 1, 3, NoLimiter(), RusanovFlux()), (; n = 6, R = 1.8, jit = 0.2)),
]
@testset "KA vs Polyester: $name $T" for (name, D, mk, kw) in cases, T in (Float64, Float32)
    t1 = @elapsed (ref = run(PolyesterBackend(), D, T, mk; kw...))
    t2 = @elapsed (dev = run(ka, D, T, mk; kw...))
    scale = maximum(abs, ref[1])
    err = maximum(abs.(dev[1] .- ref[1])) / scale
    dterr = abs(dev[2] - ref[2]) / ref[2]
    dwerr = maximum(k -> maximum(abs, dev[3][k] - ref[3][k]), eachindex(ref[3])) / maximum(k -> maximum(abs, ref[3][k]), eachindex(ref[3]))
    imerr = maximum(i -> maximum(abs, dev[4][i] - ref[4][i]) / max(maximum(abs, ref[4][i]), floatmin(T)), eachindex(ref[4]))
    @printf("%-42s %-7s state %.2e  dt %.2e  dweights %.2e  inv_moment %.2e  (Polyester %.1fs, KA %.1fs)\n", name, T, err, dterr, dwerr, imerr, t1, t2)
    @test all(isfinite, dev[1])
    @test err < 1000eps(T) && dterr < 1000eps(T)
end
