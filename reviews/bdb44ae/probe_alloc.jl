include(joinpath(@__DIR__, "header.jl"))
function measure(m, pg, deg)
    wd = m.div_weights; nb = pg.neighbor; invL = 1 / minimum(pg.meta.dx)
    f() = HC._moment_body!(1, wd.inv_moment, wd.ok, nb.ranges, nb.distances, nb.weights, invL, HC._fit_degree(wd), Val(true), deg)
    f(); return @allocated f()
end
for (D, o, R) in ((1, 3, 2.5), (1, 5, 3.5), (2, 3, 2.0), (2, 5, 2.5), (3, 3, 1.8), (3, 4, 2.0))
    pg = grid(D, Float64; n = D == 3 ? 6 : 12, R, jit = 0.3)
    m  = MUSCL(Float64, D, 1, o, NoLimiter(), RusanovFlux())
    HC.prepare_geometry!(m, pg)
    a1 = measure(m, pg, HC._FullOrderDivDegree{0, D, o}()); a0 = measure(m, pg, nothing)
    # whole prepare_geometry! after a version bump
    g() = (pg.meta.geometry_version += 1; HC.prepare_geometry!(m, pg))
    g(); a2 = @allocated g()
    t = @elapsed for _ in 1:20; g(); end
    println("D=$D MUSCL$o R=$R: _moment_body! alloc with degree_of $a1 B, without $a0 B; prepare_geometry! alloc $a2 B, $(round(t / 20 * 1e3, digits = 3)) ms/call (N=$(pg.meta.N))")
end
