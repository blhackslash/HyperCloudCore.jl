# Per-particle Σ|c|Δx: the fit in use (branch dweights), the full-degree cached fit (parent dweights), the Taylor fit
# of cached_weights = false; which particles set the step.
include(joinpath(@__DIR__, "header.jl"))
function sums(D, o, R, jit, n; seed = 1, per = true)
    bc = per ? nothing : Dict{Int, AbstractBoundaryCondition}(t => OutflowBC() for t in 1:(D == 2 ? 4 : 1))
    pg = grid(D, Float64; n, R, jit, seed, per, bc)
    m  = MUSCL(Float64, D, 1, o, NoLimiter(), RusanovFlux())
    HC.prepare_geometry!(m, pg)
    nb = pg.neighbor; invL = 1 / minimum(pg.meta.dx); dx = minimum(pg.meta.dx)
    qf = HC._divergence_degree(m)
    Bf = length(HC._generate_exponents(D, qf)) + 1
    use = Float64[]; full = Float64[]; cnt = Int[]; deg = Int[]
    for i in 1:pg.meta.N
        pg.core.is_boundary[i] && continue
        rg = nb.ranges[i]
        push!(use, sum(k -> sum(abs, m.dweights[k]), rg) * dx)
        Mi, ok = HC._inverse_moment(rg, nb.distances, nb.weights, invL, Val(qf), zero(SMatrix{Bf, Bf, Float64, Bf^2}), Val(true))
        s = 0.0
        for k in rg
            c = Mi * (nb.weights[k] * HC._intercept_basis(Val(qf), nb.distances[k] * invL))
            s += sum(d -> abs(c[d + 1]), 1:D) * invL
        end
        push!(full, s * dx); push!(cnt, length(rg)); push!(deg, HC._divergence_index(Val(0), Val(D), o, length(rg)) - 1)
    end
    eq = LinearAdvection(ntuple(d -> (1.0,), D), Float64)   # unit speeds: dt = 1/(D Σ|c|)
    dtc = get_time_step(pg, eq, m); dtu = get_time_step(pg, eq, m; cached_weights = false)
    iu = argmax(use); ifl = argmax(full)
    @printf("D=%d MUSCL%d R=%.1f jit=%.2f per=%-5s: max Σ|c|Δx in use %.3f (nb %d, deg %d) | full-degree cache %.3f (nb %d) | median %.3f vs %.3f | dt·n cached %.4f, Taylor %.4f | degrees %s\n",
            D, o, R, jit, per, use[iu], cnt[iu], deg[iu], full[ifl], cnt[ifl], sort(use)[end ÷ 2], sort(full)[end ÷ 2],
            dtc / dx, dtu / dx, sort([(q, count(==(q), deg)) for q in unique(deg)]))
end
for (D, o, R, jit, n) in ((1, 3, 2.5, 0.0, 100), (1, 3, 2.5, 0.2, 100), (1, 5, 2.5, 0.0, 100), (1, 5, 3.5, 0.0, 100), (1, 5, 3.5, 0.2, 100),
                          (2, 3, 2.0, 0.25, 32), (2, 3, 2.5, 0.2, 32), (2, 3, 2.5, 0.4, 32), (2, 5, 2.5, 0.0, 32), (2, 5, 2.5, 0.2, 32), (2, 5, 3.5, 0.2, 24), (2, 4, 2.5, 0.2, 32),
                          (3, 3, 2.0, 0.2, 10), (3, 3, 2.5, 0.2, 10))
    sums(D, o, R, jit, n)
end
sums(2, 5, 2.5, 0.2, 32; per = false)
sums(1, 3, 2.5, 0.2, 100; per = false)
