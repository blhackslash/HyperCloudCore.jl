include(joinpath(@__DIR__, "header.jl"))
D, o, R, jit, n = 2, 3, 1.2, 0.4, 16
pg = grid(D, Float64; n, R, jit, seed = 7)
m = MUSCL(Float64, D, 1, o, NoLimiter(), RusanovFlux())
HC.prepare_geometry!(m, pg)
nb = pg.neighbor; invL = 1 / minimum(pg.meta.dx)
rng = MersenneTwister(1)
F = [Flux{2, 1, Float64}((State{1, Float64}(randn(rng)), State{1, Float64}(randn(rng)))) for _ in 1:length(nb.indices)]
scratch = zeros(State{1, Float64}, length(nb.indices))
worst = (0.0, 0)
for i in 1:pg.meta.N
    rg = nb.ranges[i]; isempty(rg) && continue
    q = HC._divergence_index(Val(0), Val(2), 3, length(rg)) - 1
    idx = q + 1
    dyn = HC.compute_dynamic_divergence(m.interpolators, idx, rg, nb.distances, nb.weights, F, scratch, pg.meta.dx)
    cac = HC.apply_weights_divergence(m.div_weights, i, rg, nb.distances, nb.weights, F)
    d = abs(dyn[1] - cac[1]) / max(abs(dyn[1]), 1e-300)
    if length(rg) < 3
        @printf("i=%3d nb=%d q=%d ok=%s  dynamic %.10e  cached %.10e  rel diff %.2e\n", i, length(rg), q, m.div_weights.ok[i], dyn[1], cac[1], d)
    end
    d > worst[1] && (global worst = (d, i))
end
println("worst rel diff over all particles: ", worst, " (nb = ", length(nb.ranges[worst[2]]), ")")
i = 126; rg = nb.ranges[i]
Ns = sum(nb.weights[k] * (p = HC._intercept_basis(Val(1), nb.distances[k] * invL); p * p') for k in rg)
C = cholesky(Symmetric(Ns), check = false)
println("particle 126: offsets/dx = ", [nb.distances[k] * invL for k in rg], " weights ", [nb.weights[k] for k in rg])
println("  eigvals(N) = ", eigvals(Symmetric(Matrix(Ns))), "  plain Cholesky accepted: ", HC._pivots_ok(C, Ns))
