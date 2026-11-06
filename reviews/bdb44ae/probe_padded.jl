# Unit-level checks of the padded inverse (branch only).
include(joinpath(@__DIR__, "header.jl"))
# (a) graded prefix of the basis, every D ≤ 3, q ≤ IO ≤ 5 (3-D: 3), with and without intercept
rng = MersenneTwister(3)
bad = 0
for D in 1:3, IO in 1:(D == 3 ? 4 : 5), q in 1:IO, C in (false, true)
    d = SVector{D, Float64}(randn(rng, D))
    full = HC._fit_basis(Val(C), Val(IO), d); part = HC._fit_basis(Val(C), Val(q), d)
    global bad += full[1:length(part)] == part ? 0 : 1
end
println("(a) graded prefix mismatches: $bad")
# (b) _padded_inverse_moment(q) == pad(_inverse_moment(Val(q))) bitwise, every q incl. out-of-range q (0, IO+1)
function check_pad(D, T, IO, C; n = 6, R = 2.5, jit = 0.3)
    pg = grid(D, T; n, R, jit)
    nb = pg.neighbor; invL = one(T) / minimum(pg.meta.dx)
    B  = length(HC._generate_exponents(D, IO)) + C
    tmpl = zero(SMatrix{B, B, T, B * B})
    nbad = 0; okmis = 0
    for i in 1:pg.meta.N, q in 0:(IO + 1)
        rg = nb.ranges[i]
        P, ok = HC._padded_inverse_moment(q, rg, nb.distances, nb.weights, invL, Val(IO), tmpl, Val(C))
        qq = clamp(q, 1, IO); qq = q == 0 ? IO : qq          # q outside 1..IO-1 falls to the full degree
        q > IO && (qq = IO)
        Bq = length(HC._generate_exponents(D, qq)) + C
        Mi, ok2 = HC._inverse_moment(rg, nb.distances, nb.weights, invL, Val(qq), zero(SMatrix{Bq, Bq, T, Bq * Bq}), Val(C))
        ref = [r <= Bq && c <= Bq ? Mi[r, c] : zero(T) for r in 1:B, c in 1:B]
        nbad += Matrix(P) == ref ? 0 : 1
        okmis += ok == ok2 ? 0 : 1
    end
    return nbad, okmis
end
for D in 1:3, T in (Float64, Float32), C in (false, true)
    IO = D == 3 ? 3 : 5
    nbad, okmis = check_pad(D, T, IO, C; n = D == 3 ? 4 : (D == 2 ? 8 : 20), R = D == 1 ? 3.5 : 2.5)
    println("(b) D=$D $T C=$C IO=$IO: padded != inverse(q) padded: $nbad, ok mismatches: $okmis")
end
# (c) inference and allocation of the kernel body
pg = grid(2, Float64; n = 10, R = 2.5, jit = 0.3)
m  = MUSCL(Float64, 2, 1, 5, NoLimiter(), RusanovFlux())
wd = m.div_weights
HC.prepare_geometry!(m, pg)
nb = pg.neighbor; invL = 1 / minimum(pg.meta.dx)
deg = HC._FullOrderDivDegree{0, 2, 5}()
tm = zero(SMatrix{21, 21, Float64, 441})
@inferred HC._padded_inverse_moment(3, nb.ranges[1], nb.distances, nb.weights, invL, Val(5), tm, Val(true))
a = @allocated HC._moment_body!(1, wd.inv_moment, wd.ok, nb.ranges, nb.distances, nb.weights, invL, Val(5), Val(true), deg)
a = @allocated HC._moment_body!(1, wd.inv_moment, wd.ok, nb.ranges, nb.distances, nb.weights, invL, Val(5), Val(true), deg)
println("(c) _moment_body! allocation (2-D, IO = 5): $a bytes; isbits(deg) = $(isbits(deg))")
# (d) per-slot dweights vs the per-slot weights of the dynamic fit of the degree each particle uses, jittered clouds
function check_dw(D, T, o; n, R, jit, per = true, div_order = 0)
    bc = per ? nothing : Dict{Int, AbstractBoundaryCondition}(t => OutflowBC() for t in 1:2D)
    pg = grid(D, T; n, R, jit, per, bc)
    m  = MUSCL(T, D, 1, o, NoLimiter(), RusanovFlux(); div_order)
    HC.prepare_geometry!(m, pg)
    nb = pg.neighbor; invL = one(T) / minimum(pg.meta.dx)
    full = HC._full_div_idx(Val(o), Val(div_order), Val(D))
    degs = Dict{Int, Int}(); maxrel = zero(T); okmis = 0; nok = 0
    for i in 1:pg.meta.N
        rg = nb.ranges[i]; isempty(rg) && continue
        q  = HC._divergence_index(Val(div_order), Val(D), o, length(rg)) - 1
        degs[q] = get(degs, q, 0) + 1
        # dynamic per-slot weights (same as _slot_weight(DynamicWeights)): unit flux differences through _mls_divergence
        B0 = length(HC._generate_exponents(D, q))
        Ns = zero(SMatrix{B0 + 1, B0 + 1, T, (B0 + 1)^2})
        for k in rg; p = HC._intercept_basis(Val(q), nb.distances[k] * invL); Ns += nb.weights[k] * (p * p'); end
        Cf, ok = HC._cholesky_moment(Ns)
        okmis += ok == m.div_weights.ok[i] ? 0 : 1
        ok || continue
        nok += 1
        sc = maximum(k -> maximum(abs, m.dweights[k]), rg)
        for k in rg
            c = Cf \ (nb.weights[k] * HC._intercept_basis(Val(q), nb.distances[k] * invL))
            w = Space{D, T}(ntuple(d -> c[d + 1], Val(D))) * invL
            maxrel = max(maxrel, maximum(abs, w - m.dweights[k]) / sc)
        end
    end
    println(rpad("(d) D=$D $T MUSCL$o R=$R jit=$jit per=$per DO=$div_order", 52), " degrees used: ", sort(collect(degs)),
            "  max rel |dw - dynamic| = ", maxrel, "  ok mismatches: ", okmis)
end
check_dw(1, Float64, 3; n = 40, R = 2.5, jit = 0.0)
check_dw(1, Float64, 3; n = 40, R = 2.5, jit = 0.3)
check_dw(1, Float64, 5; n = 40, R = 2.5, jit = 0.3)
check_dw(1, Float64, 5; n = 40, R = 3.5, jit = 0.3, per = false)
check_dw(1, Float32, 5; n = 40, R = 3.5, jit = 0.3)
check_dw(1, Float64, 3; n = 40, R = 1.2, jit = 0.35)
check_dw(2, Float64, 3; n = 16, R = 2.0, jit = 0.3)
check_dw(2, Float64, 5; n = 16, R = 2.5, jit = 0.3, per = false)
check_dw(2, Float32, 5; n = 16, R = 2.5, jit = 0.3)
check_dw(2, Float64, 5; n = 16, R = 2.5, jit = 0.3, div_order = 4)
check_dw(3, Float64, 3; n = 7, R = 1.8, jit = 0.2)
check_dw(3, Float64, 4; n = 7, R = 2.0, jit = 0.2, per = false)
