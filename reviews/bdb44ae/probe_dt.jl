# Δt before/after and runs to the end with Δt = CFL · get_time_step every step (as HyperCloud's solve_equation).
include(joinpath(@__DIR__, "header.jl"))
include(joinpath(@__DIR__, "euler.jl"))
function sod_exact(x, t; x0 = 0.5)
    g = 1.4; rl, pl, rr, pr = 1.0, 1.0, 0.125, 0.1
    ps, us = 0.30313017805064707, 0.9274526200489498
    cl = sqrt(g * pl / rl); rsl = rl * (ps / pl)^(1 / g); csl = sqrt(g * ps / rsl)
    rsr = rr * ((ps / pr + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * ps / pr + 1))
    ss = sqrt(g * pr / rr) * sqrt((g + 1) / (2g) * ps / pr + (g - 1) / (2g))
    xi = (x - x0) / t
    xi < -cl && return rl
    if xi < us - csl
        u = 2 / (g + 1) * (cl + xi); c = cl - (g - 1) / 2 * u
        return rl * (c / cl)^(2 / (g - 1))
    end
    xi < us && return rsl
    xi < ss && return rsr
    return rr
end
function run(name, D, eq::HyperbolicPDE{DD, MM}, mk, init!; n, R, jit = 0.0, per = true, cfl, tend, mood = MOOD(), tab = RK3_SSP_Tableau(Float64), err = nothing, seed = 1) where {DD, MM}
    bc = per ? nothing : Dict{Int, AbstractBoundaryCondition}(t => OutflowBC() for t in 1:(D == 2 ? 4 : 1))
    pg = grid(D, Float64; n, R, jit, per, bc, seed, M = MM)
    init!(pg)
    sc = mk()
    ts = GeneralRKTimeStepper(eq, sc, mood, tab)
    dt0 = get_time_step(pg, eq, sc)
    t = 0.0; k = 0; dt1 = NaN; ok = true; mind = Inf; minp = Inf
    el = @elapsed while t < tend - 1e-12
        dt = min(cfl * get_time_step(pg, eq, sc), tend - t)
        ts(eq, pg, t, dt); t += dt; k += 1
        k == 1 && (dt1 = get_time_step(pg, eq, sc))
        if !all(i -> all(isfinite, pg.rhos[i]), 1:pg.meta.N); ok = false; break; end
        if eq isa EulerEq
            for i in interior_indices(pg); mind = min(mind, pg.rhos[i][1]); minp = min(minp, pressure(eq, pg.rhos[i])); end
        end
        k > 20000 && break
    end
    ii = interior_indices(pg)
    e = err === nothing ? NaN : err(pg, ii, t)
    u1 = [pg.rhos[i][1] for i in ii]
    @printf("%-34s cfl=%.2f dt0·n=%.5f dt1·n=%.5f steps=%5d finite=%-5s err=%.4e  min=%.4f max=%.4f minρ=%.4f minp=%.4f  %.1fs\n",
            name, cfl, dt0 * n, dt1 * n, k, ok, e, minimum(u1), maximum(u1), mind, minp, el)
    return u1
end
adv(D) = LinearAdvection(ntuple(d -> (d == 1 ? 1.0 : 0.5,), D), Float64)
setf!(f) = pg -> (for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f(pg.core.positions[i])); end)
sinf(x) = sinpi(2x[1]) * (length(x) > 1 ? sinpi(2x[2]) : 1.0)
sq(x) = abs(x[1] - 0.5) < 0.25 ? 1.0 : 0.0
l2err(f, c) = (pg, ii, t) -> sqrt(sum(i -> (pg.rhos[i][1] - f(mod.(pg.core.positions[i] .- c .* t, 1.0)))^2, ii) / sum(i -> f(mod.(pg.core.positions[i] .- c .* t, 1.0))^2, ii))
l1sod(pg, ii, t) = sum(i -> abs(pg.rhos[i][1] - sod_exact(pg.core.positions[i][1], t)), ii) / length(ii)
e1 = EulerEq(Val(1), Float64)
sod!(pg) = for i in 1:pg.meta.N
    x = pg.core.positions[i][1]
    pg.rhos[i] = HC.prim2cons(e1, x < 0.5 ? State{3, Float64}(1.0, 0.0, 1.0) : State{3, Float64}(0.125, 0.0, 0.1))
end
e2 = EulerEq(Val(2), Float64)
blast!(pg) = for i in 1:pg.meta.N
    x = pg.core.positions[i]; r = sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2)
    pg.rhos[i] = HC.prim2cons(e2, r < 0.2 ? State{4, Float64}(1.0, 0.0, 0.0, 1.0) : State{4, Float64}(0.125, 0.0, 0.0, 0.1))
end
res = Dict{String, Any}()
VK = VenkatakrishnanLimiter(:hard)
for cfl in (0.4, 0.6, 0.8, 1.0)
    res["A $cfl"] = run("1D adv sin MUSCL3 R2.5", 1, adv(1), () -> MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux()), setf!(sinf); n = 100, R = 2.5, cfl, tend = 1.0, err = l2err(sinf, (1.0,)))
    res["B $cfl"] = run("1D adv sin MUSCL3 R2.5 jit0.2", 1, adv(1), () -> MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux()), setf!(sinf); n = 100, R = 2.5, jit = 0.2, cfl, tend = 1.0, err = l2err(sinf, (1.0,)))
    res["C $cfl"] = run("1D adv sq MUSCL3 VK u1", 1, adv(1), () -> MUSCL(Float64, 1, 1, 3, VK, RusanovFlux()), setf!(sq); n = 100, R = 2.5, cfl, tend = 1.0, mood = MOOD(MOODu1(1e-8)), err = l2err(sq, (1.0,)))
    res["C2 $cfl"] = run("1D adv sq MUSCL3 VK noMOOD", 1, adv(1), () -> MUSCL(Float64, 1, 1, 3, VK, RusanovFlux()), setf!(sq); n = 100, R = 2.5, cfl, tend = 1.0, err = l2err(sq, (1.0,)))
    res["D $cfl"] = run("1D Sod MUSCL3 R2.5 VK PAD", 1, e1, () -> MUSCL(Float64, 1, 3, 3, VK, RusanovFlux()), sod!; n = 200, R = 2.5, per = false, cfl, tend = 0.2, mood = MOOD(MOODPAD()), err = l1sod)
    res["D2 $cfl"] = run("1D Sod MUSCL3 R2.5 VK u1", 1, e1, () -> MUSCL(Float64, 1, 3, 3, VK, RusanovFlux()), sod!; n = 200, R = 2.5, per = false, cfl, tend = 0.2, mood = MOOD(MOODu1(1e-4)), err = l1sod)
    res["D3 $cfl"] = run("1D Sod MUSCL3 R2.5 jit VK PAD", 1, e1, () -> MUSCL(Float64, 1, 3, 3, VK, RusanovFlux()), sod!; n = 200, R = 2.5, jit = 0.2, per = false, cfl, tend = 0.2, mood = MOOD(MOODPAD()), err = l1sod)
    res["E $cfl"] = run("1D Sod MUSCL5 R3.5 VK PAD", 1, e1, () -> MUSCL(Float64, 1, 3, 5, VK, RusanovFlux()), sod!; n = 200, R = 3.5, per = false, cfl, tend = 0.2, mood = MOOD(MOODPAD()), err = l1sod)
    res["E2 $cfl"] = run("1D adv sin MUSCL5 R3.5 RK4", 1, adv(1), () -> MUSCL(Float64, 1, 1, 5, NoLimiter(), RusanovFlux()), setf!(sinf); n = 100, R = 3.5, cfl, tend = 1.0, tab = RK4_Classical_Tableau(Float64), err = l2err(sinf, (1.0,)))
end
for cfl in (0.4, 0.8)
    res["F $cfl"] = run("2D adv sin MUSCL5 R2.5 jit0.2", 2, adv(2), () -> MUSCL(Float64, 2, 1, 5, NoLimiter(), RusanovFlux()), setf!(sinf); n = 32, R = 2.5, jit = 0.2, cfl, tend = 0.5, err = l2err(sinf, (1.0, 0.5)))
    res["G $cfl"] = run("2D blast MUSCL5 R2.5 jit VK PAD", 2, e2, () -> MUSCL(Float64, 2, 4, 5, VK, RusanovFlux()), blast!; n = 40, R = 2.5, jit = 0.2, per = false, cfl, tend = 0.1, mood = MOOD(MOODPAD()))
    res["H $cfl"] = run("2D blast MUSCL3 R2.5 jit VK PAD", 2, e2, () -> MUSCL(Float64, 2, 4, 3, VK, RusanovFlux()), blast!; n = 40, R = 2.5, jit = 0.2, per = false, cfl, tend = 0.1, mood = MOOD(MOODPAD()))
end
serialize(joinpath(@__DIR__, "dt_$(TAG).jls"), res)
