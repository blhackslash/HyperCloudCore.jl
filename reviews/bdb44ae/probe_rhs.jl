# Fixed-dt runs (same dt on both trees): states after a few RK3 steps, saved for comparison parent vs branch.
include(joinpath(@__DIR__, "header.jl"))
smooth(x) = sinpi(2x[1]) * (length(x) > 1 ? cospi(2x[2]) : 1.0) + (length(x) > 2 ? 0.3sinpi(2x[3]) : 0.0)
box(x) = all(abs.(x .- 0.5) .< 0.2) ? 1.0 : 0.0
function run_case(D, T, mk, f; n, R, jit, steps = 5, mood = MOOD(), per = true, mover = NoGridMover(), seed = 1)
    bc = per ? nothing : Dict{Int, AbstractBoundaryCondition}(t => OutflowBC() for t in 1:2D)
    pg = grid(D, T; n, R, jit, per, bc, mover, seed)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, T}(T(f(pg.core.positions[i]))); end
    eq = LinearAdvection(ntuple(d -> (T(d == 1 ? 1 : 0.5),), D), T)
    sc = mk(T)
    ts = GeneralRKTimeStepper(eq, sc, mood, RK3_SSP_Tableau(T))
    dt = T(0.2 / n)
    t0 = time()
    for k in 0:(steps - 1); ts(eq, pg, T(k * dt), dt); end
    el = time() - t0
    nnb = [length(pg.neighbor.ranges[i]) for i in 1:pg.meta.N]
    full = HC._full_div_idx(Val(HC._extract_order(sc)), Val(typeof(sc).parameters[6]), Val(D))
    lowered = count(i -> !pg.core.is_boundary[i] && HC._divergence_index(Val(typeof(sc).parameters[6]), Val(D), HC._extract_order(sc), nnb[i]) < full, 1:pg.meta.N)
    return [pg.rhos[i][1] for i in 1:pg.meta.N], lowered, el
end
M(o; lim = NoLimiter(), fl = RusanovFlux(), kw...) = T -> MUSCL(T, D_[], 1, o, lim, fl; kw...)
const D_ = Ref(1)
cases = Any[]
for o in (2, 3, 4, 5), R in (2.5, 3.5), jit in (0.0, 0.25)
    push!(cases, ("1D MUSCL$o R$R jit$jit", 1, Float64, (o,), smooth, (; n = 40, R, jit)))
end
push!(cases, ("1D MUSCL3 R2.5 streamed", 1, Float64, (3, (; streamed = true)), smooth, (; n = 40, R = 2.5, jit = 0.25)))
push!(cases, ("1D MUSCL3 R2.5 aligned", 1, Float64, (3, (; fl = AlignedRusanovFlux())), smooth, (; n = 40, R = 2.5, jit = 0.25)))
push!(cases, ("1D MUSCL5 R2.5 aligned streamed", 1, Float64, (5, (; fl = AlignedRusanovFlux(), streamed = true)), smooth, (; n = 40, R = 2.5, jit = 0.25)))
push!(cases, ("1D MUSCL5 R3.5 div_order3", 1, Float64, (5, (; div_order = 3)), smooth, (; n = 40, R = 2.5, jit = 0.25)))
push!(cases, ("1D MUSCL1 moment aligned", 1, Float64, (1, (; fl = AlignedRusanovFlux(), moment_correction = true)), smooth, (; n = 40, R = 2.5, jit = 0.25)))
push!(cases, ("1D MUSCL3 VK MOODu1 box", 1, Float64, (3, (; lim = VenkatakrishnanLimiter(:hard))), box, (; n = 40, R = 2.5, jit = 0.25, mood = MOOD(MOODu1(1e-6)))))
push!(cases, ("1D MUSCL5 VK MOODu1 box EPD0", 1, Float64, (5, (; lim = VenkatakrishnanLimiter(:hard))), box, (; n = 40, R = 2.5, jit = 0.25, mood = MOOD(EPD0(), MOODu1(1e-6)))))
push!(cases, ("1D MUSCL3 walls", 1, Float64, (3,), smooth, (; n = 40, R = 2.5, jit = 0.25, per = false)))
push!(cases, ("1D MUSCL3 R2.5 F32", 1, Float32, (3,), smooth, (; n = 40, R = 2.5, jit = 0.25)))
push!(cases, ("1D MUSCL3 R2.5 Lagrangian", 1, Float64, (3,), smooth, (; n = 40, R = 2.5, jit = 0.25, mover = LagrangianMover())))
push!(cases, ("1D MUSCL1 R1.2 starved", 1, Float64, (2,), smooth, (; n = 40, R = 1.2, jit = 0.3)))
push!(cases, ("2D MUSCL3 R2.5 jit0.3", 2, Float64, (3,), smooth, (; n = 16, R = 2.5, jit = 0.3)))
push!(cases, ("2D MUSCL3 R2.0 jit0.3", 2, Float64, (3,), smooth, (; n = 16, R = 2.0, jit = 0.3)))
push!(cases, ("2D MUSCL5 R2.5 jit0.2", 2, Float64, (5,), smooth, (; n = 16, R = 2.5, jit = 0.2)))
push!(cases, ("2D MUSCL5 R2.5 walls aligned streamed", 2, Float64, (5, (; fl = AlignedRusanovFlux(), streamed = true)), smooth, (; n = 16, R = 2.5, jit = 0.2, per = false)))
push!(cases, ("2D MUSCL3 R2.0 VK MOODu1 box", 2, Float64, (3, (; lim = VenkatakrishnanLimiter(:hard))), box, (; n = 16, R = 2.0, jit = 0.3, mood = MOOD(MOODu1(1e-6)))))
push!(cases, ("2D MUSCL3 R2.0 F32 aligned", 2, Float32, (3, (; fl = AlignedRusanovFlux())), smooth, (; n = 16, R = 2.0, jit = 0.3)))
push!(cases, ("2D MUSCL3 R2.0 Lagrangian", 2, Float64, (3,), smooth, (; n = 16, R = 2.0, jit = 0.3, mover = LagrangianMover())))
push!(cases, ("3D MUSCL3 R1.8 jit0.2", 3, Float64, (3,), smooth, (; n = 8, R = 1.8, jit = 0.2)))
push!(cases, ("3D MUSCL4 R2.0 jit0.2 streamed", 3, Float64, (4, (; streamed = true)), smooth, (; n = 8, R = 2.0, jit = 0.2)))
res = Dict{String, Any}()
for (name, D, T, spec, f, kw) in cases
    D_[] = D
    o = spec[1]; skw = length(spec) > 1 ? spec[2] : (;)
    mk = M(o; skw...)
    kw2 = Base.structdiff(kw, NamedTuple{(:mood,)})
    mood = haskey(kw, :mood) ? kw.mood : MOOD()
    u, low, el = run_case(D, T, mk, f; mood, kw2...)
    res[name] = u
    @printf("%-40s lowered=%4d  finite=%s  max|u|=%.4f  t=%.2fs\n", name, low, all(isfinite, u), maximum(abs, u), el)
end
serialize(joinpath(@__DIR__, "rhs_$(TAG).jls"), res)
