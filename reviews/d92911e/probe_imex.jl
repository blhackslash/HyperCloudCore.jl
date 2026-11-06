# 3e00390: the stepper-level IMEX order test with a space-dependent decay rate λ(1 + κ sin πx).
# Does it now detect a pair that violates each coupling condition but has the right sum (b̃·c = 0.7, b·c̃ = 0.3)?
# And one that violates only b·c̃ (sum wrong)? Cost of the test's runs.
include(joinpath(@__DIR__, "header.jl"))
struct DecaySource <: AbstractImplicitSourceTerm; λ::Float64; κ::Float64; end
rate(s::DecaySource, i, pg) = s.λ * (1 + s.κ * sinpi(pg.core.positions[i][1]))
HC.evaluate_source(s::DecaySource, U, i::Int, pg, t::Real) = -rate(s, i, pg) * U
HC.implicit_solve(s::DecaySource, U_in, γ::Real, i::Int, pg, t::Real) = U_in / (1 + γ * rate(s, i, pg))

function make_pg_1d(N)
    dx = 2 / N
    geom = get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true)
    ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1)
end
eq = LinearAdvection(((1.0,),), Float64)
function imex_run(bt, src, n; tend = 0.4)
    pg = make_pg_1d(20)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(sinpi(pg.core.positions[i][1])); end
    ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 1, 1, 2, NoLimiter(), RusanovFlux()), MOOD(), (), (src,), bt)
    dt = tend / n
    for k in 0:(n - 1); ts(eq, pg, k * dt, dt); end
    return [pg.rhos[i][1] for i in interior_indices(pg)]
end
eoc(bt, src) = (ref = imex_run(bt, src, 640); e1 = maximum(abs, imex_run(bt, src, 20) - ref); e2 = maximum(abs, imex_run(bt, src, 40) - ref); log2(e1 / e2))

sumright = IMEXButcherTableau([0.2 0.0; 1.0 0.2], [0.0 0.0; 1.0 0.0], [0.2, 1.2], [0.0, 1.0], [0.7, 0.3], [0.5, 0.5])
# implicit part: A = [γ 0; 1-2γ γ]... choose b·c̃ ≠ 1/2 but b̃·c = 1/2 (sum wrong)
onlyone = IMEXButcherTableau([0.5 0.0; 0.25 0.25], [0.0 0.0; 1.0 0.0], [0.5, 0.5], [0.0, 1.0], [0.3, 0.7], [0.5, 0.5])
for (name, bt) in (("b̃·c=0.7, b·c̃=0.3 (sum right)", sumright), ("b·c̃=0.7 only", onlyone), ("ARS222", IMEX_ARS222_Tableau(Float64)),
                   ("SSP2332", IMEX_SSP2332_Tableau(Float64)))
    println(rpad(name, 32), " b̃·c = $(round(dot(bt.b_t, bt.c), digits=3)), b·c̃ = $(round(dot(bt.b, bt.c_t), digits=3)), b·c = $(round(dot(bt.b, bt.c), digits=3)), b·1 = $(sum(bt.b)): ",
            "EOC κ=0: ", round(eoc(bt, DecaySource(3.0, 0.0)), digits = 2), ", κ=0.8: ", round(eoc(bt, DecaySource(3.0, 0.8)), digits = 2))
end
bt = IMEX_ARS233_Tableau(Float64); src = DecaySource(3.0, 0.8)
imex_run(bt, src, 20)
println("cost of one ARS233 test case (runs with n = 640, 20, 40): ", round(@elapsed(eoc(bt, src)), digits = 2), " s")
