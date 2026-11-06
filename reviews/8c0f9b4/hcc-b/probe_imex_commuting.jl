# The stepper-level IMEX order test (unit_schemes.jl "IMEX tableaus: observed order through the stepper") uses
# commuting linear operators, so only the SUM of the cross coupling conditions enters. A pair that violates
# each coupling condition (b̃·c = 0.7, b·c̃ = 0.3) but has the right sum passes it at "order 2", while it is
# first order on a non-commuting (non-autonomous) problem.
include(joinpath(@__DIR__, "header.jl"))
struct DecaySource <: AbstractImplicitSourceTerm; λ::Float64; end
HyperCloudCore.evaluate_source(s::DecaySource, U, i::Int, pg, t::Real) = -s.λ * U
HyperCloudCore.implicit_solve(s::DecaySource, U_in, γ::Real, i::Int, pg, t::Real) = U_in / (1 + γ * s.λ)
bad = IMEXButcherTableau([0.2 0.0; 1.0 0.2], [0.0 0.0; 1.0 0.0], [0.2, 1.2], [0.0, 1.0], [0.7, 0.3], [0.5, 0.5])
println("b̃·c = ", dot(bad.b_t, bad.c), "  b·c̃ = ", dot(bad.b, bad.c_t), "  b·c = ", dot(bad.b, bad.c), "  b̃·c̃ = ", dot(bad.b_t, bad.c_t))
eq  = LinearAdvection(((1.0,),), Float64)
src = DecaySource(3.0)
function imex_run(bt, n; tend = 0.4)          # as in the test
    pg = setic!(grid1d(20), x -> sin(π * x[1]))
    ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 1, 1, 2, NoLimiter(), RusanovFlux()), MOOD(), (), (src,), bt)
    dt = tend / n
    for k in 0:(n - 1); ts(eq, pg, k * dt, dt); end
    return [pg.rhos[i][1] for i in interior_indices(pg)]
end
for (name, bt) in (("constructed pair", bad), ("ARS222", IMEX_ARS222_Tableau(Float64)))
    ref = imex_run(bt, 640)
    e1, e2 = maximum(abs, imex_run(bt, 20) - ref), maximum(abs, imex_run(bt, 40) - ref)
    println(rpad(name, 18), " stepper test EOC = ", round(log2(e1 / e2), digits = 2), "  (test asserts > 1.8)")
end
# non-autonomous scalar ODE u' = -u cos t (explicit) - λ (u - sin t) (implicit)
function imex_ode(bt, n; lam = 2.0, tend = 1.0)
    A, At = Matrix(bt.a), Matrix(bt.a_t); b, bt_ = Vector(bt.b), Vector(bt.b_t); ct = Vector(bt.c_t)
    s = length(b); dt = tend / n; u = 1.0; t = 0.0
    for _ in 1:n
        KE = zeros(s); KI = zeros(s)
        for i in 1:s
            Y = u + dt * sum(At[i, j] * KE[j] + A[i, j] * KI[j] for j in 1:i-1; init = 0.0)
            ti = t + ct[i] * dt; γ = dt * A[i, i]
            Y = (Y + γ * lam * sin(ti)) / (1 + γ * lam)
            KI[i] = -lam * (Y - sin(ti)); KE[i] = -Y * cos(ti)
        end
        u += dt * sum(bt_[j] * KE[j] + b[j] * KI[j] for j in 1:s); t += dt
    end
    u
end
ref = imex_ode(bad, 40960)
println("constructed pair, non-autonomous ODE EOC = ", round(log2(abs(imex_ode(bad, 40) - ref) / abs(imex_ode(bad, 80) - ref)), digits = 2))
