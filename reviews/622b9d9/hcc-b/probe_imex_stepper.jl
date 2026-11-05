# Probe: observed temporal order of GeneralRKTimeStepper with the IMEX tableaus on u' = F(u, t) + S(u, t) at every
# particle of a constant-state periodic grid (zero flux divergence): F = -u² + sin t explicit, S = -λ(u - cos t)
# implicit with an exact solve.
include(joinpath(@__DIR__, "_prelude.jl"))

struct ExplF <: AbstractExplicitSourceTerm end
evaluate_source(::ExplF, U, i::Int, pg, t::Real) = typeof(U)(-U[1]^2 + sin(t))
struct ImplS <: AbstractImplicitSourceTerm; λ::Float64; end
evaluate_source(s::ImplS, U, i::Int, pg, t::Real) = typeof(U)(-s.λ * (U[1] - cos(t)))
implicit_solve(s::ImplS, U_in, γ::Real, i::Int, pg, t::Real) = typeof(U_in)((U_in[1] + γ * s.λ * cos(t)) / (1 + γ * s.λ))

function run_stepper(bt, dt; tend = 1.0)
    geom = get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true)
    pg = ParticleGrid(geom, (0.2,), ExponentialWeightFunction(1.0, 0.5), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0); end
    eq = LinearAdvection(((1.0,),), Float64)
    ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 1, 1, 2, NoLimiter(), RusanovFlux()), MOOD(), (ExplF(), ImplS(2.0)), bt)
    t = 0.0
    for _ in 1:round(Int, tend / dt); ts(eq, pg, t, dt); t += dt; end
    return pg.rhos[1][1]
end

@testset "IMEX stepper temporal order" begin
    for (make, p) in ((IMEX_Euler_Tableau, 1), (IMEX_ARS222_Tableau, 2), (IMEX_ARS233_Tableau, 3), (IMEX_PRSSP3_Tableau, 3), (IMEX_SSP2332_Tableau, 2))
        bt = make(Float64)
        ref = run_stepper(bt, 1 / 1280)
        e1, e2 = abs(run_stepper(bt, 1 / 40) - ref), abs(run_stepper(bt, 1 / 80) - ref)
        @printf("%-22s claimed %d  err(1/40) %.2e  err(1/80) %.2e  EOC %.2f\n", nameof(make), p, e1, e2, log2(e1 / e2))
        @test log2(e1 / e2) > p - 0.25
    end
end
