# Probe: an implicit source with an explicit tableau. The constructor warns "the implicit solve is skipped
# (a[i,i] = 0)", which suggests the source is still applied (explicitly). With b = 0 the K_I stages never enter
# the update, so the source is dropped altogether. Also: math_min/math_max round-off.
include(joinpath(@__DIR__, "_prelude.jl"))

struct ImplDecay <: AbstractImplicitSourceTerm; λ::Float64; end
HyperCloudCore.evaluate_source(s::ImplDecay, U, i::Int, pg, t::Real) = -s.λ * U
HyperCloudCore.implicit_solve(s::ImplDecay, U_in, γ::Real, i::Int, pg, t::Real) = U_in / (1 + γ * s.λ)

@testset "implicit source with an explicit tableau" begin
    pg = make_pg_1d(20); for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0); end
    eq = LinearAdvection(((1.0,),), Float64)
    ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 1, 1, 2, NoLimiter(), RusanovFlux()), MOOD(), (ImplDecay(3.0),), RK3_SSP_Tableau(Float64))
    ts(eq, pg, 0.0, 0.1)
    @printf("u after one step with S = -3u, dt = 0.1, RK3 tableau: %.15f (exact decay %.6f)\n", pg.rhos[1][1], exp(-0.3))
    @test pg.rhos[1][1] != 1.0       # fails: the source is ignored
end

@testset "math_min / math_max round-off" begin
    for T in (Float32, Float64)
        a, b = T(1e3), T(1e-3)
        @printf("%s: math_min(1e3, 1e-3) = %.8e (rel. error %.1e); math_min(1e20, 1) = %.3e\n", T, math_min(a, b),
                abs(math_min(a, b) - b) / b, math_min(T(1e20), one(T)))
        @test math_min(a, b) == b
    end
end
