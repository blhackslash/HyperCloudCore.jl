include(joinpath(@__DIR__, "header.jl"))
struct DecaySource <: AbstractImplicitSourceTerm; λ::Float64; end
HyperCloudCore.evaluate_source(s::DecaySource, U, i::Int, pg, t::Real) = -s.λ * U
HyperCloudCore.implicit_solve(s::DecaySource, U_in, γ::Real, i::Int, pg, t::Real) = U_in / (1 + γ * s.λ)
eq = LinearAdvection(((1.0,),), Float64)
@testset "explicit tableau + implicit source + active MOOD (substeps incl. K_I)" begin
    res = map((MOOD(), MOOD(EPD1(), MOODu1(10.0)))) do mood
        pg = setic!(grid1d(40), x -> 1 + 0.5sin(π * x[1]))
        ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux()), mood, (), (DecaySource(3.0),), RK4_SSP_Tableau(Float64))
        for k in 0:9; ts(eq, pg, k * 0.01, 0.01); end
        [pg.rhos[i][1] for i in interior_indices(pg)], ts
    end
    @test res[1][1] ≈ res[2][1] rtol = 1e-14
    @test HyperCloudCore._checks_substeps(res[2][2])
    # A stiff decay drives the Euler substep negative (λ Δt/C > 1), the stage candidates stay positive: DMP flags
    pg = setic!(grid1d(40), x -> 1 + 0.5sin(π * x[1]))   # not constant: the DMP skips flat stencils
    ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux()), MOOD(EPD1(), MOODu1(1e-8)), (), (DecaySource(200.0),), RK4_SSP_Tableau(Float64))
    ts(eq, pg, 0.0, 0.01)
    println("λΔt/C = ", 200 * 0.01 / 1.508, "; flagged particles: ", count(mood_flags(ts)), " of ", pg.meta.N, "; orders: ", unique(view(pg.core.particle_orders, 1:pg.meta.N)))
    @test all(mood_flags(ts))
end
