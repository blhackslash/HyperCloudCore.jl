# Probe: results do not depend on the thread count. Run with -t 1 and -t 4; the second run compares with
# the file written by the first. 2-D box with OutflowBC/FixedDirichlet, MUSCL3 + VK + MOOD, jittered cloud,
# plus an adaptivity call.
using Test, HyperCloudCore, StaticArrays, Random, Serialization
import HyperCloudCore: flux, max_eigenvalue, velocity
struct Adv2 <: HyperbolicPDE{2, 1, Float64} end
flux(::Adv2, U::State{1, Float64}) = Flux{2, 1, Float64}((U, 0.5 * U))
max_eigenvalue(::Adv2, U::State, d::Int) = d == 1 ? 1.0 : 0.5
velocity(::Adv2, U::State, d::Int) = SMatrix{1, 1, Float64}(d == 1 ? 1.0 : 0.5)

function run_case()
    dx = 1 / 32
    bc = Dict{Int, AbstractBoundaryCondition}(1 => FixedDirichlet(), 2 => OutflowBC(), 3 => FixedDirichlet(), 4 => OutflowBC())
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); bc_map = bc)
    pg = ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(1.0, 2.5dx), 1; randomness = (0.25dx, 0.25dx), rng = MersenneTwister(5))
    for i in 1:pg.meta.N
        x = pg.core.positions[i]
        pg.rhos[i] = State{1, Float64}(abs(x[1] - 0.4) < 0.2 && abs(x[2] - 0.4) < 0.2 ? 1.0 : 0.0)
    end
    eq = Adv2()
    interp = MUSCL(Float64, 2, 1, 3, VenkatakrishnanLimiter(:hard), RusanovFlux())
    ts = GeneralRKTimeStepper(eq, interp, MOOD(EPD1(), MOODu1(1e-6)), RK3_SSP_Tableau(Float64))
    t = 0.0
    for k in 1:5
        dt = 0.3 * get_time_step(pg, eq, interp)
        ts(eq, pg, t, dt); t += dt
    end
    pg.core.positions[100] = pg.core.positions[100] + Space{2, Float64}(0.0, 0.0)
    return (copy(view(pg.rhos, 1:pg.meta.N)), copy(view(pg.core.particle_orders, 1:pg.meta.N)), t)
end

out = run_case()
f = joinpath(@__DIR__, "threads_ref.jls")
if Threads.nthreads() == 1
    serialize(f, out); println("reference written (1 thread), t = ", out[3], ", orders dropped: ", count(<(3), out[2]))
else
    ref = deserialize(f)
    @testset "thread count independence ($(Threads.nthreads()) threads vs 1)" begin
        @test out[1] == ref[1]
        @test out[2] == ref[2]
        @test out[3] == ref[3]
    end
end
