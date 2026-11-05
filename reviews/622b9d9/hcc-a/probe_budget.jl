# Probe: every Growable reachable from the grid, the stepper and the scheme is registered in MemoryBudget.jl
# (CLAUDE.md rule), and reserve! + adaptivity / reorder interplay.
using Test, HyperCloudCore, StaticArrays, Random
import HyperCloudCore: flux, max_eigenvalue, velocity
struct Adv1 <: HyperbolicPDE{1, 1, Float64} end
flux(::Adv1, U::State{1, Float64}) = Flux{1, 1, Float64}((U,))
max_eigenvalue(::Adv1, U::State, d::Int) = 1.0
velocity(::Adv1, U::State, d::Int) = SMatrix{1, 1, Float64}(1.0)

function collect_growables!(acc, x, seen = IdDict{Any, Nothing}())
    (isbits(x) || x isa Module || x isa Function || x isa DataType) && return acc
    haskey(seen, x) && return acc
    seen[x] = nothing
    if x isa Growable
        push!(acc, x); return acc
    end
    if x isa AbstractArray
        eltype(x) <: Union{Number, SArray} || foreach(y -> collect_growables!(acc, y, seen), x)
        return acc
    end
    x isa AbstractDict && return acc
    for f in fieldnames(typeof(x))
        isdefined(x, f) && collect_growables!(acc, getfield(x, f), seen)
    end
    return acc
end

@testset "probe_budget" begin
    N  = 60
    dx = 2 / N
    geom = get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true)
    eq = Adv1()
    for (name, interp, mood) in (("MUSCL3 MOODu2", MUSCL(Float64, 1, 1, 3, VenkatakrishnanLimiter(:hard), RusanovFlux()), MOOD(EPD1(), MOODu2(1e-6))),
                                 ("WENO2", WENO(Float64, 1, 1, 2), MOOD()),
                                 ("Upwind1", UpwindDivergence(Float64, 1, 1, 1, :Classic, UpwindFlux()), MOOD()),
                                 ("Central2", CentralDivergence(Float64, 1, 1, 2), MOOD()))
        pg = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1)
        for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(exp(-pg.core.positions[i][1]^2 / 0.02)); end
        ts = GeneralRKTimeStepper(eq, interp, mood, RK3_SSP_Tableau(Float64))
        ts(eq, pg, 0.0, 0.4dx)
        all_g = collect_growables!(Any[], (pg, ts, interp))
        reg = IdDict{Any, Nothing}()
        for o in (pg, ts, interp)
            foreach(b -> (reg[b] = nothing), HyperCloudCore.particle_buffers(o, false))
            foreach(b -> (reg[b] = nothing), HyperCloudCore.slot_buffers(o))
        end
        reg[pg.neighbor.ranges] = nothing                    # handled on its own
        missing_g = [g for g in all_g if !haskey(reg, g)]
        @testset "$name" begin
            println(name, ": ", length(all_g), " growables, unregistered: ",
                    [(typeof(parent(g)), length(g)) for g in missing_g])
            @test isempty(missing_g)
        end
    end
end
