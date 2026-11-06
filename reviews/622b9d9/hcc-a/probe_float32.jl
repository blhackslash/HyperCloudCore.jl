# Probe: Float32 end to end through the builders (grid, weights, MLS volumes, MOOD U2, MUSCL, RK3, adaptivity,
# reserve!), on a 2-D moving grid. Checks element types stay Float32 and the adaptivity stays conservative.
using Test, HyperCloudCore, StaticArrays, Random
import HyperCloudCore: flux, max_eigenvalue, velocity, fluid_velocity
struct Adv2F <: HyperbolicPDE{2, 1, Float32} end
flux(::Adv2F, U::State{1, Float32}) = Flux{2, 1, Float32}((U, 0.5f0 * U))
max_eigenvalue(::Adv2F, U::State, d::Int) = d == 1 ? 1.0f0 : 0.5f0
velocity(::Adv2F, U::State, d::Int) = SMatrix{1, 1, Float32}(d == 1 ? 1.0f0 : 0.5f0)

@testset "probe_float32" begin
    T = Float32
    ctx = Dict{Symbol, Any}(:T => T, :D => 2, :M => 1, :Equation => Adv2F(), :ExplicitSources => (), :ImplicitSources => ())
    ctx[:Domain] = build_domain(Dict{Symbol, Any}(:domain => :rectangular, :mins => (0.0, 0.0), :maxs => (1.0, 1.0), :periodic => true), ctx)
    ctx[:WeightConf] = Dict{Symbol, Any}(:name => :exponential, :range => 2.5, :alpha => 1.0)
    pg = build_particle_grid(Dict{Symbol, Any}(:Ns => (24, 24), :randomness_factor => (0.2, 0.2), :mover => :sinusoidal,
                                               :mover_amplitude => 0.3, :volumes => :mls), ctx)
    @test eltype(pg.core.volumes) === T && eltype(pg.core.positions) === Space{2, T}
    for i in 1:pg.meta.N
        x = pg.core.positions[i]
        pg.rhos[i] = State{1, T}(1 + exp(-sum(abs2, x .- T(0.5)) / T(0.02)))
    end
    ctx[:Flux]    = build_flux(Dict{Symbol, Any}(:name => :Rusanov), ctx)
    ctx[:Limiter] = build_limiter(Dict{Symbol, Any}(:name => :VK, :mode => :hard), ctx)
    ctx[:MOOD]    = build_mood(Dict{Symbol, Any}(:criterion => :U2, :strategy => :EPD1, :delta_relax => 1e-4), ctx)
    ctx[:Scheme]  = build_scheme(Dict{Symbol, Any}(:name => :MUSCL, :order => 3), ctx)
    ctx[:Tableau] = build_tableau(Dict{Symbol, Any}(:stepper => :RK3), ctx)
    ts = build_timestepper(ctx)
    ad = build_adaptivity(Dict{Symbol, Any}(:enabled => true), ctx)
    @test ad isa ParticleAdaptivity{T}
    eq = ctx[:Equation]; t = 0.0f0; maxres = 0.0f0; nev = 0
    for k in 1:15
        dt = T(0.3) * get_time_step(pg, eq, ctx[:Scheme])
        @test dt isa T && isfinite(dt)
        ts(eq, pg, t, dt); t += dt
        r = adapt_particles!(pg, ad, 3)
        nev += r.inserted + r.removed
        maxres = max(maxres, maximum(abs, r.mass_residual))
    end
    println("Float32: N = $(pg.meta.N), events = $nev, max |mass residual| = $maxres, total mass = $(total_mass(pg))")
    @test all(i -> all(isfinite, pg.rhos[i]), 1:pg.meta.N)
    @test maxres < 1.0f-4
end
