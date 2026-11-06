# The new tests of 1d80418/7183bd9, run against either code base (each in its own @testset, errors counted)
using Test, HyperCloudCore, StaticArrays, Random, Logging
const HC = HyperCloudCore
kmax = 3
ts = @testset "new tests" begin
    @testset "param2uvec / mixed constructors" begin
        @test (try HC.param2uvec((1, 2.0)) isa SVector{2, Float64} catch; false end)
        @test (try ExponentialWeightFunction(1.0, 0.25f0) isa ExponentialWeightFunction{Float64} catch; false end)
        @test (try InverseWeightFunction(1, 0.5) isa InverseWeightFunction{Float64} catch; false end)
    end
    @testset "wall by index (Float32 offset)" begin
        for (lo, hi, Ns) in ((100, 101, 10_000), (1000, 1001, 1000))
            dx   = (Float32(hi) - Float32(lo)) / Ns
            geom = get_rectangular_domain(Float32, (lo,), (hi,))
            cd   = HC.ComputationalDomain(geom, (dx,), 2.5f0)
            pos, isb, _, _, _ = HC.get_points(cd, geom; nominal_dx = (dx,), interp_range_factor = 2.5f0)
            xs   = sort([p[1] for (p, b) in zip(pos, isb) if !b])
            @test length(xs) == Ns + 1 && xs[1] == lo && xs[end] == hi
            @test minimum(diff(xs)) > dx / 2
        end
    end
    @testset "non-divisible extent" begin
        geom = get_rectangular_domain(Float64, (0.0,), (1.0,))
        cd   = HC.ComputationalDomain(geom, (0.3,), 2.5)
        pos, isb, _, _, _ = HC.get_points(cd, geom; nominal_dx = (0.3,), interp_range_factor = 2.5)
        xs   = sort([p[1] for (p, b) in zip(pos, isb) if !b])
        @test xs[1] == 0 && maximum(abs.(xs ./ 0.3 .- round.(xs ./ 0.3))) < 1e-12 && xs[end] < 1
    end
    @testset "starved warning" begin
        geom = get_rectangular_domain(Float64, (0.0,), (1.0,); is_periodic = true)
        @test_logs (:warn, r"fewer than 2 neighbours") match_mode = :any ParticleGrid(geom, (0.025,),
            ExponentialWeightFunction(1.0, 1.2 * 0.025), 1; randomness = (0.2 * 0.025,), rng = MersenneTwister(3))
    end
    @testset "Float32 ghost layers" begin
        dxm  = 1.0f0 / 197
        dx32 = (dxm, dxm / 6)
        cd   = HC.ComputationalDomain(get_rectangular_domain(Float32, (0.0, 0.0), (1.0, 1.0)), dx32, 3.5f0)
        @test round(Int, -cd.canvas_mins[2] / dx32[2]) == ceil(Int, 3.5 * Float64(dxm) / Float64(dx32[2]) - 1e-6)
    end
    @testset "walls + volume estimator" begin
        for (D, n) in ((1, 20), (2, 16)), est in (nothing, ShepardVolumes())
            bcD   = Dict{Int, AbstractBoundaryCondition}(t => FixedDirichlet() for t in (D == 1 ? (1,) : (1, 2, 3, 4)))
            geomD = get_rectangular_domain(Float64, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D); bc_map = bcD)
            h     = 1 / n
            pgD   = ParticleGrid(geomD, ntuple(_ -> h, D), ExponentialWeightFunction(1.0, 2.5h), 1;
                                 randomness = ntuple(_ -> 0.45h, D), rng = MersenneTwister(1))
            for i in 1:pgD.meta.N
                x = pgD.core.positions[i]
                pgD.rhos[i] = State{1, Float64}(1 + sum(x) + 0.3sinpi(3x[1]))
            end
            update_volumes!(pgD, ShepardVolumes())
            gD(i) = pgD.core.is_boundary[i]
            inner() = sum(i -> pgD.core.volumes[i] * pgD.rhos[i][1], filter(!gD, 1:pgD.meta.N))
            scale   = sum(i -> pgD.core.volumes[i] * abs(pgD.rhos[i][1]), 1:pgD.meta.N)
            c0  = inner()
            rep = adapt_particles!(pgD, ParticleAdaptivity(Float64; volume_estimator = est), kmax)
            @test abs(inner() - c0) < 1e-13 * scale
            @test abs(rep.mass_residual[1]) < 1e-13 * scale
        end
    end
end
