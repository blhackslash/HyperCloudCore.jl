using Test, HyperCloudCore, StaticArrays, Random
@testset "new tag test" begin
    @testset "Boundary tags: periodic edges ignored, bc_map mismatches warned" begin
        bc   = Dict{Int, AbstractBoundaryCondition}(3 => OutflowBC(), 4 => OutflowBC())
        geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = (true, false), bc_map = bc)
        pg   = @test_logs ParticleGrid(geom, (0.05, 0.05), ExponentialWeightFunction(1.0, 0.125), 1)   # no warning
        ghosts = [i for i in 1:pg.meta.N if pg.core.is_boundary[i]]
        @test !isempty(ghosts)
        @test all(i -> pg.core.tags[i] == (pg.core.positions[i][2] < 0 ? 3 : 4), ghosts)   # 29 of 120 were 1 or 2
        # 1-D: the left wall carries tag 1, the right wall 2 (both carried 1 up to 0.7.9, so a tag-2 entry, e.g. an
        # outflow at the right end of a Sod tube, was dead); a map with only tag 1 leaves the right wall uncovered
        bc1 = Dict{Int, AbstractBoundaryCondition}(1 => FixedDirichlet(), 2 => OutflowBC())
        geom1 = get_rectangular_domain(Float64, (0.0,), (1.0,); bc_map = bc1)
        pg1 = @test_logs ParticleGrid(geom1, (0.1,), ExponentialWeightFunction(1.0, 0.25), 1)
        @test all(i -> !pg1.core.is_boundary[i] || pg1.core.tags[i] == (pg1.core.positions[i][1] < 0.5 ? 1 : 2), 1:pg1.meta.N)
        geom1b = get_rectangular_domain(Float64, (0.0,), (1.0,); bc_map = Dict{Int, AbstractBoundaryCondition}(1 => OutflowBC()))
        @test_logs (:warn, r"tag\(s\) \[2\] have no condition") ParticleGrid(geom1b, (0.1,), ExponentialWeightFunction(1.0, 0.25), 1)
        @test_logs ParticleGrid(geom1b, (0.1,), ExponentialWeightFunction(1.0, 0.25), 1)       # once per session
        # 3-D: faces 1/2 (x), 3/4 (y), 5/6 (z), the nearest face, the lower tag at edges and corners; periodic z: no 5, 6
        for per in (false, (false, false, true))
            g3  = get_rectangular_domain(Float64, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0); is_periodic = per)
            pg3 = ParticleGrid(g3, (0.125, 0.125, 0.125), ExponentialWeightFunction(1.0, 0.3), 1)
            function face(x)
                best, bd = 1, Inf
                for d in 1:3
                    per isa Tuple && per[d] && continue
                    abs(x[d]) < bd && (best = 2d - 1; bd = abs(x[d]))
                    abs(x[d] - 1) < bd && (best = 2d; bd = abs(x[d] - 1))
                end
                return best
            end
            gh = [i for i in 1:pg3.meta.N if pg3.core.is_boundary[i]]
            @test all(i -> pg3.core.tags[i] == face(pg3.core.positions[i]), gh)
            @test sort(unique(pg3.core.tags[gh])) == (per === false ? collect(1:6) : collect(1:4))
        end
        # 2-D box with only the left/right walls mapped: the bottom/top particles keep their state
        geom2 = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0);
                                       bc_map = Dict{Int, AbstractBoundaryCondition}(1 => OutflowBC(), 2 => OutflowBC()))
        @test_logs (:warn, r"tag\(s\) \[3, 4\] have no condition") ParticleGrid(geom2, (0.1, 0.1), ExponentialWeightFunction(1.0, 0.25), 1)
    end
end
