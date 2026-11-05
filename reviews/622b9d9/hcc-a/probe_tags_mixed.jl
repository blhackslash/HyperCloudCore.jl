# Probe: 2-D channel periodic in x, walls in y. Ghosts below/above the domain must get tags 3/4 (bottom/top);
# the nearest-edge rule also measures the distance to the (non-existent) periodic left/right edges.
using Test, HyperCloudCore, StaticArrays
@testset "probe_tags_mixed" begin
    bc = Dict{Int, AbstractBoundaryCondition}(3 => OutflowBC(), 4 => OutflowBC())
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = (true, false), bc_map = bc)
    pg = ParticleGrid(geom, (0.05, 0.05), ExponentialWeightFunction(1.0, 0.125), 1)
    N = pg.meta.N
    ghosts = [i for i in 1:N if pg.core.is_boundary[i]]
    wrong = [(pg.core.positions[i], pg.core.tags[i]) for i in ghosts if !(pg.core.tags[i] in (3, 4))]
    println("ghosts: $(length(ghosts)), with a left/right tag in a channel periodic in x: $(length(wrong)); e.g. ", first(wrong, 4))
    @test isempty(wrong)
end
