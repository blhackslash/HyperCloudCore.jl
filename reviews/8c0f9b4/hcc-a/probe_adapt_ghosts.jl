# Does an adaptivity event near a wall change the states of FixedDirichlet ghost particles?
using Test, HyperCloudCore, StaticArrays
@testset "adaptivity leaves Dirichlet ghost states alone" begin
    bc   = Dict{Int, AbstractBoundaryCondition}(1 => FixedDirichlet())
    geom = get_rectangular_domain(Float64, (0.0,), (1.0,); bc_map = bc)
    dx   = 0.05
    pg   = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1)
    N    = pg.meta.N
    for i in 1:N; pg.rhos[i] = State{1, Float64}(1 + pg.core.positions[i][1]); end
    # move the interior particle at x = 0.05 next to the wall particle at x = 0 (merge candidate)
    i5 = argmin(i -> abs(pg.core.positions[i][1] - 0.05), 1:N)
    pg.core.positions[i5] = Space{1, Float64}(0.015)
    pg.rhos[i5] = State{1, Float64}(1.015)
    pg.neighbor(pg); update_volumes!(pg, ShepardVolumes())
    ghosts = [i for i in 1:N if pg.core.is_boundary[i]]
    gpos   = [pg.core.positions[i] for i in ghosts]
    gU0    = Dict(pg.core.positions[i] => pg.rhos[i] for i in ghosts)
    m0     = total_mass(pg)
    rep = adapt_particles!(pg, ParticleAdaptivity(Float64), 3)
    @show rep
    gU1 = Dict(pg.core.positions[i] => pg.rhos[i] for i in 1:pg.meta.N if pg.core.is_boundary[i])
    changed = [(p[1], gU0[p][1], gU1[p][1]) for p in keys(gU0) if gU1[p] != gU0[p]]
    println("ghost states changed by the remap (x, before, after): ", sort(changed))
    @test rep.removed == 2 && rep.inserted == 1
    @test isempty(changed)
end
