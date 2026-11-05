# Probe: adaptivity inserts between an interior particle and a ghost; the new particle may sit outside the
# physical domain but is flagged interior (src/Grid/Adaptivity.jl `_detect_events`, insertion branch).
using Test, HyperCloudCore, StaticArrays

@testset "probe_adapt" begin
    dx   = 0.01
    geom = get_rectangular_domain(Float64, (0.0,), (1.0,))
    pg   = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1)
    N    = pg.meta.N
    for i in 1:N
        pg.rhos[i] = State{1, Float64}(1.0)
        pg.core.is_boundary[i] || (pg.core.positions[i] = 0.992 * pg.core.positions[i])   # interior compresses, ghosts stay
    end
    pg.neighbor(pg)
    update_volumes!(pg, ShepardVolumes())
    rep = adapt_particles!(pg, ParticleAdaptivity(Float64), 3)
    println("adapt report: ", rep)
    N2 = pg.meta.N
    outside = [pg.core.positions[i][1] for i in 1:N2 if !pg.core.is_boundary[i] && !geom.is_interior(pg.core.positions[i])]
    println("interior-flagged particles outside [0, 1]: ", outside)
    @test isempty(outside)
end
