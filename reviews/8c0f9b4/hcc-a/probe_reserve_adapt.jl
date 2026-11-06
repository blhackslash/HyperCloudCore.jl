# adapt_particles! on a reserved grid: "a CapacityError is thrown before the grid is changed"?
using Test, HyperCloudCore, StaticArrays
@testset "reserved grid, insertion outgrows the slot reservation" begin
    geom = get_rectangular_domain(Float64, (0.0,), (1.0,); is_periodic = true)
    dx = 0.02
    pg = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1)
    N = pg.meta.N
    for i in 1:N; pg.rhos[i] = State{1, Float64}(sin(2π * pg.core.positions[i][1])); end
    # one gap of 1.7 dx at the seam, the other 49 gaps slightly below dx (no merge candidates)
    order = sortperm([pg.core.positions[i][1] for i in 1:N])
    s = (1 - 1.7dx) / (N - 1)
    for (k, i) in enumerate(order); pg.core.positions[i] = Space{1, Float64}(0.001 + (k - 1) * s); end
    pg.neighbor(pg); update_volumes!(pg, ShepardVolumes())
    spp = slots_per_particle(pg)
    est = memory_estimate(pg; particles = 0, slots = 0)
    limit = est.fixed + ceil(Int, (N + 1.5) * (est.per_particle + spp * est.per_slot))
    r = reserve!(pg; limit = limit, tolerance = 0, slots_per_particle = spp)
    @show N r spp
    N0 = pg.meta.N; m0 = total_mass(pg)
    err = try
        adapt_particles!(pg, ParticleAdaptivity(Float64), 3); nothing
    catch e
        e
    end
    @show err
    println("N before ", N0, ", after ", pg.meta.N, "; mass change ", total_mass(pg) - m0)
    @test err isa CapacityError
    @test pg.meta.N == N0          # docstring: thrown before the grid is changed
    # the neighbour structure after the error
    nb = pg.neighbor
    @show length(nb.indices) maximum(last(nb.ranges[i]) for i in 1:pg.meta.N)
end
