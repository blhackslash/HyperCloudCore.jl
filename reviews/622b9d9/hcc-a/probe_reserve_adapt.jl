# Probe: adapt_particles! on a host grid whose buffers were frozen by reserve!: the CapacityError is thrown
# after the remap has already rewritten states and volumes in place, so the grid is left changed.
using Test, HyperCloudCore, StaticArrays

@testset "probe_reserve_adapt" begin
    N0   = 40
    dx   = 2 / N0
    geom = get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true)
    pg   = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0 + 0.5 * sin(π * pg.core.positions[i][1])); end
    # open a hole by moving particle 10 onto particle 9's neighbour gap edge: remove it through the remap
    ad = ParticleAdaptivity(Float64)
    ev = HyperCloudCore._AdaptEvent{1, Float64}(pg.core.positions[10], [10], Space{1, Float64}[])
    np, nr, nv = Space{1, Float64}[], State{1, Float64}[], Float64[]
    HyperCloudCore._remap_event!(pg, ad, ev, np, nr, nv)
    mask = pg.shared.bit_buffer; fill!(view(mask, 1:pg.meta.N), false); mask[10] = true
    HyperCloudCore._apply_topology!(pg, mask, np, nr, nv, 3)
    pg.neighbor(pg)
    N = pg.meta.N
    est = memory_estimate(pg)
    reserve!(pg; limit = est.bytes + 1, tolerance = 0)            # exactly N particles
    rho0 = copy(view(pg.rhos, 1:N)); vol0 = copy(view(pg.core.volumes, 1:N)); m0 = total_mass(pg)
    err = try
        adapt_particles!(pg, ad, 3); nothing
    catch e
        e
    end
    println("error: ", typeof(err))
    @test err isa CapacityError
    changed = count(i -> pg.rhos[i] != rho0[i] || pg.core.volumes[i] != vol0[i], 1:N)
    println("particles whose state/volume changed although the call failed: $changed; mass change ",
            total_mass(pg) - m0, "; N = $(pg.meta.N)")
    @test changed == 0
    @test total_mass(pg) ≈ m0
end
