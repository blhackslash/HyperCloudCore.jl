# Probe: neighbour search against brute force in 1-3 D, periodic / mixed / non-periodic, Float32 and Float64,
# jittered clouds; 3-D Morton reordering is a permutation; slot-major conversion keeps the lists.
using Test, HyperCloudCore, StaticArrays, Random

function brute_neighbours(pg)
    N = pg.meta.N; pos = pg.core.positions; R2 = pg.meta.R^2
    L, Li = pg.domain.L_wrap, pg.domain.invL_wrap
    [sort([j for j in 1:N if j != i && sum(abs2, HyperCloudCore.get_distance(pos, i, j, L, Li)) <= R2]) for i in 1:N]
end

@testset "probe_nsearch" begin
    for T in (Float64, Float32), D in (1, 2, 3), per in (true, false, :mixed)
        D == 1 && per === :mixed && continue
        n   = D == 3 ? 8 : 16
        dx  = T(1) / n
        isper = per === :mixed ? ntuple(d -> d == 1, D) : per
        geom = get_rectangular_domain(T, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D); is_periodic = isper)
        pg  = ParticleGrid(geom, ntuple(_ -> dx, D), ExponentialWeightFunction(one(T), T(2.2) * dx), 1;
                           randomness = ntuple(_ -> T(0.3) * dx, D), rng = MersenneTwister(7))
        N   = pg.meta.N
        nb  = pg.neighbor
        got = [sort(collect(view(nb.indices, nb.ranges[i]))) for i in 1:N]
        @testset "T=$T D=$D periodic=$per" begin
            @test got == brute_neighbours(pg)
            @test pg.meta.max_nb == maximum(length, got)
            # distances and weights consistent with positions
            @test all(i -> all(k -> nb.distances[k] ≈ HyperCloudCore.get_distance(pg.core.positions, i, nb.indices[k], pg.domain.L_wrap, pg.domain.invL_wrap), nb.ranges[i]), 1:N)
            @test all(i -> all(k -> nb.weights[k] ≈ nb.weight_func(sum(abs2, nb.distances[k])), nb.ranges[i]), 1:N)
            @test eltype(pg.core.volumes) === T && sum(view(pg.core.volumes, 1:N)) ≈ domain_volume(pg)
            # slot-major conversion keeps the lists in order
            sm = HyperCloudCore.with_slot_layout(nb, N, HyperCloudCore.SlotMajorSlots)
            @test all(i -> [sm.indices[k] for k in sm.ranges[i]] == [nb.indices[k] for k in nb.ranges[i]], 1:N)
        end
    end
    @testset "3-D reorder is a permutation of the cloud" begin
        geom = get_rectangular_domain(Float64, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0); is_periodic = true)
        pg = ParticleGrid(geom, (0.125, 0.125, 0.125), ExponentialWeightFunction(1.0, 0.3), 1)
        N = pg.meta.N
        before = sort(collect(view(pg.core.positions, 1:N)))
        shuffle!(MersenneTwister(3), view(pg.core.positions, 1:N))
        pg.reorder(pg)
        @test sort(collect(view(pg.core.positions, 1:N))) == before
    end
end
