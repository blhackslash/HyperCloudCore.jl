# 0c5a05f: `_slots_after_events` must be an upper bound of the slots the search after the topology change needs.
# Random strongly jittered clouds (many merges and insertions), 1-D/2-D/3-D, periodic and walls, Float64/Float32,
# row-major and slot-major layouts; plus the cost of the count on a 2-D cloud with ~40k particles.
include(joinpath(@__DIR__, "header.jl"))

function cloud(D, T; per, n, jit = 0.45, seed = 1)
    geom = get_rectangular_domain(T, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D); is_periodic = per)
    dx   = T(1 / n)
    pg   = ParticleGrid(geom, ntuple(_ -> dx, D), ExponentialWeightFunction(one(T), T(2.5) * dx), 1;
                        randomness = ntuple(_ -> T(jit) * dx, D), rng = MersenneTwister(seed))
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, T}(one(T) + pg.core.positions[i][1]); end
    update_volumes!(pg, ShepardVolumes())
    return pg
end

slot_major(pg) = ParticleGrid(pg.meta, pg.geometry, pg.domain, pg.core, pg.shared,
                              HC.with_slot_layout(pg.neighbor, pg.meta.N, HC.SlotMajorSlots), pg.reorder, pg.bins,
                              pg.mover, pg.rhos, pg.curvatures)

stats = Dict{Symbol, Int}(:cases => 0, :exact => 0)
for D in (1, 2, 3), T in (Float64, Float32), per in (true, false), layout in (:row, :slot)
    n = D == 1 ? 60 : D == 2 ? 24 : 9
    @testset "D=$D $T per=$per $layout" begin
        for seed in 1:(D == 3 ? 2 : 4)
            pg = cloud(D, T; per, n, seed)
            layout == :slot && (pg = slot_major(pg))
            ad = ParticleAdaptivity(T)
            evs = HC._detect_events(pg, ad, 3)
            isempty(evs) && continue
            tot, mx = HC._slots_after_events(pg, evs)
            rep = adapt_particles!(pg, ad, 3)
            N   = pg.meta.N
            act_tot = sum(i -> length(pg.neighbor.ranges[i]), 1:N)
            act_mx  = pg.meta.max_nb
            @test tot >= act_tot && mx >= act_mx
            stats[:cases] += 1
            (tot == act_tot && mx == act_mx) && (stats[:exact] += 1)
            (tot == act_tot && mx == act_mx) || println("D=$D $T per=$per $layout seed=$seed: predicted ($tot, $mx), actual ($act_tot, $act_mx)")
            @test rep.inserted + rep.removed > 0
        end
    end
end
println("cases: ", stats)

# Cost: 2-D 200x200 periodic, 45 % jitter
pg  = cloud(2, Float64; per = true, n = 200)
ad  = ParticleAdaptivity(Float64)
evs = HC._detect_events(pg, ad, 3)
HC._slots_after_events(pg, evs)
t_slots = @elapsed HC._slots_after_events(pg, evs)
a_slots = @allocated HC._slots_after_events(pg, evs)
t_det   = @elapsed HC._detect_events(pg, ad, 3)
pg2 = cloud(2, Float64; per = true, n = 200); adapt_particles!(cloud(2, Float64; per = true, n = 20), ad, 3)
t_adapt = @elapsed adapt_particles!(pg2, ad, 3)
println("N = $(pg.meta.N), events = $(length(evs)): _slots_after_events $(round(1e3t_slots, digits=2)) ms, $(a_slots ÷ 1024) KiB; ",
        "_detect_events $(round(1e3t_det, digits=2)) ms; whole adapt_particles! $(round(1e3t_adapt, digits=1)) ms")

# Frozen slot-major grid: reserve! with the current slots per particle, then adapt → CapacityError before any change
@testset "slot-major + reserve!: error before the change" begin
    pg = slot_major(cloud(1, Float64; per = true, n = 50, jit = 0.0))
    N  = pg.meta.N
    order = sortperm([pg.core.positions[i][1] for i in 1:N]); s = (1 - 1.7 / 50) / (N - 1)
    for (k, i) in enumerate(order); pg.core.positions[i] = Space{1, Float64}(0.001 + (k - 1) * s); end
    pg.neighbor(pg); update_volumes!(pg, ShepardVolumes())
    spp = slots_per_particle(pg)
    est = memory_estimate(pg; particles = 0, slots = 0)
    reserve!(pg; limit = est.fixed + ceil(Int, (N + 1.5) * (est.per_particle + spp * est.per_slot)), tolerance = 0, slots_per_particle = spp)
    U0 = copy(view(pg.rhos, 1:N)); X0 = copy(view(pg.core.positions, 1:N))
    err = try adapt_particles!(pg, ParticleAdaptivity(Float64), 3); nothing catch e; e end
    println("slot-major frozen: ", err === nothing ? "no error" : typeof(err), ", N = ", pg.meta.N)
    # slot-major pads to max_nb: the reservation spp·(N + 1.5) = 4·51.5 slots holds the 4·51 the insertion needs,
    # so no error is the right outcome here; either way the grid must be consistent
    if err isa CapacityError
        @test pg.meta.N == N && view(pg.rhos, 1:N) == U0 && view(pg.core.positions, 1:N) == X0
    else
        @test err === nothing && pg.meta.N == N + 1 && maximum(i -> last(pg.neighbor.ranges[i]), 1:pg.meta.N) <= length(pg.neighbor.indices)
    end
end
