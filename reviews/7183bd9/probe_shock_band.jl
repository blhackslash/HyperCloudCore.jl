# 622b9d9 area A #9: shock-band radius in 2-D. A reduced-order particle A; a merge event whose starting particle lies
# between 4R and the new band radius from A. Old code (4R band): the event is accepted and, if A lies in its local set,
# A's state changes. New code: no event started within the band.
include(joinpath(@__DIR__, "header.jl"))
function run_case(dist_R; seed = 3, kmax = 3)
    n = 40; dx = 2 / n
    geom = get_rectangular_domain(Float64, (-1.0, -1.0), (1.0, 1.0); is_periodic = true)
    pg = ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(1.0, 2.5dx), 1; randomness = (0.05dx, 0.05dx), rng = MersenneTwister(seed))
    R, N = pg.meta.R, pg.meta.N
    L, Li = pg.domain.L_wrap, pg.domain.invL_wrap
    d(i, x) = sqrt(sum(abs2, HC._wrapped_diff(pg.core.positions[i], x, L, Li)))
    A = argmin(i -> d(i, Space{2, Float64}(0.0, 0.0)), 1:N)
    xA = pg.core.positions[A]
    target = xA + Space{2, Float64}(dist_R * R, 0.0)
    q = argmin(i -> d(i, target), 1:N)                                 # partner, at ≈ dist_R·R from A
    p = argmin(i -> i == q ? Inf : (pg.core.positions[i][1] < pg.core.positions[q][1] - 0.5dx ? Inf : d(i, pg.core.positions[q])), 1:N)
    pg.core.positions[p] = HC._wrap_point(pg, pg.core.positions[q] + Space{2, Float64}(0.0, 0.2dx))   # close pair: a merge
    pg.neighbor(pg)
    for i in 1:N
        x = pg.core.positions[i]; pg.rhos[i] = State{1, Float64}(1 + 0.5sinpi(x[1]) * cospi(x[2]) + 0.3x[1])
    end
    update_volumes!(pg, ShepardVolumes())
    fill!(view(pg.core.particle_orders, 1:N), kmax); pg.core.particle_orders[A] = 1
    UA, VA = pg.rhos[A], pg.core.volumes[A]
    ad = ParticleAdaptivity(Float64)
    evs = HC._detect_events(pg, ad, kmax)
    starts = [isempty(ev.removed) ? NaN : minimum(r -> d(r, xA), ev.removed) / R for ev in evs]
    # A's index after compaction: by position
    rep = adapt_particles!(pg, ad, kmax)
    A2 = findfirst(i -> pg.core.positions[i] == xA, 1:pg.meta.N)
    return (events = length(evs), start_dist_R = starts, centre_dist_R = [d(A, ev.center) / R for ev in evs],
            dU = Float64(pg.rhos[A2][1] - UA[1]), dV = pg.core.volumes[A2] - VA, band_R = max(4R, R * (1.5 + 2sqrt(2)) + dx) / R)
end
for dist in (3.8, 4.05, 4.15, 4.3, 4.5, 4.8, 5.2)
    r = run_case(dist)
    @printf("partner at %.2f R: events=%d removed particles at %s R, centre at %s R (band %.2f R): ΔU_A=%+.2e ΔV_A=%+.2e\n",
            dist, r.events, string(round.(r.start_dist_R; digits = 2)), string(round.(r.centre_dist_R; digits = 2)), r.band_R, r.dU, r.dV)
end
