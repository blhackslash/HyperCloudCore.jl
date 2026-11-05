# d92911e: adaptivity remap with boundary particles held fixed.
# 1-D / 2-D boxes with walls (FixedDirichlet / OutflowBC), Float64 / Float32, strongly jittered clouds with a smooth
# non-constant state, so that many events fall next to a wall. Checks:
#  - ghost states and volumes untouched,
#  - Σ V U over the interior (and over all particles) exact to round-off,
#  - every written interior particle within the old local bounds (min/max of old states within R of its position)
#  - with a volume_estimator: what is exact afterwards (interior or all).
include(joinpath(@__DIR__, "header.jl"))

function wall_grid(D, T; jit = 0.45, seed = 1, n = 20, bc = :dirichlet)
    tags = D == 2 ? (1, 2, 3, 4) : (1,)
    bcm  = Dict{Int, AbstractBoundaryCondition}(t => (bc == :dirichlet ? FixedDirichlet() : OutflowBC()) for t in tags)
    geom = get_rectangular_domain(T, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D); bc_map = bcm)
    dx   = T(1 / n)
    pg   = ParticleGrid(geom, ntuple(_ -> dx, D), ExponentialWeightFunction(one(T), T(2.5) * dx), 1;
                        randomness = ntuple(_ -> T(jit) * dx, D), rng = MersenneTwister(seed))
    for i in 1:pg.meta.N
        x = pg.core.positions[i]
        pg.rhos[i] = State{1, T}(T(1) + sum(x) + T(0.3) * sinpi(T(3) * x[1]))
    end
    update_volumes!(pg, ShepardVolumes())
    return pg
end

near_wall(pg, c) = any(i -> pg.core.is_boundary[i] &&
                            sqrt(sum(abs2, pg.core.positions[i] - c)) <= pg.meta.R * (1 + 2sqrt(length(c))) + minimum(pg.meta.dx),
                       1:pg.meta.N)

function check(D, T; seed = 1, bc = :dirichlet, est = nothing, jit = 0.45)
    pg  = wall_grid(D, T; seed, bc, jit)
    N   = pg.meta.N
    ad  = ParticleAdaptivity(T; volume_estimator = est)
    evs = HC._detect_events(pg, ad, 3)
    nwall = count(ev -> near_wall(pg, ev.center), evs)
    isb = copy(view(pg.core.is_boundary, 1:N))
    g0  = Dict(pg.core.positions[i] => (pg.rhos[i], pg.core.volumes[i]) for i in 1:N if isb[i])
    oldpos = copy(view(pg.core.positions, 1:N)); oldU = copy(view(pg.rhos, 1:N))
    content(g) = sum(i -> Float64(pg.core.volumes[i]) * Float64(pg.rhos[i][1]), filter(i -> pg.core.is_boundary[i] == g, 1:pg.meta.N))
    scale = sum(i -> Float64(pg.core.volumes[i]) * abs(Float64(pg.rhos[i][1])), 1:N)
    c_in, c_gh = content(false), content(true)
    rep = adapt_particles!(pg, ad, 3)
    g1  = Dict(pg.core.positions[i] => (pg.rhos[i], pg.core.volumes[i]) for i in 1:pg.meta.N if pg.core.is_boundary[i])
    R   = pg.meta.R
    # local bounds: old states within R of the particle's (new) position
    viol = 0.0
    for i in 1:pg.meta.N
        pg.core.is_boundary[i] && continue
        x = pg.core.positions[i]
        ks = [k for k in 1:N if sum(abs2, oldpos[k] - x) <= R^2]
        lo, hi = minimum(k -> oldU[k][1], ks), maximum(k -> oldU[k][1], ks)
        viol = max(viol, Float64(lo - pg.rhos[i][1]), Float64(pg.rhos[i][1] - hi))
    end
    d_in = (content(false) - c_in) / scale
    d_gh = (content(true) - c_gh) / scale
    return (; events = length(evs), nwall, ins = rep.inserted, rem = rep.removed, ghosts_same = g0 == g1,
            ghost_states_same = Dict(k => v[1] for (k, v) in g0) == Dict(k => v[1] for (k, v) in g1),
            d_in, d_gh, mres = Float64(rep.mass_residual[1]) / scale, viol)
end

for D in (1, 2), T in (Float64, Float32), bc in (:dirichlet, :outflow)
    @testset "walls D=$D $T $bc" begin
        tot_wall = 0
        for seed in 1:(D == 1 ? 6 : 3)
            r = check(D, T; seed, bc)
            tot_wall += r.nwall
            println("D=$D $T $bc seed=$seed: ", r)
            @test r.ghosts_same
            @test abs(r.d_in) < 50eps(T) && r.d_gh == 0
            @test r.viol <= 10eps(T)
        end
        @test tot_wall > 0
    end
end

# with the stepper's volume estimator (moving grids): which sum is exact afterwards?
for D in (1, 2), T in (Float64,)
    for seed in 1:3
        r = check(D, T; seed, est = ShepardVolumes())
        println("volume_estimator D=$D $T seed=$seed: d_in=$(r.d_in) d_gh=$(r.d_gh) mass_residual=$(r.mres) ghosts_same=$(r.ghosts_same) ghost_states_same=$(r.ghost_states_same) nwall=$(r.nwall)")
    end
end
