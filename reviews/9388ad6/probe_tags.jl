# Probe for claude/boundary-tags: tag_func vs the parent's tag_func (2-D bitwise), 1-D/3-D, invariants.
# julia -t 2 --project=<worktree>/test /tmp/review_pr4/probes/probe_tags.jl
using HyperCloudCore, StaticArrays, Random, Logging
const HC = HyperCloudCore

# Parent (claude/review-grid-adaptivity) tag function, copied verbatim (D == 2 branch, else 1)
function old_tag(pos, mins_f, maxs_f, is_per_svec, ::Type{T}) where {T}
    D = length(pos)
    inside = all(d -> !(pos[d] < mins_f[d] || pos[d] > maxs_f[d]), 1:D)
    inside && return 0
    if D == 2
        d_left   = is_per_svec[1] ? T(Inf) : abs(pos[1] - mins_f[1])
        d_right  = is_per_svec[1] ? T(Inf) : abs(pos[1] - maxs_f[1])
        d_bottom = is_per_svec[2] ? T(Inf) : abs(pos[2] - mins_f[2])
        d_top    = is_per_svec[2] ? T(Inf) : abs(pos[2] - maxs_f[2])
        min_d = min(d_left, d_right, d_bottom, d_top)
        if min_d == d_left; return 1
        elseif min_d == d_right; return 2
        elseif min_d == d_bottom; return 3
        else; return 4
        end
    else
        return 1
    end
end

# Geometric reference: among the axes on which the point is OUTSIDE, the one it is farthest beyond (exterior
# distance); "only outside on one axis" points get that axis' face.
function outside_axes(pos, mins, maxs, per)
    [d for d in eachindex(pos) if !per[d] && (pos[d] < mins[d] || pos[d] > maxs[d])]
end

quiet(f) = with_logger(f, NullLogger())

println("== 1. 2-D: new tags == parent tags on ParticleGrids ==")
global n2_grids = 0; global n2_pts = 0; global n2_bad = 0
for T in (Float64, Float32),
    (mins, maxs) in (((0.0, 0.0), (1.0, 1.0)), ((-1.0, -1.0), (1.0, 1.0)), ((2.3, -7.0), (5.1, -3.2)), ((0.0, 0.0), (3.0, 0.5))),
    per in (false, true, (true, false), (false, true)),
    (dx, dy) in ((0.05, 0.05), (0.04, 0.1), (0.1, 0.025), (1/30, 1/30)),
    jit in (0.0, 0.2, 0.45),
    rf in (2.5, 5.5)

    g   = get_rectangular_domain(T, mins, maxs; is_periodic = per)
    dxs = (T(dx * (maxs[1] - mins[1])), T(dy * (maxs[2] - mins[2])))
    pg  = quiet(() -> ParticleGrid(g, dxs, ExponentialWeightFunction(1.0, rf * maximum(dxs)), 1;
                                   randomness = (T(jit) * dxs[1], T(jit) * dxs[2]), rng = MersenneTwister(7)))
    global n2_grids += 1
    for i in 1:pg.meta.N
        p = pg.core.positions[i]
        global n2_pts += 1
        t_old = old_tag(p, g.mins, g.maxs, g.is_periodic, T)
        if pg.core.tags[i] != t_old || g.get_tag(p) != t_old || pg.core.is_boundary[i] != (t_old != 0)
            global n2_bad += 1
            n2_bad <= 5 && println("  MISMATCH T=$T box=$mins..$maxs per=$per p=$p new=$(pg.core.tags[i]) old=$t_old")
        end
    end
end
println("  grids $n2_grids, particles $n2_pts, mismatches $n2_bad")

println("== 2. 2-D: random + adversarial points (ties, corners, Inf, NaN, huge) ==")
let bad = 0, n = 0, rng = MersenneTwister(1)
    for T in (Float64, Float32), per in (false, true, (true, false), (false, true))
        mins, maxs = (T(-0.3), T(1.0)), (T(0.7), T(2.5))
        g = get_rectangular_domain(T, mins, maxs; is_periodic = per)
        pts = SVector{2, T}[]
        for _ in 1:200_000
            push!(pts, SVector{2, T}(T(-1.5) + T(3.5) * rand(rng, T), T(0) + T(3.5) * rand(rng, T)))
        end
        # exact ties: points on the diagonals through the corners, lattice of exact distances
        for a in T.(range(0, 0.5, length = 51)), b in T.(range(0, 0.5, length = 51))
            for (x, y) in ((mins[1] - a, mins[2] - b), (maxs[1] + a, mins[2] - b), (mins[1] - a, maxs[2] + b),
                           (maxs[1] + a, maxs[2] + b), (mins[1] + a, mins[2] - a), (mins[1] - a, mins[2] + a),
                           (maxs[1] - b, maxs[2] + b), (maxs[1] + b, maxs[2] - b))
                push!(pts, SVector{2, T}(x, y))
            end
        end
        for s in (T(Inf), -T(Inf), floatmax(T), -floatmax(T))
            push!(pts, SVector{2, T}(s, T(1.5)), SVector{2, T}(T(0.1), s), SVector{2, T}(s, s))
        end
        for p in pts
            n += 1
            g.get_tag(p) == old_tag(p, g.mins, g.maxs, g.is_periodic, T) || (bad += 1; bad <= 5 && println("  MISMATCH $T per=$per p=$p new=$(g.get_tag(p)) old=$(old_tag(p, g.mins, g.maxs, g.is_periodic, T))"))
        end
        # NaN (not reachable from the generator, documented only)
        pn = SVector{2, T}(T(NaN), T(-1))
        per === false && T === Float64 && println("  NaN point $(pn): new $(g.get_tag(pn)), old $(old_tag(pn, g.mins, g.maxs, g.is_periodic, T)) (is_interior false? $(!g.is_interior(pn)))")
    end
    println("  points $n, mismatches $bad")
end

println("== 3. 1-D tags ==")
for T in (Float64, Float32), (lo, hi, dx) in ((0.0, 1.0, 0.1), (-1.0, 1.0, 2/30), (-5.0, 5.0, 0.02), (2.5, 2.75, 0.01)), jit in (0.0, 0.3)
    g  = get_rectangular_domain(T, (lo,), (hi,))
    pg = quiet(() -> ParticleGrid(g, (T(dx),), ExponentialWeightFunction(1.0, T(5.5dx)), 1; randomness = (T(jit * dx),), rng = MersenneTwister(3)))
    x  = [pg.core.positions[i][1] for i in 1:pg.meta.N]; tg = pg.core.tags[1:pg.meta.N]
    ok = all(i -> tg[i] == (x[i] < T(lo) ? 1 : x[i] > T(hi) ? 2 : 0), eachindex(x))
    println("  $T [$lo,$hi] dx=$dx jit=$jit: ok=$ok n1=$(count(==(1), tg)) n2=$(count(==(2), tg)) interior=$(count(==(0), tg))")
end
let g = get_rectangular_domain(Float64, (0.0,), (1.0,); is_periodic = true)
    pg = quiet(() -> ParticleGrid(g, (0.1,), ExponentialWeightFunction(1.0, 0.25), 1))
    println("  periodic 1-D: N_ghost = $(pg.meta.N_ghost); get_tag(-0.1) = $(g.get_tag(SVector(-0.1))), get_tag(1.2) = $(g.get_tag(SVector(1.2)))")
end

println("== 4. 3-D tags ==")
for per in (false, (false, false, true), (true, false, false), (false, true, true), true), T in (Float64, Float32)
    mins, maxs = (T(0), T(-1), T(2)), (T(1), T(0.5), T(2.6))
    g  = get_rectangular_domain(T, mins, maxs; is_periodic = per)
    dx = (T(1/8), T(1.5/12), T(0.6/5))
    pg = quiet(() -> ParticleGrid(g, dx, ExponentialWeightFunction(1.0, T(2.5) * maximum(dx)), 1; randomness = T(0.2) .* dx, rng = MersenneTwister(5)))
    N  = pg.meta.N; gh = [i for i in 1:N if pg.core.is_boundary[i]]
    perv = g.is_periodic
    # independent reference: argmin over non-periodic faces of |x_d - wall|, lower index first on ties
    function ref(p)
        cands = [(abs(p[d] - (k == 0 ? g.mins[d] : g.maxs[d])), 2d - 1 + k) for d in 1:3 if !perv[d] for k in 0:1]
        isempty(cands) ? 1 : minimum(cands)[2]
    end
    ok  = all(i -> pg.core.tags[i] == ref(pg.core.positions[i]), gh)
    cnt = [count(i -> pg.core.tags[i] == t, gh) for t in 1:6]
    # ghosts outside on exactly one axis whose tag is a different axis (face-plane distance from the inside wins)
    wrong_axis = count(gh) do i
        p = pg.core.positions[i]; ax = outside_axes(p, g.mins, g.maxs, perv)
        length(ax) == 1 && (pg.core.tags[i] + 1) ÷ 2 != ax[1]
    end
    one_axis = count(i -> length(outside_axes(pg.core.positions[i], g.mins, g.maxs, perv)) == 1, gh)
    println("  per=$per $T: ghosts $(length(gh)) ok=$ok counts(1..6)=$cnt; single-axis ghosts $one_axis, of these tagged with another axis $wrong_axis")
end

println("== 5. 2-D: ghosts outside on one axis only that carry the other axis' tag (parent behaviour, kept) ==")
for (n, rf, jit) in ((20, 2.5, 0.0), (20, 5.5, 0.0), (50, 2.5, 0.2), (100, 5.5, 0.2))
    g  = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0))
    h  = 1 / n
    pg = quiet(() -> ParticleGrid(g, (h, h), ExponentialWeightFunction(1.0, rf * h), 1; randomness = (jit * h, jit * h), rng = MersenneTwister(2)))
    gh = [i for i in 1:pg.meta.N if pg.core.is_boundary[i]]
    bad = [i for i in gh if (ax = outside_axes(pg.core.positions[i], g.mins, g.maxs, g.is_periodic); length(ax) == 1 && (pg.core.tags[i] + 1) ÷ 2 != ax[1])]
    ex = isempty(bad) ? "" : " e.g. $(round.(Tuple(pg.core.positions[bad[1]]), digits = 4)) -> $(pg.core.tags[bad[1]])"
    println("  n=$n R=$(rf)h jit=$jit: ghosts $(length(gh)), outside one axis but tagged with the other $(length(bad))$ex")
end

println("== 6. spherical unchanged ==")
let g = get_spherical_domain(Float64, (0.0, 0.0, 0.0), 1.0)
    pg = quiet(() -> ParticleGrid(g, (0.2, 0.2, 0.2), ExponentialWeightFunction(1.0, 0.5), 1))
    println("  3-D sphere: ghost tags $(unique(pg.core.tags[i] for i in 1:pg.meta.N if pg.core.is_boundary[i]))")
end

println("== 7. reorder / adaptivity / device copy keep tags consistent with positions ==")
let bc = Dict{Int, AbstractBoundaryCondition}(1 => FixedDirichlet(), 2 => OutflowBC())
    g  = get_rectangular_domain(Float64, (0.0,), (1.0,); bc_map = bc)
    dx = 0.02
    pg = quiet(() -> ParticleGrid(g, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1; randomness = (0.3dx,), rng = MersenneTwister(9)))
    consistent(pg) = all(i -> pg.core.tags[i] == g.get_tag(pg.core.positions[i]), 1:pg.meta.N)
    println("  after construction (reordered): $(consistent(pg))")
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1 + sin(6pg.core.positions[i][1])); end
    # force events: push two interior particles together, open a gap
    xs = [pg.core.positions[i][1] for i in 1:pg.meta.N]
    i1 = argmin(i -> abs(xs[i] - 0.5), 1:pg.meta.N); i2 = argmin(i -> abs(xs[i] - 0.3), 1:pg.meta.N)
    pg.core.positions[i1] = Space{1, Float64}(xs[argmin(i -> abs(xs[i] - 0.52), 1:pg.meta.N)] - 1e-4)
    pg.core.positions[i2] = Space{1, Float64}(0.7001)
    pg.neighbor(pg); update_volumes!(pg, ShepardVolumes())
    N0 = pg.meta.N
    rep = adapt_particles!(pg, ParticleAdaptivity(Float64), 2)
    println("  adapt: inserted $(rep.inserted), removed $(rep.removed), N $N0 -> $(pg.meta.N); consistent: $(consistent(pg)); ghost tags $(sort(unique(pg.core.tags[i] for i in 1:pg.meta.N if pg.core.is_boundary[i])))")
    pg.reorder(pg)
    println("  after explicit reorder: $(consistent(pg))")
    dg = HC.device_grid(Array, pg)
    println("  device_grid(Array) tags equal: $(Array(dg.core.tags)[1:pg.meta.N] == pg.core.tags[1:pg.meta.N])")
end
