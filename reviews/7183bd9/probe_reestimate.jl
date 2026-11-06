# 1d80418: interior-only balance in _reestimate_volumes!. Walled 1-D/2-D boxes (FixedDirichlet / OutflowBC), 45 %
# jitter, Float64/Float32, volume_estimator = ShepardVolumes() / MLSVolumes(); and periodic boxes (must be unchanged
# versus the old code: printed hash of the states and volumes).
include(joinpath(@__DIR__, "header.jl"))
_interior(pg) = sum(i -> Float64(pg.core.volumes[i]) * Float64(pg.rhos[i][1]), filter(i -> !pg.core.is_boundary[i], 1:pg.meta.N))
_ghost(pg)    = sum(i -> Float64(pg.core.volumes[i]) * Float64(pg.rhos[i][1]), filter(i -> pg.core.is_boundary[i], 1:pg.meta.N); init = 0.0)
function grid(D, T; jit = 0.45, seed = 1, n = 20, bc = :dirichlet, periodic = false, est = nothing)
    tags = D == 2 ? (1, 2, 3, 4) : (1,)
    bcm  = periodic ? Dict{Int, AbstractBoundaryCondition}() :
           Dict{Int, AbstractBoundaryCondition}(t => (bc == :dirichlet ? FixedDirichlet() : OutflowBC()) for t in tags)
    geom = get_rectangular_domain(T, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D); bc_map = bcm, is_periodic = periodic)
    dx   = T(1 / n)
    pg   = ParticleGrid(geom, ntuple(_ -> dx, D), ExponentialWeightFunction(one(T), T(2.5) * dx), 1;
                        randomness = ntuple(_ -> T(jit) * dx, D), rng = MersenneTwister(seed))
    for i in 1:pg.meta.N
        x = pg.core.positions[i]
        pg.rhos[i] = State{1, T}(T(1) + sum(x) + T(0.3) * sinpi(T(3) * x[1]))
    end
    update_volumes!(pg, est === nothing ? ShepardVolumes() : est)   # the volumes of the estimator in use
    return pg
end
function run(D, T, est; seed, bc = :dirichlet, periodic = false)
    pg = grid(D, T; seed, bc, periodic, est)
    N = pg.meta.N
    oldpos = copy(view(pg.core.positions, 1:N)); oldU = copy(view(pg.rhos, 1:N))
    gs0 = Dict(pg.core.positions[i] => pg.rhos[i] for i in 1:N if pg.core.is_boundary[i])
    scale = sum(i -> Float64(pg.core.volumes[i]) * abs(Float64(pg.rhos[i][1])), 1:N)
    ci, cg, ct = _interior(pg), _ghost(pg), Float64(total_mass(pg)[1])
    rep = adapt_particles!(pg, ParticleAdaptivity(T; volume_estimator = est), 3)
    R = pg.meta.R; viol = 0.0
    for i in 1:pg.meta.N
        pg.core.is_boundary[i] && continue
        x = pg.core.positions[i]
        ks = [k for k in 1:N if sum(abs2, HC._wrapped_diff(oldpos[k], x, pg.domain.L_wrap, pg.domain.invL_wrap)) <= (2R)^2]
        lo, hi = minimum(k -> oldU[k][1], ks), maximum(k -> oldU[k][1], ks)
        viol = max(viol, Float64(lo - pg.rhos[i][1]), Float64(pg.rhos[i][1] - hi))
    end
    gs1 = Dict(pg.core.positions[i] => pg.rhos[i] for i in 1:pg.meta.N if pg.core.is_boundary[i])
    h = hash((collect(view(pg.rhos, 1:pg.meta.N)), collect(view(pg.core.volumes, 1:pg.meta.N))))
    return (ev = rep.inserted + rep.removed, d_in = (_interior(pg) - ci) / scale, d_gh = (_ghost(pg) - cg) / scale,
            d_tot = (Float64(total_mass(pg)[1]) - ct) / scale, mres = Float64(rep.mass_residual[1]) / scale,
            ghost_states_same = gs0 == gs1, viol, h)
end
for periodic in (false, true), D in (1, 2), T in (Float64, Float32), est in (nothing, ShepardVolumes(), MLSVolumes(1)), seed in 1:3
    periodic && T == Float32 && continue
    r = run(D, T, est; seed, periodic)
    @printf("%s D=%d %-7s est=%-14s seed=%d events=%2d d_in=%+.2e d_gh=%+.2e d_tot=%+.2e mres=%+.2e ghost_states_same=%d viol=%.1e hash=%x\n",
            periodic ? "periodic" : "walls   ", D, T, est === nothing ? "nothing" : string(nameof(typeof(est))), seed, r.ev, r.d_in, r.d_gh, r.d_tot, r.mres, r.ghost_states_same, r.viol, r.h)
end
