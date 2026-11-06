# Is Dict(t => OutflowBC() for t in 1:6) on the new tags the same as the old all-tag-1 Dict(1 => OutflowBC())?
using HyperCloudCore, StaticArrays, Logging, Random
import HyperCloudCore: flux, velocity, max_eigenvalue, prim2cons, cons2prim, param2vel
using LinearAlgebra
abstract type InitialCondition end
include(joinpath(dirname(Base.active_project()), "PDEs", "linear_advection.jl"))
eq = LinearAdvection(((1.0,), (0.5,), (0.25,)), Float64)
for (n, rf, jit, per) in ((8, 2.5, 0.0, false), (12, 2.5, 0.3, false), (10, 5.5, 0.2, false), (10, 2.5, 0.3, (false, false, true)))
    g  = get_rectangular_domain(Float64, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0); is_periodic = per)
    h  = 1 / n
    pg = with_logger(() -> ParticleGrid(g, (h, h, h), ExponentialWeightFunction(1.0, rf * h), 1; randomness = (jit * h, jit * h, jit * h), rng = MersenneTwister(4)), NullLogger())
    N  = pg.meta.N
    for i in 1:N; x = pg.core.positions[i]; pg.rhos[i] = State{1, Float64}(sin(3x[1]) + cos(2x[2]) * x[3]^2); end
    buf_new = copy(pg.rhos)
    for t in 1:6; OutflowBC()(pg, buf_new, t, nothing, eq, 0.0); end
    tags0 = copy(pg.core.tags[1:N])
    for i in 1:N; pg.core.is_boundary[i] && (pg.core.tags[i] = 1); end
    buf_old = copy(pg.rhos)
    OutflowBC()(pg, buf_old, 1, nothing, eq, 0.0)
    gh = [i for i in 1:N if pg.core.is_boundary[i]]
    diff = count(i -> buf_new[i] != buf_old[i], gh)
    orphan_new = count(i -> buf_new[i] == pg.rhos[i] && buf_old[i] != pg.rhos[i], gh)
    md = maximum(i -> abs(buf_new[i][1] - buf_old[i][1]), gh)
    println("n=$n R=$(rf)h jit=$jit per=$per: ghosts $(length(gh)), different outflow states $diff (max |Δ| $(round(md, sigdigits = 3))), left at the initial state only per-tag: $orphan_new")
end
