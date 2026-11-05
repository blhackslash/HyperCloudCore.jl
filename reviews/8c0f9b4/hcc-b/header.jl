using Test
using HyperCloudCore
import HyperCloudCore: flux, velocity, max_eigenvalue, prim2cons, cons2prim, fluid_velocity,
                       implicit_solve, evaluate_source, pre_solve_update!,
                       math_min, math_max, basis_length, param2vel,
                       update_size!, update_content!, _extract_order, _order_reach, _divergence_degree,
                       has_cached_gradients, grid_velocity
using PDEStudioCore
using StaticArrays, LinearAlgebra, Random
const REPO = "/workspace/julia/HyperCloudDev/HyperCloudCore.jl/test"
include(joinpath(REPO, "ICs/_main.jl"))
include(joinpath(REPO, "PDEs/_main.jl"))
interior_indices(pg) = findall(i -> !pg.core.is_boundary[i], 1:pg.meta.N)
function grid1d(N; range = 2.5, jit = 0.0, T = Float64, seed = 1)
    dx = T(2) / N
    geom = get_rectangular_domain(T, (-1.0,), (1.0,); is_periodic = true)
    wf = ExponentialWeightFunction(one(T), T(range) * dx)
    jit == 0 ? ParticleGrid(geom, (dx,), wf, 1) :
               ParticleGrid(geom, (dx,), wf, 1; randomness = (T(jit) * dx,), rng = MersenneTwister(seed))
end
function grid2d(n; range = 3.5, jit = 0.0, T = Float64, seed = 1)
    dx = T(1) / n
    geom = get_rectangular_domain(T, (0.0, 0.0), (1.0, 1.0); is_periodic = true)
    wf = ExponentialWeightFunction(one(T), T(range) * dx)
    jit == 0 ? ParticleGrid(geom, (dx, dx), wf, 1) :
               ParticleGrid(geom, (dx, dx), wf, 1; randomness = (T(jit) * dx, T(jit) * dx), rng = MersenneTwister(seed))
end
function setic!(pg, f)
    T = eltype(pg.core.positions[1])
    for i in 1:pg.meta.N
        pg.rhos[i] = State{1, T}(T(f(pg.core.positions[i])))
    end
    pg
end
# run to tend with fixed dt from cfl*dx; returns max error vs exact at interior
function advect_err(pg, eq, scheme, bt; tend = 0.5, cfl = 0.2, f0 = x -> sin(pi * x[1]), c = (1.0,), mood = MOOD())
    T = eltype(pg.core.positions[1])
    setic!(pg, f0)
    ts = GeneralRKTimeStepper(eq, scheme, mood, bt)
    dx = minimum(pg.meta.dx)
    n = ceil(Int, tend / (cfl * dx)); dt = T(tend / n)
    t = zero(T)
    for k in 1:n
        ts(eq, pg, t, dt); t += dt
    end
    err = zero(T); mass = zero(T)
    for i in interior_indices(pg)
        x = pg.core.positions[i]
        ex = f0(x .- T.(c) .* tend)
        err = max(err, abs(pg.rhos[i][1] - ex))
    end
    return err
end
