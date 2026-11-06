# Usage: julia -t 2 --project=<ROOT>/test probe.jl <ROOT> <tag>
using Test
using HyperCloudCore
import HyperCloudCore: flux, velocity, max_eigenvalue, prim2cons, cons2prim, fluid_velocity,
                       implicit_solve, evaluate_source, pre_solve_update!,
                       math_min, math_max, basis_length, param2vel,
                       update_size!, update_content!, _extract_order, _order_reach, _divergence_degree,
                       has_cached_gradients, grid_velocity
using PDEStudioCore
using StaticArrays, LinearAlgebra, Random, Serialization, Printf
const HC = HyperCloudCore
const ROOT = ARGS[1]
const TAG  = length(ARGS) >= 2 ? ARGS[2] : "x"
include(joinpath(ROOT, "test/ICs/_main.jl"))
include(joinpath(ROOT, "test/PDEs/_main.jl"))
interior_indices(pg) = findall(i -> !pg.core.is_boundary[i], 1:pg.meta.N)
function grid(D, T; n, R = 2.5, jit = 0.0, per = true, seed = 1, lo = 0.0, hi = 1.0, bc = nothing, mover = NoGridMover(), M = 1)
    geom = per ? get_rectangular_domain(T, ntuple(_ -> lo, D), ntuple(_ -> hi, D); is_periodic = true) :
                 get_rectangular_domain(T, ntuple(_ -> lo, D), ntuple(_ -> hi, D); bc_map = bc)
    dx = T((hi - lo) / n)
    ParticleGrid(geom, ntuple(_ -> dx, D), ExponentialWeightFunction(one(T), T(R) * dx), M;
                 randomness = ntuple(_ -> T(jit) * dx, D), rng = MersenneTwister(seed), mover = mover)
end
