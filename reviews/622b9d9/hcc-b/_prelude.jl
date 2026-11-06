# Same imports and physics helpers as test/runtests.jl (no test groups run)
using Test, HyperCloudCore, PDEStudioCore, StaticArrays, LinearAlgebra, Random, Printf
import HyperCloudCore: flux, velocity, max_eigenvalue, prim2cons, cons2prim, fluid_velocity,
                       implicit_solve, evaluate_source, pre_solve_update!,
                       math_min, math_max, basis_length, param2vel,
                       update_size!, update_content!, _extract_order, _order_reach, _divergence_degree,
                       has_cached_gradients, grid_velocity
const REPO_TEST = "/workspace/julia/HyperCloudDev/HyperCloudCore.jl/test"
include(joinpath(REPO_TEST, "ICs/_main.jl"))
include(joinpath(REPO_TEST, "PDEs/_main.jl"))
function make_pg_1d(N::Int; mover = NoGridMover(), range = 2.5, T = Float64)
    dx   = T(2) / N
    geom = get_rectangular_domain(T, (-1.0,), (1.0,); is_periodic = true)
    wf   = ExponentialWeightFunction(one(T), T(range) * dx)
    return ParticleGrid(geom, (dx,), wf, 1; mover = mover)
end
