using Test
using Printf
using HyperCloudCore
import HyperCloudCore: flux, velocity, max_eigenvalue, prim2cons, cons2prim, fluid_velocity,
                       implicit_solve, evaluate_source, pre_solve_update!,
                       math_min, math_max, basis_length, param2vel,
                       update_size!, update_content!, _extract_order, _order_reach, _divergence_degree,
                       has_cached_gradients, grid_velocity
using PDEStudioCore
using StaticArrays, LinearAlgebra, Random
const HC = HyperCloudCore
const REPO = "/workspace/julia/HyperCloudDev/HyperCloudCore.jl/.claude/worktrees/grid-adaptivity/test"
include(joinpath(REPO, "ICs/_main.jl"))
include(joinpath(REPO, "PDEs/_main.jl"))
interior_indices(pg) = findall(i -> !pg.core.is_boundary[i], 1:pg.meta.N)
