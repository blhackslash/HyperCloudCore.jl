using Test
using HyperCloudCore
import HyperCloudCore: flux, velocity, max_eigenvalue, prim2cons, cons2prim, fluid_velocity,
                       implicit_solve, evaluate_source, pre_solve_update!,
                       math_min, math_max, basis_length, param2vel,
                       update_size!, update_content!, _extract_order, _order_reach, _divergence_degree,
                       has_cached_gradients, grid_velocity
using PDEStudioCore
using StaticArrays
using LinearAlgebra
using Random

include("ICs/_main.jl")
include("PDEs/_main.jl")
include("SimulationFunctions/_main.jl")
include("Builder/kinetic.jl")        # physics-side builder (kinetic relaxation) stays with the tests

include("kinetic_source.jl")
include("time_integration.jl")

PDEStudioCore.set_target_module!(@__MODULE__)

# -------------------------------------------------------------------------
# Shared test helpers
# -------------------------------------------------------------------------

"Uniform 1-D periodic grid on [-1, 1] with N particles (no randomness)."
function make_pg_1d(N::Int; mover = NoGridMover(), range = 2.5, T = Float64)
    dx   = T(2) / N
    geom = get_rectangular_domain(T, (-1.0,), (1.0,); is_periodic = true)
    wf   = ExponentialWeightFunction(one(T), T(range) * dx)
    return ParticleGrid(geom, (dx,), wf, 1; mover = mover)
end

"Scalar 1-D linear advection with speed `c`."
make_advection_1d(c = 1.0) = LinearAdvection(((c,),), Float64)

"Set a smooth initial condition `f(x)` on a scalar grid."
function set_ic!(pg, f)
    for i in 1:pg.meta.N
        pg.rhos[i] = State{1, Float64}(f(pg.core.positions[i][1]))
    end
    return pg
end

interior_indices(pg) = findall(i -> !pg.core.is_boundary[i], 1:pg.meta.N)

@testset "HyperCloudCore" begin
    set_stat_preset!("hyperbolic")
    set_save_path!(mktempdir())

    include("unit_tests.jl")
    include("pipeline_tests.jl")
end
