# A 1-D run where tag 2 matters: u_t + u_x = 0 on [0, 1], a Gaussian pulse that leaves through the right wall,
# bc_map = Dict(1 => FixedDirichlet(), 2 => OutflowBC()) (the HyperCloud Sod-tube map), and Dict(1 => OutflowBC()).
# Run on the branch and on the parent: julia -t 2 --project=<tree>/test probe_tag2_run.jl
using HyperCloudCore, StaticArrays, Logging
import HyperCloudCore: flux, velocity, max_eigenvalue, prim2cons, cons2prim, param2vel
using LinearAlgebra
abstract type InitialCondition end
include(joinpath(dirname(Base.active_project()), "PDEs", "linear_advection.jl"))

set_ic!(pg, f) = (for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f(pg.core.positions[i][1])); end; pg)

function run_case(bc_map; N = 100, tmax = 0.4)
    dx   = 1.0 / N
    geom = get_rectangular_domain(Float64, (0.0,), (1.0,); bc_map = bc_map)
    pg   = with_logger(NullLogger()) do
        ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1)
    end
    set_ic!(pg, x -> exp(-(x - 0.6)^2 / 0.005))
    eq = LinearAdvection(((1.0,),), Float64)
    ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 1, 1, 2, NoLimiter(), RusanovFlux()), MOOD(), RK2_SSP_Tableau(Float64))
    t, dt = 0.0, 0.4dx
    while t < tmax - 1e-12
        ts(eq, pg, t, dt); t += dt
    end
    n = pg.meta.N; x = [pg.core.positions[i][1] for i in 1:n]; u = [pg.rhos[i][1] for i in 1:n]
    inner = [i for i in 1:n if !pg.core.is_boundary[i]]
    right = [i for i in 1:n if pg.core.is_boundary[i] && x[i] > 0.5]
    return (; max_inner = maximum(abs, u[inner]), max_right_ghost = maximum(abs, u[right]),
              mass = sum(u[inner]) * dx, tags_right = sort(unique(pg.core.tags[right])))
end

for (name, bc) in (("1=>Dirichlet, 2=>Outflow", Dict{Int, AbstractBoundaryCondition}(1 => FixedDirichlet(), 2 => OutflowBC())),
                   ("1=>Outflow only        ", Dict{Int, AbstractBoundaryCondition}(1 => OutflowBC())),
                   ("1=>Outflow, 2=>Outflow ", Dict{Int, AbstractBoundaryCondition}(1 => OutflowBC(), 2 => OutflowBC())))
    r = run_case(bc)
    println("$name: right ghost tags $(r.tags_right), max|u| interior $(round(r.max_inner, sigdigits = 4)), ",
            "max|u| right ghosts $(round(r.max_right_ghost, sigdigits = 4)), interior mass $(round(r.mass, sigdigits = 4)) (pulse mass at t=0: $(round(sqrt(pi * 0.005), sigdigits = 4)))")
end
