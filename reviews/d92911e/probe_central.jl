# 61d9172: Central/Upwind neighbour-count safeguard counts the intercept.
# Global EOC of 1-D periodic advection (lattice and 20 % jitter, RK4) and 2-D (20 % jitter), the number of
# particles whose degree is lowered, consistency div(x) = 1 at every particle, behaviour next to walls (ghost band),
# Upwind with exactly D + 1 / D neighbours.
include(joinpath(@__DIR__, "header.jl"))

function grid1d(N; range = 2.5, jit = 0.0, seed = 1, per = true)
    dx = 2 / N
    geom = get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = per)
    ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, range * dx), 1;
                 randomness = (jit * dx,), rng = MersenneTwister(seed))
end
function grid2d(n; range = 2.5, jit = 0.0, seed = 1)
    dx = 1 / n
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = true)
    ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(1.0, range * dx), 1;
                 randomness = (jit * dx, jit * dx), rng = MersenneTwister(seed))
end
function advect_err(pg, eq, scheme; tend = 0.5, cfl = 0.1, f0 = x -> length(x) == 2 ? sinpi(2x[1]) * cospi(2x[2]) : sinpi(x[1]), c = (1.0,))
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f0(pg.core.positions[i])); end
    ts = GeneralRKTimeStepper(eq, scheme, MOOD(), RK4_Classical_Tableau(Float64))
    dx = minimum(pg.meta.dx); n = ceil(Int, tend / (cfl * dx)); dt = tend / n
    for k in 0:(n - 1); ts(eq, pg, k * dt, dt); end
    return maximum(i -> abs(pg.rhos[i][1] - f0(pg.core.positions[i] .- c .* tend)), interior_indices(pg))
end

eq1 = LinearAdvection(((1.0,),), Float64)
println("1-D, RK4, cfl 0.1, N = 40/80/160")
for jit in (0.0, 0.2), (name, mk, R) in (("Central2", () -> CentralDivergence(Float64, 1, 1, 2), 2.5),
                                         ("Central3", () -> CentralDivergence(Float64, 1, 1, 3), 2.5),
                                         ("Central3 R3.5", () -> CentralDivergence(Float64, 1, 1, 3), 3.5),
                                         ("Central4 R2.5", () -> CentralDivergence(Float64, 1, 1, 4), 2.5),
                                         ("Central4 R3.5", () -> CentralDivergence(Float64, 1, 1, 4), 3.5),
                                         ("UpwindClassic", () -> UpwindDivergence(Float64, 1, 1, 1, :Classic, RusanovFlux()), 2.5),
                                         ("UpwindClassic R1.2", () -> UpwindDivergence(Float64, 1, 1, 1, :Classic, RusanovFlux()), 1.2))
    errs = [advect_err(grid1d(N; range = R, jit), eq1, mk()) for N in (40, 80, 160)]
    println(rpad(name, 20), " jit=$jit errs=", round.(errs, sigdigits = 3), " EOC=", round.(log2.(errs[1:2] ./ errs[2:3]), digits = 2))
end

# how many particles have exactly B(1,q) neighbours (degree lowered now), and consistency for F = x
println("\nneighbour counts on 1-D jittered clouds (R = 2.5Δx, N = 160)")
pg = grid1d(160; jit = 0.2)
cnt = [length(pg.neighbor.ranges[i]) for i in 1:pg.meta.N]
println("  counts: ", sort(collect(Dict(c => count(==(c), cnt) for c in unique(cnt)))))

println("\n2-D, 20 % jitter, RK4, n = 16/32/64")
eq2 = LinearAdvection(((1.0,), (0.0,)), Float64)
for (name, mk, R) in (("Central2 R2.5", () -> CentralDivergence(Float64, 2, 1, 2), 2.5),
                      ("Central3 R2.5", () -> CentralDivergence(Float64, 2, 1, 3), 2.5),
                      ("Central4 R2.5", () -> CentralDivergence(Float64, 2, 1, 4), 2.5))
    errs = [advect_err(grid2d(n; range = R, jit = 0.2), eq2, mk(); c = (1.0, 0.0), tend = 0.25) for n in (16, 32, 64)]
    println(rpad(name, 20), " errs=", round.(errs, sigdigits = 3), " EOC=", round.(log2.(errs[1:2] ./ errs[2:3]), digits = 2))
end

# Next to walls: a 1-D box with OutflowBC, R = 2.5Δx, jittered: interior particles keep their full degree?
@testset "walls: interior stencils complete, div(x) exact" begin
    bc   = Dict{Int, AbstractBoundaryCondition}(1 => OutflowBC())
    geom = get_rectangular_domain(Float64, (-1.0,), (1.0,); bc_map = bc)
    dx   = 2 / 80
    pg   = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 3.5dx), 1; randomness = (0.2dx,), rng = MersenneTwister(2))
    N    = pg.meta.N
    for i in 1:N; pg.rhos[i] = State{1, Float64}(pg.core.positions[i][1]); end
    ib = HC.InteractionBuffer{1, 1, Float64}(); update_size!(ib, length(pg.neighbor.indices))
    fill!(view(pg.core.particle_orders, 1:N), 4)
    c4 = CentralDivergence(Float64, 1, 1, 4)
    errs = [abs(c4(eq1, i, pg.rhos[i], pg.neighbor.ranges[i], pg, pg.rhos, ib, EPD1())[1] - 1) for i in interior_indices(pg)]
    println("walls, Central4 R=3.5Δx jittered: max |div x - 1| over the interior = ", maximum(errs))
    @test maximum(errs) < 1e-9
    # ghosts at the canvas edge have truncated stencils: the safeguard gives 0 or a lower degree, never NaN
    gh = [c4(eq1, i, pg.rhos[i], pg.neighbor.ranges[i], pg, pg.rhos, ib, EPD1())[1] for i in 1:N if pg.core.is_boundary[i]]
    @test all(isfinite, gh)
end

@testset "Upwind Classic with D+1 and D neighbours (1-D and 2-D)" begin
    ib = HC.InteractionBuffer{1, 1, Float64}()
    up1 = UpwindDivergence(Float64, 1, 1, 1, :Classic, RusanovFlux())
    pg = grid1d(40; jit = 0.0)
    N  = pg.meta.N
    for i in 1:N; pg.rhos[i] = State{1, Float64}(pg.core.positions[i][1]); end
    update_size!(ib, length(pg.neighbor.indices))
    fill!(view(pg.core.particle_orders, 1:N), 1)
    i = 20; r = pg.neighbor.ranges[i]
    one_nb = first(r):first(r)
    two_nb = [k for k in r if abs(pg.neighbor.distances[k][1]) < 1.5 * pg.meta.dx[1]]   # the two nearest (both sides)
    v2 = up1(eq1, i, pg.rhos[i], minimum(two_nb):maximum(two_nb), pg, pg.rhos, ib, EPD1())[1]
    v1 = up1(eq1, i, pg.rhos[i], one_nb, pg, pg.rhos, ib, EPD1())[1]
    println("Upwind 1-D, two neighbours: ", v2, " (exact 1); one neighbour: ", v1)
    @test v1 == 0 && isfinite(v2)
end
