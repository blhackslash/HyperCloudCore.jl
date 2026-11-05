# docs/src/spatial/muscl.md (545ae4c): Σ|c_ij|Δx of the CFL estimate vs the fit in use, 1-D lattice.
include(joinpath(@__DIR__, "header.jl"))
function sumc(R, q; N = 40)
    dx = 2 / N
    geom = get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true)
    pg = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, R * dx), 1)
    i = 10; sl = pg.neighbor.ranges[i]
    s = 0.0
    for k in sl
        F = [Flux{1, 1, Float64}((State{1, Float64}(kk == k ? 1.0 : 0.0),)) for kk in 1:length(pg.neighbor.indices)]
        c = HC._mls_divergence(sl, pg.neighbor.distances, pg.neighbor.weights, F, dx, HC.basis_length(Val(1), Val(q)), Val(q), Val(1))[1]
        s += abs(c)
    end
    return s * dx, length(sl)
end
eq = LinearAdvection(((1.0,),), Float64)
for (name, order, R) in (("MUSCL3", 3, 2.5), ("MUSCL5", 5, 3.5), ("MUSCL5", 5, 2.5))
    m = MUSCL(Float64, 1, 1, order, NoLimiter(), RusanovFlux())
    qcfl = HC._divergence_degree(m)
    nnb = sumc(R, 1)[2]
    quse = HC._divergence_index(Val(0), Val(1), order, nnb) - 1
    a, b = sumc(R, qcfl)[1], sumc(R, quse)[1]
    # the step get_time_step returns (no cache) vs the one from the fit in use
    dx = 2 / 40
    geom = get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true)
    pg = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, R * dx), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0); end
    dt = get_time_step(pg, eq, m; cached_weights = false)
    println("$name R=$(R)Δx: CFL degree $qcfl Σ|c|Δx = $(round(a, digits=3)); degree in use $quse Σ|c|Δx = $(round(b, digits=3)); ratio $(round(a / b, digits=2)); get_time_step·Σ|c_use|/Δx = $(round(dt * b / dx, digits=3))")
end
println("cached path (after one step):")
for (name, order, R) in (("MUSCL3", 3, 2.5), ("MUSCL5", 5, 3.5), ("MUSCL5", 5, 2.5))
    m = MUSCL(Float64, 1, 1, order, NoLimiter(), RusanovFlux())
    dx = 2 / 40
    geom = get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true)
    pg = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, R * dx), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0); end
    ts = GeneralRKTimeStepper(eq, m, MOOD(), RK3_SSP_Tableau(Float64))
    ts(eq, pg, 0.0, 0.1dx)
    dtc = get_time_step(pg, eq, m); dtu = get_time_step(pg, eq, m; cached_weights = false)
    nnb = length(pg.neighbor.ranges[1]); quse = HC._divergence_index(Val(0), Val(1), order, nnb) - 1
    b = sumc(R, quse)[1]
    println("$name R=$(R)Δx: Σ|c|Δx implied by the cached dt = $(round(dx / dtc, digits=3)), by the uncached dt = $(round(dx / dtu, digits=3)), fit in use $(round(b, digits=3))")
end
