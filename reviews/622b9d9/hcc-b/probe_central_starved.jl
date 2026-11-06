# Probe (F10): Central/Upwind accept a degree-q divergence fit with basis_count(D, q) neighbours, but the fit has an
# intercept (basis_count + 1 unknowns). 1-D lattice with R = 1.5Δx: 2 neighbours; Central 2 (degree 2, 3 unknowns)
# passes the check and gets the Tikhonov-regularised rank-deficient fit.
include(joinpath(@__DIR__, "_prelude.jl"))
function rhs_err(order, n; R = 1.5)
    dx = 2 / n
    geom = get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true)
    pg = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, R * dx), 1)
    eq = LinearAdvection(((1.0,),), Float64); N = pg.meta.N
    for i in 1:N; pg.rhos[i] = State{1, Float64}(sin(π * pg.core.positions[i][1])); end
    m = CentralDivergence(Float64, 1, 1, order)
    ts = GeneralRKTimeStepper(eq, m, MOOD(), RK3_SSP_Tableau(Float64))
    HyperCloudCore.update_size!(ts, N, length(pg.neighbor.indices), false)
    fill!(view(pg.core.particle_orders, 1:N), order)
    HyperCloudCore.compute_explicit_rhs!(ts, eq, pg, ts.K_E_stages[1], ts.K_X_stages[1], pg.rhos, 0.0, true)
    return length(pg.neighbor.ranges[1]), maximum(abs(parent(ts.K_E_stages[1])[i][1] + π * cos(π * pg.core.positions[i][1])) for i in 1:N)
end
for o in (1, 2), n in (40, 80)
    nn, e = rhs_err(o, n); @printf("Central %d, %d neighbours, n = %d: max RHS error %.3e\n", o, nn, n, e)
end
