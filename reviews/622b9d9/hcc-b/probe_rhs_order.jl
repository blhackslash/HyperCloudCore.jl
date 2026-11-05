# Probe: consistency order of the spatial operators: the RHS K = -div F of linear advection for a smooth periodic
# state against the exact -a·∇u, max norm over all particles, at n and 2n particles per unit length.
include(joinpath(@__DIR__, "_prelude.jl"))

function rhs_error(make_interp, D, n; jit = 0.0, R = 2.5, order = 1)
    dx = 2 / n
    geom = get_rectangular_domain(Float64, ntuple(_ -> -1.0, D), ntuple(_ -> 1.0, D); is_periodic = true)
    pg = ParticleGrid(geom, ntuple(_ -> dx, D), ExponentialWeightFunction(1.0, R * dx), 1;
                      randomness = ntuple(_ -> jit * dx, D), rng = MersenneTwister(7))
    a = D == 1 ? (1.0,) : (1.0, 0.5)
    eq = LinearAdvection(ntuple(d -> (a[d],), D), Float64)
    u(x) = prod(sin(π * x[d] + 0.3d) for d in 1:D)
    du(x) = sum(a[d] * π * cos(π * x[d] + 0.3d) * prod(sin(π * x[e] + 0.3e) for e in 1:D if e != d; init = 1.0) for d in 1:D)
    N = pg.meta.N
    for i in 1:N; pg.rhos[i] = State{1, Float64}(u(pg.core.positions[i])); end
    m = make_interp(D)
    ts = GeneralRKTimeStepper(eq, m, MOOD(), RK3_SSP_Tableau(Float64))
    HyperCloudCore.update_size!(ts, N, length(pg.neighbor.indices), false); HyperCloudCore.update_size!(m, N)
    fill!(view(pg.core.particle_orders, 1:N), order)
    HyperCloudCore.compute_explicit_rhs!(ts, eq, pg, ts.K_E_stages[1], ts.K_X_stages[1], pg.rhos, 0.0, true)
    return maximum(abs(parent(ts.K_E_stages[1])[i][1] + du(pg.core.positions[i])) for i in 1:N)
end

cases = [
    ("Central 1",  D -> CentralDivergence(Float64, D, 1, 1), 1, 2),
    ("Central 2",  D -> CentralDivergence(Float64, D, 1, 2), 2, 2),
    ("Central 3",  D -> CentralDivergence(Float64, D, 1, 3), 3, 4),
    ("WENO 2",     D -> WENO(Float64, D, 1, 2), 2, 1),
    ("WENO 3",     D -> WENO(Float64, D, 1, 3), 3, 1),
    ("Upwind Cl.", D -> UpwindDivergence(Float64, D, 1, 1, :Classic, RusanovFlux()), 1, 1),
    ("MUSCL 1",    D -> MUSCL(Float64, D, 1, 1, NoLimiter(), RusanovFlux()), 1, 1),
    ("MUSCL 2",    D -> MUSCL(Float64, D, 1, 2, NoLimiter(), RusanovFlux()), 2, 2),
    ("MUSCL 3",    D -> MUSCL(Float64, D, 1, 3, NoLimiter(), RusanovFlux()), 3, 3),
    ("MUSCL 4",    D -> MUSCL(Float64, D, 1, 4, NoLimiter(), RusanovFlux()), 4, 4),
]
@testset "RHS consistency order" begin
    for (D, jit, R, n) in ((1, 0.0, 2.5, 40), (1, 0.0, 3.5, 40), (1, 0.25, 3.5, 40), (2, 0.0, 3.5, 16), (2, 0.25, 3.5, 16))
        for (name, mk, order, _) in cases
            e1 = rhs_error(mk, D, n; jit, R, order); e2 = rhs_error(mk, D, 2n; jit, R, order)
            @printf("D=%d jit=%.2f R=%.1f  %-11s  err %.2e -> %.2e  EOC %.2f\n", D, jit, R, name, e1, e2, log2(e1 / e2))
        end
    end
end
