# Positivity of limited reconstructions near vacuum: direct _update_phi call and a Float32 MUSCL2 dip
include(joinpath(@__DIR__, "common.jl"))
for (T, fi, d) in ((Float32, 1f-8, -1f-7), (Float32, 0f0, -1f-6), (Float64, 1e-17, -1e-15), (Float64, 0.0, -1e-15))
    phi = HCC._update_phi(BarthJespersenLimiter(:hard), ones(State{1, T}), State{1, T}(d), State{1, T}(fi), State{1, T}(one(T)), State{1, T}(fi))[1]
    println("T=$T f_i=$fi u_min=$fi u_max=1 delta=$d: phi = $phi, limited interface value = $(fi + phi * d)")
end
# MUSCL2/3 hard BJ on a Float32 lattice with near-vacuum dips next to O(1) data: worst interface value below u_min
for T in (Float32, Float64), N in (50, 64, 100, 128, 200, 400, 1000), order in (2, 3), lim in (BarthJespersenLimiter(:hard), VenkatakrishnanLimiter(:hard))
    pg = grid1d(N; T = T)
    vac = T == Float32 ? 1f-9 : 1e-18
    setic!(pg, x -> (mod(round(Int, (x[1] + 1) * N / 2), 7) == 3 ? vac : one(T)))
    n = pg.meta.N
    m = MUSCL(T, 1, 1, order, lim, RusanovFlux())
    HCC.update_size!(m, n)
    ib = HCC.InteractionBuffer{1, 1, T}(); HCC.update_size!(ib, length(pg.neighbor.indices))
    fill!(view(pg.core.particle_orders, 1:n), order)
    nb = pg.neighbor
    for i in 1:n; HCC.update_content!(m, i, pg.rhos[i], nb.ranges[i], pg, pg.rhos, ib); end
    worst = zero(T); nneg = 0
    for i in 1:n
        lo = min(pg.rhos[i][1], minimum(pg.rhos[nb.indices[k]][1] for k in nb.ranges[i]))
        for k in nb.ranges[i]
            p = HCC.build_basis(Val(order - 1), T(0.5) * nb.distances[k])
            r = pg.rhos[i][1] + sum(m.gradients[i][b][1] * p[b] for b in eachindex(p))
            worst = max(worst, lo - r); r < 0 && (nneg += 1)
        end
    end
    println("T=$T N=$N MUSCL$order $(nameof(typeof(lim))): max undershoot below local min = $worst, negative interface values = $nneg")
end
