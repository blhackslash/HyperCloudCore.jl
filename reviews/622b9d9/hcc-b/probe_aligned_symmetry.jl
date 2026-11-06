# Probe: AlignedRusanovFlux claims "the rule is symmetric in the pair, so both sides evaluate the same flux".
# The orientation of axis d on slot (i, j) is sign(c_ij,d) from i's side and sign(c_ji,d) from j's side; both
# sides agree on the interface flux only if sign(c_ji,d) = -sign(c_ij,d). Count the (pair, axis) where they agree
# in sign (then i and j evaluate F* with opposite dissipation signs).
include(joinpath(@__DIR__, "_prelude.jl"))

function sign_stats(D, order, div_order, jit, Rk; n = D == 1 ? 40 : 20)
    dx = 1 / n
    geom = get_rectangular_domain(Float64, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D); is_periodic = true)
    pg = ParticleGrid(geom, ntuple(_ -> dx, D), ExponentialWeightFunction(1.0, Rk * dx), 1;
                      randomness = ntuple(_ -> jit * dx, D), rng = MersenneTwister(3))
    m = MUSCL(Float64, D, 1, order, NoLimiter(), AlignedRusanovFlux(); div_order = div_order)
    HyperCloudCore.update_size!(m, pg.meta.N); HyperCloudCore.prepare_geometry!(m, pg)
    nb = pg.neighbor; N = pg.meta.N
    slot = Dict{Tuple{Int, Int}, Int}()
    for i in 1:N, k in nb.ranges[i]; slot[(i, nb.indices[k])] = k; end
    same = 0; total = 0; same_mag = 0.0; tot_mag = 0.0
    for i in 1:N, k in nb.ranges[i]
        j = nb.indices[k]; i < j || continue
        kk = get(slot, (j, i), 0); kk == 0 && continue
        for d in 1:D
            a, b = m.dweights[k][d], m.dweights[kk][d]
            (a == 0 || b == 0) && continue
            total += 1; tot_mag += abs(a) + abs(b)
            if sign(a) == sign(b); same += 1; same_mag += abs(a) + abs(b); end
        end
    end
    return same, total, same_mag / tot_mag
end

@testset "aligned orientation symmetry" begin
    for (D, order, div_order, jit, Rk) in ((1, 1, 0, 0.2, 2.5), (2, 1, 0, 0.2, 2.5), (2, 1, 0, 0.4, 2.5), (2, 3, 3, 0.2, 3.5), (2, 4, 0, 0.2, 3.5))
        same, total, frac = sign_stats(D, order, div_order, jit, Rk)
        @printf("D=%d order=%d div_order=%d jitter=%.1f R=%.1fdx: %d of %d (pair, axis) with sign(c_ij)=sign(c_ji) (%.1f%% of |c| mass)\n",
                D, order, div_order, jit, Rk, same, total, 100frac)
        @test same == 0   # what the docstring's symmetry claim needs
    end
end
