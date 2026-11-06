# Probes for src/Grid: reordering after growth, boundary classification, 1-D tags, Shepard quadrature order.
using Test, HyperCloudCore, StaticArrays, Random
import HyperCloudCore: limiter_phi, _update_phi

@testset "probe_grid" begin

@testset "_update_phi: a tiny increment must not limit (Float32)" begin
    f_i = State{1, Float32}(0.0f0); umax = State{1, Float32}(1.0f0); umin = State{1, Float32}(-1.0f0)
    phi = HyperCloudCore._update_phi(BarthJespersenLimiter(:hard), ones(State{1, Float32}), State{1, Float32}(1.0f-8), f_i, umax, umin)
    println("BJ phi for increment 1e-8 within [-1, 1] (Float32): ", phi)
    @test phi[1] == 1.0f0
end

@testset "reorder after growth (documented: call between steps)" begin
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = true)
    pg = ParticleGrid(geom, (0.1, 0.1), ExponentialWeightFunction(1.0, 0.25), 1)
    N = pg.meta.N
    # fresh grid, shuffled positions: works
    perm = shuffle(MersenneTwister(1), 1:N)
    pos0 = pg.core.positions[1:N]
    for i in 1:N; pg.core.positions[i] = pos0[perm[i]]; end
    @test (pg.reorder(pg); true)
    # after growth of the buffers (adaptivity does this): permute! with a longer permutation vector
    HyperCloudCore.ensure_capacity!(pg, N + 5)
    for i in 1:N; pg.core.positions[i] = pos0[perm[i]]; end
    ok = try
        pg.reorder(pg); true
    catch e
        println("reorder after ensure_capacity!: ", typeof(e), ": ", sprint(showerror, e)[1:min(end, 200)])
        false
    end
    @test ok
end

@testset "lattice points on the boundary are classified symmetrically" begin
    for (lo, hi, dx, irf) in ((0.0, 1.0, 0.1, 2.5), (-1.0, 1.0, 2 / 30, 2.5), (0.0, 1.0, 1 / 30, 3.0), (0.0, 0.7, 0.07, 2.5), (0.0, 1.0, 0.1, 2.5))
        geom = get_rectangular_domain(Float64, (lo,), (hi,); is_periodic = false)
        pg = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, irf * dx), 1)
        N = pg.meta.N
        xs = sort([pg.core.positions[i][1] for i in 1:N if !pg.core.is_boundary[i]])
        n_expected = round(Int, (hi - lo) / dx) + 1
        println("[$lo, $hi], dx = $dx: interior = $(length(xs)) (lattice points in the closed interval: $n_expected), ",
                "first = $(xs[1]), last = $(xs[end])")
        @test length(xs) == n_expected
    end
end

@testset "1-D rectangular domain: left and right walls get distinct tags" begin
    geom = get_rectangular_domain(Float64, (0.0,), (1.0,))
    pg = ParticleGrid(geom, (0.1,), ExponentialWeightFunction(1.0, 0.25), 1)
    N = pg.meta.N
    tl = unique(pg.core.tags[i] for i in 1:N if pg.core.positions[i][1] < 0)
    tr = unique(pg.core.tags[i] for i in 1:N if pg.core.positions[i][1] > 1)
    println("1-D tags left: $tl, right: $tr")
    @test tl != tr
end

@testset "Shepard volume quadrature order on a non-periodic box (claimed 2nd order)" begin
    f = x -> x^2                                   # ∫_0^1 x² = 1/3
    errs = Dict{Symbol, Vector{Float64}}(:shepard => Float64[], :mls2 => Float64[], :shepard_interior => Float64[])
    hs = (1 / 20, 1 / 40, 1 / 80, 1 / 160)
    for h in hs
        geom = get_rectangular_domain(Float64, (0.0,), (1.0,))
        pg = ParticleGrid(geom, (h,), ExponentialWeightFunction(1.0, 2.5h), 1)
        N = pg.meta.N
        x = i -> pg.core.positions[i][1]
        q = () -> sum(pg.core.volumes[i] * f(x(i)) for i in 1:N)
        push!(errs[:shepard], abs(q() - 1 / 3))
        push!(errs[:shepard_interior], abs(sum(pg.core.volumes[i] for i in 1:N if !pg.core.is_boundary[i]) - 1))
        update_volumes!(pg, MLSVolumes(2; samples_per_dx = 4))
        push!(errs[:mls2], abs(q() - 1 / 3))
    end
    for (k, e) in errs
        eoc = [log2(e[j] / e[j + 1]) for j in 1:length(e) - 1]
        println(rpad(k, 18), " errors = ", round.(e; sigdigits = 3), "   EOC = ", round.(eoc; digits = 2))
    end
    eoc_sh = log2(errs[:shepard][end - 1] / errs[:shepard][end])
    @test eoc_sh > 1.5
end

end
