# =========================================================================
# UNIT TESTS: API contracts, utilities, grid buffers, tableaus, MOOD kernel,
# time stepper, moving grids. Everything here runs in well under a second.
# =========================================================================

@testset "API contracts & fallbacks" begin
    struct DummyPDE <: HyperbolicPDE{1, 1, Float64} end
    struct DummyExplicitSource <: AbstractExplicitSourceTerm end
    struct DummyImplicitSource <: AbstractImplicitSourceTerm end
    struct DummyInterpolator <: DivergenceInterpolator end
    struct DummyTimeStepper <: TimeStepper end

    eq = DummyPDE()
    u  = SVector{1, Float64}(1.5)

    @testset "PDE" begin
        @test_throws ErrorException flux(eq, u)
        @test_throws ErrorException max_eigenvalue(eq, u, 1)
        @test_throws ErrorException velocity(eq, u, 1)
        @test_throws ErrorException fluid_velocity(eq, u)
        @test prim2cons(eq, u) == u
        @test cons2prim(eq, u) == u

        f_L = SVector{1}((SVector{1, Float64}(1.0),))
        f_R = SVector{1}((SVector{1, Float64}(2.0),))
        @test evaluate_nc_jump(eq, f_L, f_R, SVector{1, Float64}(0.5))[1][1] == 0.0
    end

    @testset "Sources" begin
        @test_throws ErrorException evaluate_source(DummyExplicitSource(), u, 1, nothing, 0.0)
        @test_throws ErrorException evaluate_source(DummyImplicitSource(), u, 1, nothing, 0.0)
        @test_throws ErrorException implicit_solve(DummyImplicitSource(), u, 0.1, 1, nothing, 0.0)
        @test evaluate_source(NoExplicitSource(), u, 1, nothing, 0.0) == zero(u)
        @test evaluate_source(NoImplicitSource(), u, 1, nothing, 0.0) == zero(u)
        @test implicit_solve(NoImplicitSource(), u, 0.1, 1, nothing, 0.0) == u
        @test pre_solve_update!(DummyImplicitSource(), nothing, nothing, 0.0) === nothing
    end

    @testset "Interpolator defaults" begin
        g = DummyInterpolator()
        @test_throws ErrorException update_size!(g, 100)
        @test_throws ErrorException _extract_order(g)
        @test update_content!(g, 1, u, 1:5, nothing, nothing) === nothing
        @test _order_reach(g, EPD1()) == 0
        @test has_cached_gradients(g) == false
    end

    @testset "TimeStepper default" begin
        @test_throws ErrorException update_size!(DummyTimeStepper(), 100, 50)
    end
end

@testset "Parameter casting & branchless math" begin
    @test param2uvec(5.0) == SVector(5.0)
    @test param2uvec((1.0, 2.0, 3.0)) == SVector(1.0, 2.0, 3.0)
    @test param2uvec([4.0, 5.0]) == SVector(4.0, 5.0)
    @test param2uvec(SVector(6.0, 7.0)) == SVector(6.0, 7.0)

    @test param2xvec(-1.5) == SVector(-1.5)
    @test param2xvec((0.0, 1.0)) isa SVector{2, Float64}
    @test param2xvec([2.0, 3.0]) isa SVector{2, Float64}
    @test param2xvec(SVector(4.0, 5.0)) isa SVector{2, Float64}

    @test param2fvec(2.5)[1][1] == 2.5
    @test param2fvec(([1.0, 2.0], [3.0, 4.0]))[2] == SVector(3.0, 4.0)
    @test param2fvec([[5.0], [6.0], [7.0]])[3][1] == 7.0
    @test length(param2fvec(((1.0, 2.0), (3.0, 4.0)))) == 2
    @test length(param2fvec([8.0, 9.0])[1]) == 2

    @test param2svec((1.5, 2.5))[2][1] == 2.5
    @test param2svec(3.14)[1][1] == 3.14
    @test param2svec(param2fvec(2.5)) == param2fvec(2.5)

    @test param2vel(2.5)[1][1, 1] == 2.5
    @test param2vel((1.0, 2.0)) isa SVector{2, SMatrix{1, 1, Float64, 1}}
    @test param2vel(((1.0, 2.0), (3.0, 4.0)))[2] == SMatrix{2, 2}(3.0, 0.0, 0.0, 4.0)

    @test math_max(10.0, 5.0) == 10.0 && math_min(10.0, 5.0) == 5.0
    @test math_max(-2.0, 3.0) == 3.0  && math_min(-2.0, 3.0) == -2.0
    @test math_max([1.0, 5.0], [2.0, 4.0]) == [2.0, 5.0]
    @test math_min([1.0, 5.0], [2.0, 4.0]) == [1.0, 4.0]
end

@testset "Grid capacity management" begin
    pg = make_pg_1d(20)
    N0 = length(pg.rhos)
    req = N0 + 50
    cap = ceil(Int, req * 1.25)
    HyperCloudCore.ensure_capacity!(pg, req)

    for v in (pg.rhos, pg.curvatures,
              pg.core.positions, pg.core.is_boundary, pg.core.volumes, pg.core.tags, pg.core.particle_orders,
              pg.shared.rho_buffer, pg.shared.pos_buffer, pg.shared.float_buffer, pg.shared.bit_buffer, pg.shared.int_buffer,
              pg.reorder.permutation, pg.reorder.inv_permutation, pg.reorder.new_permutation_buffer, pg.reorder.seen_buffer,
              pg.bins.cell_particles, pg.bins.bin_of)
        @test length(v) == cap
    end

    nb0 = length(pg.neighbor.indices)
    HyperCloudCore.ensure_capacity!(pg.neighbor, nb0 + 100)
    nb_cap = ceil(Int, (nb0 + 100) * 1.25)
    @test length(pg.neighbor.indices) == nb_cap == length(pg.neighbor.weights) == length(pg.neighbor.distances)

    HyperCloudCore.ensure_capacity!(pg, req - 10)        # never shrinks
    @test length(pg.rhos) == cap
end

@testset "Grid construction" begin
    @testset "Mixed periodicity keeps ghost layers on the non-periodic axes" begin
        geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = (true, false))
        pg = ParticleGrid(geom, (0.1, 0.1), ExponentialWeightFunction(1.0, 0.3), 1)
        @test pg.meta.N_ghost > 0
        @test pg.domain.canvas_mins[1] == 0.0 && pg.domain.canvas_maxs[1] == 1.0        # periodic axis: no band
        @test pg.domain.canvas_mins[2] < 0.0 && pg.domain.canvas_maxs[2] > 1.0          # ghost band in y
        @test any(i -> pg.core.is_boundary[i] && pg.core.positions[i][2] < 0, 1:pg.meta.N)
        @test count(i -> pg.core.positions[i][1] ≈ 1.0, 1:pg.meta.N) == 0               # no duplicated seam
    end

    @testset "Jittered periodic grids have no boundary particles at the seam" begin
        for D in (1, 2), seed in 1:3
            geom = get_rectangular_domain(Float64, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D); is_periodic = true)
            pg = ParticleGrid(geom, ntuple(_ -> 0.05, D), ExponentialWeightFunction(1.0, 0.125), 1;
                              randomness = ntuple(_ -> 0.01, D), rng = MersenneTwister(seed))
            N = pg.meta.N
            @test N == 20^D && pg.meta.N_ghost == 0
            @test !any(view(pg.core.is_boundary, 1:N))
            @test all(i -> all(0 .<= pg.core.positions[i] .< 1), 1:N)
        end
    end

    @testset "Small periodic domains do not duplicate neighbours" begin
        geom = get_rectangular_domain(Float64, (0.0,), (1.0,); is_periodic = true)
        for R in (0.45, 0.6)                                   # two and one coarse bins along the axis
            pg = ParticleGrid(geom, (0.1,), ExponentialWeightFunction(1.0, R), 1)
            nb = pg.neighbor
            @test pg.bins.coarse_dims[1] <= 2
            @test all(i -> allunique(view(nb.indices, nb.ranges[i])), 1:pg.meta.N)
            @test all(i -> !(i in view(nb.indices, nb.ranges[i])), 1:pg.meta.N)
            @test all(i -> length(nb.ranges[i]) == count(j -> j != i && (dd = pg.core.positions[j][1] - pg.core.positions[i][1]; abs(dd - round(dd)) <= R + 1e-12), 1:pg.meta.N), 1:pg.meta.N)
        end
    end

    @testset "Narrow ghost band equals the brute-force selection" begin
        geom = get_spherical_domain(Float64, (0.0, 0.0), 1.0)
        dx   = 2 / 40
        pg   = ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(1.0, 2.5dx), 1)
        cd   = pg.domain
        Ns   = ntuple(d -> round(Int, (cd.canvas_maxs[d] - cd.canvas_mins[d]) / dx) + 1, 2)
        pts  = [Space{2, Float64}(cd.canvas_mins[1] + (I[1] - 1) * dx, cd.canvas_mins[2] + (I[2] - 1) * dx) for I in CartesianIndices(Ns)]
        inner, outer = filter(geom.is_interior, pts), filter(!geom.is_interior, pts)
        cutoff_sq = (2.5dx + dx)^2
        expected = count(g -> any(p -> sum(abs2, g - p) <= cutoff_sq, inner), outer)
        @test pg.meta.N_interior == length(inner)
        @test pg.meta.N_ghost == expected
        @test all(i -> !pg.core.is_boundary[i] || any(j -> !pg.core.is_boundary[j] &&
                  sum(abs2, pg.core.positions[i] - pg.core.positions[j]) <= cutoff_sq, 1:pg.meta.N), 1:pg.meta.N)
    end
end

@testset "MLS divergence" begin
    @testset "Exact for a linear flux with one solve for all components" begin
        geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = true)
        dx = 1 / 20
        pg = ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(1.0, 2.5dx), 2)
        nb = pg.neighbor
        df_flux = Vector{Flux{2, 2, Float64}}(undef, length(nb.indices))
        scratch = Vector{State{2, Float64}}(undef, length(nb.indices))
        for i in (1, pg.meta.N ÷ 2)
            sl = nb.ranges[i]
            for k in sl
                d = nb.distances[k]
                df_flux[k] = Flux{2, 2, Float64}((State{2, Float64}(1.5d[1], -0.5d[1] + 2d[2]),
                                                  State{2, Float64}(2.0d[2], 3.0d[2] - d[1])))
            end
            for order in 1:3
                div = HyperCloudCore.Interpolator{2, order, 1}()(sl, nb.distances, nb.weights, df_flux, scratch; scale = pg.meta.dx)
                @test isapprox(div, State{2, Float64}(3.5, 2.5); atol = 1e-8)
            end
        end
    end

    @testset "1-D limiters refuse multi-dimensional grids" begin
        geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = true)
        pg = ParticleGrid(geom, (0.1, 0.1), ExponentialWeightFunction(1.0, 0.25), 1)
        g = zero(SVector{2, State{1, Float64}})
        @test_throws ErrorException HyperCloudCore._limit_slopes(MinmodLimiter(:hard), g, 1:0, State{1, Float64}(0.0),
                                                                 State{1, Float64}[], pg, pg.neighbor.distances, Val(1))
    end
end

@testset "Cached MLS moment matrices" begin
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = true)
    dx = 1 / 24
    pg = ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(1.0, 2.5dx), 2; randomness = (0.2dx, 0.2dx), rng = MersenneTwister(3))
    nb = pg.neighbor
    N = pg.meta.N
    df = [State{2, Float64}(sin(3k), cos(k)) for k in 1:length(nb.indices)]
    df_flux = [Flux{2, 2, Float64}((State{2, Float64}(sin(k), k / 10), State{2, Float64}(cos(2k), -k / 7))) for k in 1:length(nb.indices)]
    scratch = similar(df)

    @testset "Divergence fit: weights sum to zero, odd-order degree, regularisation" begin
        # The divergence fit has a constant term, so a constant flux difference has zero divergence on any
        # stencil (the Taylor fit through the particle does not have this property on irregular stencils).
        const_flux = fill(Flux{2, 2, Float64}((State{2, Float64}(1.0, -2.0), State{2, Float64}(0.5, 3.0))), length(nb.indices))
        for q in 1:3, i in (1, N ÷ 2, N)
            d = HyperCloudCore.Interpolator{2, q, 1}()(nb.ranges[i], nb.distances, nb.weights, const_flux, scratch; scale = pg.meta.dx)
            @test maximum(abs, d) < 1e-9
        end
        # Even fit degrees lose an order on symmetric stencils: odd orders fit the divergence one degree up,
        # if the stencil has 1.5× as many neighbours as unknowns (in 3-D only up to degree 3)
        @test HyperCloudCore._div_idx(Val(2), 1) == 2 && HyperCloudCore._div_idx(Val(2), 2) == 2
        @test HyperCloudCore._div_idx(Val(2), 3) == 4 && HyperCloudCore._div_idx(Val(2), 4) == 4
        @test HyperCloudCore._div_idx(Val(2), 5) == 6 && HyperCloudCore._div_idx(Val(3), 5) == 5
        @test HyperCloudCore._divergence_index(Val(0), Val(1), 3, 4) == 3     # 1-D, 4 neighbours: no bump
        @test HyperCloudCore._divergence_index(Val(0), Val(1), 3, 6) == 4
        @test HyperCloudCore._divergence_index(Val(3), Val(1), 1, 6) == 4     # a fixed MLS order ignores the particle order
        @test HyperCloudCore._divergence_index(Val(0), Val(1), 4, 3) == 3     # too few neighbours for degree 3
        @test HyperCloudCore._mls_reg(Float64) == sqrt(eps(Float64)) && HyperCloudCore._mls_reg(Float32) == sqrt(eps(Float32))
    end

    @testset "Conservation defect diagnostic" begin
        lat = ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(1.0, 2.5dx), 2)      # lattice, no jitter
        for order in (2, 3)
            m = MUSCL(Float64, 2, 2, order, NoLimiter(), RusanovFlux())
            fill!(view(lat.core.particle_orders, 1:lat.meta.N), order)
            r = HyperCloudCore.conservation_defect(m, lat)
            @test r.onesided.pairs == 0 && r.mixed.pairs == 0 && r.equal.pairs > 0 && r.total.pairs == r.equal.pairs
            @test r.total.rel < 1e-10 && maximum(r.particle) < 1e-10          # symmetric stencils: antisymmetric weights
            @test all(i -> r.degree[i] == HyperCloudCore._divergence_index(Val(0), Val(2), order, length(lat.neighbor.ranges[i])) - 1, 1:lat.meta.N)
        end
        m = MUSCL(Float64, 2, 2, 3, NoLimiter(), RusanovFlux())
        fill!(view(pg.core.particle_orders, 1:N), 3)
        r = HyperCloudCore.conservation_defect(m, pg)                        # jittered cloud: a finite pair term
        @test r.mixed.pairs == 0 && 0 < r.total.rel < 1 && length(r.particle) == N && all(x -> 0 <= x < 1, r.particle)
        pg.core.particle_orders[N ÷ 2] = 1                                   # one MOOD-style drop: mixed pairs of order one
        rm = HyperCloudCore.conservation_defect(m, pg)
        # a mixed pair does not cancel at all (its defect exceeds the face magnitude), an equal pair cancels partly
        @test rm.mixed.pairs == length(nb.ranges[N ÷ 2]) && rm.degree[N ÷ 2] == 1
        @test rm.mixed.rel > 1 > rm.equal.rel && rm.mixed.rel > 2 * rm.equal.rel
        fill!(view(pg.core.particle_orders, 1:N), 0)                         # as constructed
    end

    @testset "Conditional regularisation of degenerate stencils" begin
        for T in (Float64, Float32)
            # Healthy stencil: the plain factorisation is used as it is (no Tikhonov term)
            dists = [Space{2, T}(T(0.37) * cos(k), T(0.37) * sin(k) + T(0.1) * k) for k in 1:12]
            wts   = [exp(-sum(abs2, d)) for d in dists]
            Ns = sum(w * (HyperCloudCore.build_basis(Val(2), d) * HyperCloudCore.build_basis(Val(2), d)') for (w, d) in zip(wts, dists))
            C, ok = HyperCloudCore._cholesky_moment(Ns)
            @test ok && C.factors == cholesky(Symmetric(Ns)).factors
            # Collinear neighbours (all on the x axis): the y-moments vanish, so the plain factorisation fails or
            # leaves round-off pivots; the retry gives a finite, damped fit whose x-derivative is still right
            xd = [Space{2, T}(T(0.3) * k, zero(T)) for k in -4:4 if k != 0]
            xw = ones(T, length(xd))
            Nx = sum(w * (HyperCloudCore.build_basis(Val(2), d) * HyperCloudCore.build_basis(Val(2), d)') for (w, d) in zip(xw, xd))
            @test !HyperCloudCore._pivots_ok(cholesky(Symmetric(Nx), check = false), Nx)
            @test HyperCloudCore._cholesky_moment(Nx)[2]
            f = [State{1, T}(2 * d[1] + d[1]^2) for d in xd]              # f = 2x + x²: ∂x f(0) = 2, ∂y f = 0
            g = HyperCloudCore.Interpolator{2, 2, 1}()(1:length(xd), xd, xw, f; scale = one(T))
            @test all(s -> all(isfinite, s), g)
            @test isapprox(g[1][1], 2; atol = 100 * sqrt(eps(T))) && abs(g[2][1]) < 100 * sqrt(eps(T))
        end
    end

    for (order, div_order) in ((3, 0), (4, 2), (2, 0))
        interp = MUSCL(Float64, 2, 2, order, NoLimiter(), RusanovFlux(); div_order = div_order)
        @test !HyperCloudCore.is_current(interp.grad_weights, pg)
        HyperCloudCore.prepare_geometry!(interp, pg)
        @test HyperCloudCore.is_current(interp.grad_weights, pg) && HyperCloudCore.is_current(interp.div_weights, pg)
        grad_deg = max(1, order - 1)
        div_idx  = HyperCloudCore._full_div_idx(Val(order), Val(div_order), Val(2))
        @test div_idx == Dict((3, 0) => 4, (4, 2) => 2, (2, 0) => 2)[(order, div_order)]   # odd orders: one degree up
        @test interp.div_weights !== interp.grad_weights                # the divergence fit has a constant term
        @test HyperCloudCore._has_intercept(interp.div_weights) && !HyperCloudCore._has_intercept(interp.grad_weights)
        for i in (1, N ÷ 3, N)
            sl = nb.ranges[i]
            interp.grad_weights.ok[i] || continue
            direct = HyperCloudCore.Interpolator{2, grad_deg, 1}()(sl, nb.distances, nb.weights, df; scale = dx)
            cached = HyperCloudCore.apply_weights(interp.grad_weights, i, sl, nb.distances, nb.weights, df)
            @test all(isapprox.(direct, cached; atol = 1e-9))
            direct_div = HyperCloudCore.compute_dynamic_divergence(interp.interpolators, div_idx, sl, nb.distances, nb.weights, df_flux, scratch, pg.meta.dx)
            cached_div = HyperCloudCore.apply_weights_divergence(interp.div_weights, i, sl, nb.distances, nb.weights, df_flux)
            @test isapprox(direct_div, cached_div; atol = 1e-9)
        end
        # A geometry change invalidates the cache; the next pass rebuilds it
        refresh_geometry!(pg)
        @test !HyperCloudCore.is_current(interp.grad_weights, pg)
        HyperCloudCore.prepare_geometry!(interp, pg)
        @test HyperCloudCore.is_current(interp.grad_weights, pg)
    end

    # The stepper produces the same solution with and without the cache. The wrapper hides the cache hook,
    # so every particle takes the direct solves.
    struct NoCacheMUSCL{I} <: DivergenceInterpolator; inner::I; end
    HyperCloudCore.update_size!(w::NoCacheMUSCL, N::Int) = update_size!(w.inner, N)
    HyperCloudCore.update_content!(w::NoCacheMUSCL, args...) = update_content!(w.inner, args...)
    (w::NoCacheMUSCL)(args...) = w.inner(args...)
    HyperCloudCore._extract_order(w::NoCacheMUSCL) = _extract_order(w.inner)
    HyperCloudCore._divergence_degree(w::NoCacheMUSCL) = _divergence_degree(w.inner)
    HyperCloudCore._order_reach(w::NoCacheMUSCL, s::MOODStrategy) = _order_reach(w.inner, s)
    HyperCloudCore.has_cached_gradients(w::NoCacheMUSCL) = has_cached_gradients(w.inner)
    HyperCloudCore.cached_gradients(w::NoCacheMUSCL) = HyperCloudCore.cached_gradients(w.inner)

    N1 = 60; eq = make_advection_1d(); dt = 0.4 * (2 / N1)
    pg_a = set_ic!(make_pg_1d(N1), x -> exp(-x^2 / 0.02)); pg_b = set_ic!(make_pg_1d(N1), x -> exp(-x^2 / 0.02))
    m_a = MUSCL(Float64, 1, 1, 3, VenkatakrishnanLimiter(:hard), RusanovFlux())
    m_b = NoCacheMUSCL(MUSCL(Float64, 1, 1, 3, VenkatakrishnanLimiter(:hard), RusanovFlux()))
    ts_a = GeneralRKTimeStepper(eq, m_a, MOOD(EPD1(), MOODu1(1e-6)), RK3_SSP_Tableau(Float64))
    ts_b = GeneralRKTimeStepper(eq, m_b, MOOD(EPD1(), MOODu1(1e-6)), RK3_SSP_Tableau(Float64))
    for k in 0:3; ts_a(eq, pg_a, k * dt, dt); ts_b(eq, pg_b, k * dt, dt); end
    @test HyperCloudCore.is_current(m_a.grad_weights, pg_a) && !HyperCloudCore.is_current(m_b.inner.grad_weights, pg_b)
    @test all(i -> isapprox(pg_a.rhos[i][1], pg_b.rhos[i][1]; atol = 1e-11), 1:pg_a.meta.N)
end

@testset "Execution backends" begin
    for D in 1:3, deg in 0:5
        expected = deg == 0 ? 0 : typeof(HyperCloudCore.basis_length(Val(D), Val(deg))).parameters[1]
        @test HyperCloudCore.basis_count(D, deg) == expected
    end

    N  = 80
    eq = make_advection_1d()
    dt = 0.4 * (2 / N)
    ic = x -> exp(-x^2 / 0.02)
    mover = PrescribedMover((x, t) -> Space{1, Float64}(0.3 * sin(π * x[1])))
    # The config-built sinusoidal mover must be a bits type (a closure over `T` could not enter a GPU kernel).
    sinus = HyperCloudCore.build_mover(Dict{Symbol, Any}(:mover => :sinusoidal, :mover_amplitude => 0.3),
                                       Dict{Symbol, Any}(:T => Float64, :D => 1, :Domain => get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true)))
    @test sinus isa PrescribedMover{<:HyperCloudCore.SinusoidalVelocity{1, Float64}} && isbits(sinus)
    @test sinus.velocity(Space{1, Float64}(-0.5), 0.0) ≈ Space{1, Float64}(0.3 * sin(2π * 0.25))
    cases = [
        ("MUSCL3 VK MOODu2 RK3",  () -> MUSCL(Float64, 1, 1, 3, VenkatakrishnanLimiter(:hard), RusanovFlux()), MOOD(EPD1(), MOODu2(1e-6)), RK3_SSP_Tableau(Float64), NoGridMover()),
        ("MUSCL2 MOODu1 EPD2 RK2", () -> MUSCL(Float64, 1, 1, 2, NoLimiter(), RusanovFlux()), MOOD(EPD2(), MOODu1(1e-6)), RK2_SSP_Tableau(Float64), NoGridMover()),
        ("MUSCL3 BJ soft RK2",     () -> MUSCL(Float64, 1, 1, 3, BarthJespersenLimiter(:soft), RusanovFlux()), MOOD(), RK2_SSP_Tableau(Float64), NoGridMover()),
        ("MUSCL2 Minmod RK2",      () -> MUSCL(Float64, 1, 1, 2, MinmodLimiter(:hard), RusanovFlux()), MOOD(), RK2_SSP_Tableau(Float64), NoGridMover()),
        ("Central2 RK2",           () -> CentralDivergence(Float64, 1, 1, 2), MOOD(), RK2_SSP_Tableau(Float64), NoGridMover()),
        ("Upwind Euler",           () -> UpwindDivergence(Float64, 1, 1, 1, :Classic, UpwindFlux()), MOOD(), RK1_Euler_Tableau(Float64), NoGridMover()),
        ("MUSCL3 moving RK2",      () -> MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux()), MOOD(), RK2_SSP_Tableau(Float64), mover),
        ("MUSCL4 IMEX ARS222",     () -> MUSCL(Float64, 1, 1, 4, NoLimiter(), RusanovFlux()), MOOD(EPD1(), MOODu1(1e-6)), IMEX_ARS222_Tableau(Float64), NoGridMover()),
        ("MUSCL3 MOODu2 streamed",  () -> MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux(); streamed = true), MOOD(EPD1(), MOODu2(1e-6)), RK3_SSP_Tableau(Float64), NoGridMover()),
        ("MUSCL5 aligned streamed", () -> MUSCL(Float64, 1, 1, 5, NoLimiter(), AlignedRusanovFlux(); div_order = 5, streamed = true), MOOD(EPD1(), MOODu2(1e-6)), RK3_SSP_Tableau(Float64), NoGridMover()),
        ("MUSCL4 MOODu2 SSPRK54",  () -> MUSCL(Float64, 1, 1, 4, NoLimiter(), RusanovFlux()), MOOD(EPD1(), MOODu2(1e-6)), RK4_SSP_Tableau(Float64), NoGridMover()),
        ("WENO2 RK2",              () -> WENO(Float64, 1, 1, 2), MOOD(), RK2_SSP_Tableau(Float64), NoGridMover()),
        ("Upwind1 Tiwari RK2",     () -> UpwindDivergence(Float64, 1, 1, 1, :Tiwari, UpwindFlux()), MOOD(), RK2_SSP_Tableau(Float64), NoGridMover()),
        ("MUSCL3 sinusoidal RK2",  () -> MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux()), MOOD(), RK2_SSP_Tableau(Float64), sinus),
    ]
    function run_case(backend, make_interp, mood, tab, mv)
        set_backend!(backend)
        pg = set_ic!(make_pg_1d(N; mover = mv), ic)
        interp = make_interp()
        ts = GeneralRKTimeStepper(eq, interp, mood, tab)
        for k in 0:3; ts(eq, pg, k * dt, dt); end
        return (copy(view(pg.rhos, 1:pg.meta.N)), copy(view(pg.core.positions, 1:pg.meta.N)),
                get_time_step(pg, eq, interp), copy(view(pg.core.particle_orders, 1:pg.meta.N)))
    end
    ka = KABackend(HyperCloudCore.KA.CPU(); workgroupsize = 32)
    try
        @testset "$name" for (name, make_interp, mood, tab, mv) in cases
            ref = run_case(PolyesterBackend(), make_interp, mood, tab, mv)
            thr = run_case(ThreadsBackend(), make_interp, mood, tab, mv)
            @test thr == ref                                          # same arithmetic, bitwise
            dev = run_case(ka, make_interp, mood, tab, mv)            # compiled as device kernels (PoCL)
            @test all(isapprox.(first.(dev[1]), first.(ref[1]); atol = 1e-12))
            @test all(isapprox.(first.(dev[2]), first.(ref[2]); atol = 1e-12))
            @test isapprox(dev[3], ref[3]; rtol = 1e-12)
            @test dev[4] == ref[4]
        end
    finally
        set_backend!(PolyesterBackend())
    end

    @testset "Kernel views and device grids" begin
        pg = make_pg_1d(20)
        kv = HyperCloudCore.Adapt.adapt(Array, pg)                      # what a kernel receives
        @test kv.meta isa HyperCloudCore.GridMetadataView && kv.meta.N == pg.meta.N
        @test kv.geometry === nothing && kv.shared === nothing && kv.reorder === nothing
        @test kv.core.positions === parent(pg.core.positions)            # no copy on the host, wrapper gone
        dg = HyperCloudCore.device_grid(Array, pg)
        @test dg.meta isa HyperCloudCore.GridMetadata && dg.meta.geometry_version == pg.meta.geometry_version
        @test dg.core.positions == pg.core.positions && dg.geometry === pg.geometry
        bt = HyperCloudCore.Adapt.adapt(Array, RK3_SSP_Tableau(Float64))
        @test bt.a_t isa SMatrix && bt.b_t isa SVector && isbits(bt)
    end

    @testset "Growable buffers" begin
        g = Growable{Float64}(Array)
        @test g isa AbstractVector{Float64} && length(g) == 0 && parent(g) isa Vector{Float64}
        HyperCloudCore.ensure_capacity!(g, 8); fill!(g, 2.0)
        n = length(g)
        HyperCloudCore.ensure_capacity!(g, 40)
        @test length(g) >= 40 && all(==(2.0), view(g, 1:n))                # contents survive growth
        g[1] = 3.0
        @test g[1] == 3.0 && parent(g)[1] == 3.0
        @test HyperCloudCore.kernel_view(g) === parent(g)                   # kernels receive the raw array
        @test HyperCloudCore.kernel_view((g, 1)) === (parent(g), 1)

        eq = make_advection_1d()
        interp = MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux())
        ts = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64))
        @test ts.rho_n isa Growable && parent(ts.rho_n) isa Vector{State{1, Float64}}
        @test interp.gradients isa Growable && interp.grad_weights.inv_moment isa Growable
        @test HyperCloudCore.kernel_view(ts.int_buffer).df_flux isa Vector{Flux{1, 1, Float64}}
        @test HyperCloudCore.kernel_view(interp).gradients isa Vector
        pg = set_ic!(make_pg_1d(20), x -> exp(-x^2 / 0.02))
        ts(eq, pg, 0.0, 0.01)
        @test length(ts.rho_n) >= pg.meta.N && mood_flags(ts) isa AbstractVector{Bool} && length(mood_flags(ts)) == pg.meta.N
        HyperCloudCore.check_resident(HyperCloudCore.KA.CPU(), HyperCloudCore.kernel_view((pg, ts.int_buffer, interp, 1.0, Val(2))))
    end
end

@testset "Boundary conditions as kernels" begin
    N  = 60
    dx = 2.0 / N
    make_pg_bc(bc) = ParticleGrid(get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = false,
                                                         bc_map = Dict{Int, AbstractBoundaryCondition}(1 => bc)),
                                  (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1)
    eq     = make_advection_1d()
    interp = MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux())
    ts     = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64))

    @testset "FixedDirichlet restores the initial state" begin
        pg  = set_ic!(make_pg_bc(FixedDirichlet()), x -> x)
        n   = pg.meta.N
        buf = fill(State{1, Float64}(99.0), n)
        HyperCloudCore.apply_boundary_conditions!(pg, buf, ts, eq, 0.0)
        ghosts = findall(view(pg.core.is_boundary, 1:n))
        @test !isempty(ghosts) && all(buf[i] == pg.rhos[i] for i in ghosts)
        @test all(buf[i][1] == 99.0 for i in interior_indices(pg))
    end

    @testset "OutflowBC copies the nearest resolved donor outwards" begin
        pg  = set_ic!(make_pg_bc(OutflowBC()), x -> x)
        n   = pg.meta.N
        x   = i -> pg.core.positions[i][1]
        buf = copy(pg.rhos)
        HyperCloudCore.apply_boundary_conditions!(pg, buf, ts, eq, 0.0)
        inner = interior_indices(pg)
        x_lo, x_hi = minimum(x, inner), maximum(x, inner)
        for i in findall(view(pg.core.is_boundary, 1:n))
            @test buf[i][1] == (x(i) < 0 ? x_lo : x_hi)       # zero gradient: the outermost interior value
        end
        @test all(buf[i] == pg.rhos[i] for i in inner)
        # Orphaned targets (no donor within reach) fall back to their initial state
        pg2 = set_ic!(make_pg_bc(OutflowBC()), x -> x)
        fill!(view(pg2.core.is_boundary, 1:pg2.meta.N), true)   # nobody is a donor ...
        fill!(view(pg2.core.tags, 1:pg2.meta.N), 1)             # ... and everybody is a target of tag 1
        buf2 = fill(State{1, Float64}(99.0), pg2.meta.N)
        HyperCloudCore.apply_boundary_conditions!(pg2, buf2, ts, eq, 0.0)
        @test all(buf2[i] == pg2.rhos[i] for i in 1:pg2.meta.N)
    end

    @testset "Backends agree with boundary conditions" begin
        dt = 0.4dx
        function run(backend, bc)
            set_backend!(backend)
            pg = set_ic!(make_pg_bc(bc), x -> exp(-(x + 0.5)^2 / 0.02))
            ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux()), MOOD(EPD1(), MOODu1(1e-6)), RK3_SSP_Tableau(Float64))
            for k in 0:3; ts(eq, pg, k * dt, dt); end
            return copy(view(pg.rhos, 1:pg.meta.N))
        end
        try
            for bc in (FixedDirichlet(), OutflowBC())
                ref = run(PolyesterBackend(), bc)
                @test run(ThreadsBackend(), bc) == ref
                dev = run(KABackend(HyperCloudCore.KA.CPU(); workgroupsize = 32), bc)
                @test all(isapprox.(first.(dev), first.(ref); atol = 1e-12))
            end
        finally
            set_backend!(PolyesterBackend())
        end
    end
end

@testset "Butcher tableaus" begin
    explicit = (RK1_Euler_Tableau, RK2_SSP_Tableau, RK2_Ralston_Tableau, RK3_SSP_Tableau, RK4_Classical_Tableau, RK4_SSP_Tableau)
    imex     = (IMEX_Euler_Tableau, IMEX_ARS233_Tableau, IMEX_ARS222_Tableau, IMEX_PRSSP3_Tableau, IMEX_SSP2332_Tableau)

    for make in (explicit..., imex...)
        bt = make(Float64)
        s  = HyperCloudCore.num_stages(bt)
        @test bt isa IMEXButcherTableau{Float64}
        @test size(bt.a) == size(bt.a_t) == (s, s)
        @test sum(bt.b_t) ≈ 1.0                                    # explicit part is consistent
        @test all(i -> isapprox(sum(bt.a_t[i, :]), bt.c_t[i]; atol = 1e-14), 1:s)
    end
    for make in explicit
        bt = make(Float64)
        @test all(iszero, bt.a) && all(iszero, bt.b)               # zero implicit part
    end
    # Order conditions of the explicit tableaus (the highest order p ≤ 4 whose conditions hold to round-off)
    function rk_order(bt)
        A, b, c = Matrix(bt.a_t), Vector(bt.b_t), Vector(bt.c_t)
        conds = ([sum(b) - 1], [b' * c - 1/2], [b' * (c .^ 2) - 1/3, b' * (A * c) - 1/6],
                 [b' * (c .^ 3) - 1/4, b' * (c .* (A * c)) - 1/8, b' * (A * (c .^ 2)) - 1/12, b' * (A * (A * c)) - 1/24])
        p = 0
        for (q, res) in enumerate(conds)
            maximum(abs, res) < 1e-13 || break
            p = q
        end
        return p
    end
    @test rk_order(RK1_Euler_Tableau(Float64)) == 1 && rk_order(RK2_SSP_Tableau(Float64)) == 2
    @test rk_order(RK3_SSP_Tableau(Float64)) == 3 && rk_order(RK4_Classical_Tableau(Float64)) == 4
    @test rk_order(RK4_SSP_Tableau(Float64)) == 4 && HyperCloudCore.num_stages(RK4_SSP_Tableau(Float64)) == 5
    # ... and the observed order on y' = -2 t y (y = exp(-t²)) with the explicit tableau applied by hand
    function ode_error(bt, dt)
        A, b, c = Matrix(bt.a_t), Vector(bt.b_t), Vector(bt.c_t); s = length(b)
        f(t, y) = -2t * y
        y = 1.0; t = 0.0
        while t < 1.0 - 1e-12
            k = zeros(s)
            for i in 1:s
                k[i] = f(t + c[i] * dt, y + dt * sum(A[i, j] * k[j] for j in 1:i-1; init = 0.0))
            end
            y += dt * sum(b .* k); t += dt
        end
        return abs(y - exp(-1.0))
    end
    for (make, order) in ((RK3_SSP_Tableau, 3), (RK4_Classical_Tableau, 4), (RK4_SSP_Tableau, 4))
        eoc = log2(ode_error(make(Float64), 0.05) / ode_error(make(Float64), 0.025))
        @test abs(eoc - order) < 0.2
    end
    @test build_tableau(Dict{Symbol, Any}(:stepper => :RK4SSP), Dict{Symbol, Any}(:T => Float64)).b_t ≈ RK4_SSP_Tableau(Float64).b_t
    for make in imex
        @test sum(make(Float64).b) ≈ 1.0
    end
    @test_throws AssertionError IMEXButcherTableau([1.0 1.0; 0.0 1.0], zeros(2, 2), [1.0, 1.0], [0.0, 1.0], [0.5, 0.5], [0.5, 0.5])
    @test_throws AssertionError IMEXButcherTableau(zeros(2, 2), [0.0 1.0; 0.0 0.0], [0.0, 1.0], [0.0, 1.0], [0.5, 0.5], [0.5, 0.5])
end

@testset "MOOD kernel" begin
    pg = make_pg_1d(40)
    N  = pg.meta.N
    @test N == 40                                   # periodic axis: no duplicated seam point
    eq = make_advection_1d()
    interp = MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux())
    orders = pg.core.particle_orders
    nb_r, nb_i = pg.neighbor.ranges, pg.neighbor.indices

    # Every particle on the uniform periodic grid has the same symmetric neighbourhood
    n_nb = length(nb_r[20])
    @test n_nb == 4                                 # range 2.5 dx → ±1, ±2

    @testset "Interface orders" begin
        i  = 10
        j  = i + 2                                   # a neighbour of i
        k  = i - 2                                   # another neighbour, not adjacent to j
        fill!(orders, 3); orders[j] = 1
        @test HyperCloudCore._get_interface_orders(EPD0(),       pg, i, j) == (3, 1)
        @test HyperCloudCore._get_interface_orders(StrictEPD0(), pg, i, j) == (3, 1)
        @test HyperCloudCore._get_interface_orders(EPD1(),       pg, i, j) == (1, 1)
        @test HyperCloudCore._get_interface_orders(EPD1(),       pg, i, k) == (3, 3)
        @test HyperCloudCore._get_interface_orders(EPD2(),       pg, i, k) == (1, 1)   # j ∈ N(i) lowers nbmin(i)
    end

    @testset "Order reach" begin
        @test _order_reach(interp, EPD0()) == 1
        @test _order_reach(interp, EPD1()) == 1
        @test _order_reach(interp, EPD2()) == 2
        @test _order_reach(CentralDivergence(Float64, 1, 1, 2), EPD2()) == 0
        @test _divergence_degree(interp) == 3                       # MUSCL3: the divergence is fitted with degree 3
        @test _divergence_degree(CentralDivergence(Float64, 1, 1, 2)) == 2
    end

    @testset "Mask expansion" begin
        mask, scratch = zeros(Bool, N), zeros(Bool, N)
        mask[7] = true
        HyperCloudCore.expand_mask!(mask, scratch, pg, 1)
        @test count(mask) == 1 + n_nb
        HyperCloudCore.expand_mask!(mask, scratch, pg, 1)
        @test count(mask) == 1 + 2n_nb
    end

    @testset "Forcing halo uses the snapshot (no cascade)" begin
        p = 20
        fill!(orders, 3)
        orders[p] = 1                                          # p dropped to 1 in this pass
        triggered = zeros(Bool, N); triggered[p] = true
        snapshot  = copy(orders)
        changed   = zeros(Bool, N)
        nbs = [nb_i[k] for k in nb_r[p]]
        @test HyperCloudCore.forces_neighbours(StrictEPD0()) && !HyperCloudCore.forces_neighbours(EPD2())
        @test all(HyperCloudCore.forced_order(StrictEPD0(), j, pg, triggered, snapshot) == 1 for j in nbs)
        @test HyperCloudCore.forced_order(StrictEPD0(), p + 3, pg, triggered, snapshot) == 3
        @test HyperCloudCore.forced_order(EPD2(), nbs[1], pg, triggered, snapshot) == 3
        HyperCloudCore.force_neighbours!(StrictEPD0(), pg, triggered, snapshot, orders, changed)
        @test all(orders[j] == 1 for j in nbs)
        @test all(changed[j] for j in nbs)
        @test orders[p + 3] == 3 && !changed[p + 3]           # two hops away: untouched
        @test count(changed) == n_nb && !changed[p]            # p itself is not "changed" by forcing
        # A neighbour that dropped itself (to 2) is forced down to 1 from the snapshot, not from p's new value
        q = nbs[1]
        fill!(orders, 3); orders[p] = 1; orders[q] = 2; fill!(changed, false)
        triggered[q] = true; snapshot = copy(orders)
        HyperCloudCore.force_neighbours!(StrictEPD0(), pg, triggered, snapshot, orders, changed)
        @test orders[q] == 1 && changed[q]
        @test all(orders[j] == 1 for j in nbs)
        # EPD* strategies never force
        fill!(orders, 3); fill!(changed, false)
        HyperCloudCore.force_neighbours!(EPD2(), pg, triggered, snapshot, orders, changed)
        @test all(==(3), orders) && !any(changed)
    end

    @testset "detect_and_drop! terminates at order 1" begin
        ts = GeneralRKTimeStepper(eq, interp, MOOD(EPD1(), OnlyMOOD()), RK3_SSP_Tableau(Float64))
        update_size!(ts, N, length(nb_i), false)
        update_size!(interp, N)
        fill!(orders, 3)
        fill!(ts.work.dirty, false); fill!(ts.work.triggered_step, false)   # resize! leaves them undefined
        f = (full) -> HyperCloudCore.detect_and_drop!(ts.mood, ts.work, interp, pg, pg.rhos, pg.rhos, full)
        @test f(true)  == true  && all(==(2), view(orders, 1:N))
        @test f(false) == false                              # dirty mask empty → nobody checked
        fill!(ts.work.dirty, true)
        @test f(false) == true  && all(==(1), view(orders, 1:N))
        @test f(true)  == false && all(==(1), view(orders, 1:N))   # order-1 particles are never flagged
        @test all(view(ts.work.triggered_step, 1:N))
        # NoMOOD short-circuits
        ts0 = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64))
        update_size!(ts0, N, length(nb_i), false)
        @test HyperCloudCore.detect_and_drop!(ts0.mood, ts0.work, interp, pg, pg.rhos, pg.rhos, true) == false
    end

    @testset "detect_and_drop! flags inadmissible candidates (PAD)" begin
        ts = GeneralRKTimeStepper(eq, interp, MOOD(EPD1(), MOODu1(0.0)), RK3_SSP_Tableau(Float64))
        update_size!(ts, N, length(nb_i), false)
        update_size!(interp, N)
        fill!(ts.work.dirty, false); fill!(ts.work.triggered_step, false)
        q = findfirst(i -> !pg.core.is_boundary[i], 1:N)
        cand = copy(pg.rhos); cand[q] = cand[q] * NaN        # passes the DMP: comparisons with NaN are false
        f = (; kw...) -> (fill!(orders, 3); HyperCloudCore.detect_and_drop!(ts.mood, ts.work, interp, pg, pg.rhos, cand, true; kw...))
        @test f() == false                                   # without an equation: no PAD
        @test f(eq = eq) == true && orders[q] == 2 && count(==(2), view(orders, 1:N)) == 1
        @test is_admissible(eq, pg.rhos[q]) && !is_admissible(eq, cand[q])
    end
end

@testset "Time stepper (static grid)" begin
    N  = 60
    eq = make_advection_1d()
    ic = x -> exp(-x^2 / 0.02)
    dx = 2 / N
    dt = 0.4 * dx

    @testset "Explicit, IMEX and MOOD fixed point run and stay finite" begin
        for (tab, mood) in ((RK3_SSP_Tableau(Float64), MOOD()),
                            (RK3_SSP_Tableau(Float64), MOOD(EPD2(), MOODu2(1e-4 * dx))),
                            (IMEX_PRSSP3_Tableau(Float64), MOOD(StrictEPD0(), MOODu1(1e-4 * dx))))
            pg = set_ic!(make_pg_1d(N), ic)
            interp = MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux())
            ts = GeneralRKTimeStepper(eq, interp, mood, tab)
            mass0 = sum(r[1] for r in pg.rhos)
            for k in 0:4; ts(eq, pg, k * dt, dt); end
            @test all(r -> isfinite(r[1]), pg.rhos)
            @test isapprox(sum(r[1] for r in pg.rhos), mass0; rtol = 1e-2)   # collocation: not exactly conservative
            @test length(mood_flags(ts)) == pg.meta.N
            @test all(o -> 1 <= o <= 3, view(pg.core.particle_orders, 1:pg.meta.N))
        end
    end

    @testset "OnlyMOOD drives every particle to order 1 within one step" begin
        pg = set_ic!(make_pg_1d(N), ic)
        interp = MUSCL(Float64, 1, 1, 4, NoLimiter(), RusanovFlux())
        ts = GeneralRKTimeStepper(eq, interp, MOOD(EPD1(), OnlyMOOD()), RK2_SSP_Tableau(Float64))
        ts(eq, pg, 0.0, dt)
        @test all(==(1), view(pg.core.particle_orders, 1:pg.meta.N))
        @test all(view(mood_flags(ts), 1:pg.meta.N))
    end

    @testset "First-order MUSCL advances the solution and matches the upwind scheme" begin
        pg_m, pg_u = set_ic!(make_pg_1d(N), ic), set_ic!(make_pg_1d(N), ic)
        ts_m = GeneralRKTimeStepper(eq, MUSCL(Float64, 1, 1, 1, NoLimiter(), RusanovFlux()), MOOD(), RK1_Euler_Tableau(Float64))
        ts_u = GeneralRKTimeStepper(eq, UpwindDivergence(Float64, 1, 1, 1, :Classic, RusanovFlux()), MOOD(), RK1_Euler_Tableau(Float64))
        r0 = copy(pg_m.rhos)
        ts_m(eq, pg_m, 0.0, dt); ts_u(eq, pg_u, 0.0, dt)
        @test maximum(abs.(first.(view(pg_m.rhos, 1:pg_m.meta.N) .- view(r0, 1:pg_m.meta.N)))) > 1e-4
        @test all(i -> isapprox(pg_m.rhos[i][1], pg_u.rhos[i][1]; atol = 1e-12), 1:pg_m.meta.N)
    end

    @testset "dt must have the grid's element type" begin
        pg = set_ic!(make_pg_1d(N), ic)
        ts = GeneralRKTimeStepper(eq, CentralDivergence(Float64, 1, 1, 2), MOOD(), RK2_SSP_Tableau(Float64))
        @test_throws MethodError ts(eq, pg, 0.0, Float32(dt))
    end
end

@testset "Threaded loops do not allocate" begin
    # The @batch bodies used to build types from the method's static parameters (and the neighbour search
    # captured a reassigned variable), which allocated per particle and destroyed the thread scaling.
    N  = 4000
    eq = make_advection_1d()
    pg = set_ic!(make_pg_1d(N), x -> exp(-x^2 / 0.02))
    @test pg.meta.geometry_version >= 1
    v0 = pg.meta.geometry_version
    pg.neighbor(pg)
    @test pg.meta.geometry_version == v0 + 1
    refresh_geometry!(pg)
    @test pg.meta.geometry_version == v0 + 2
    @test (@allocated pg.neighbor(pg)) < 16_384

    dt = 0.4 * (2 / N)
    # Allocations per step must not grow with the particle count: every launch costs a constant amount
    # (closure/task set-up), nothing may be allocated per particle. Compared against a grid 10x smaller.
    # The warm-up steps until MOOD has triggered once, so the trigger path (first reached at the third
    # step of this profile) is compiled before the measurement; the minimum over a few steps leaves out
    # the extra launches of a step with a trigger.
    function step_alloc(n)
        pgn = set_ic!(make_pg_1d(n), x -> exp(-x^2 / 0.02))
        interp = MUSCL(Float64, 1, 1, 3, VenkatakrishnanLimiter(:hard), RusanovFlux())
        ts = GeneralRKTimeStepper(eq, interp, MOOD(EPD1(), MOODu1(1e-6)), RK3_SSP_Tableau(Float64))
        h = 0.4 * (2 / n)
        k = 0
        while !any(mood_flags(ts)) && k < 10
            ts(eq, pgn, k * h, h); k += 1
        end
        @test any(mood_flags(ts))
        return minimum(@allocated(ts(eq, pgn, (k + j) * h, h)) for j in 0:2)
    end
    for backend in (PolyesterBackend(), ThreadsBackend())
        set_backend!(backend)
        tasks = backend isa ThreadsBackend                 # `Threads.@threads` spawns a task per thread and launch
        small, large = step_alloc(N ÷ 10), step_alloc(N)
        @test large <= small + 8192
        @test large < (tasks ? 256 * 1024 : 16 * 1024)
        interp = MUSCL(Float64, 1, 1, 3, VenkatakrishnanLimiter(:hard), RusanovFlux())
        get_time_step(pg, eq, interp); update_volumes!(pg, ShepardVolumes())   # compile first
        @test (@allocated get_time_step(pg, eq, interp)) < (tasks ? 32_768 : 8192)
        @test (@allocated update_volumes!(pg, ShepardVolumes())) < (tasks ? 32_768 : 8192)
    end
    set_backend!(PolyesterBackend())

    # Slope limiters run inside the MUSCL pre-gather kernel: static scratch only, nothing allocated per call
    # (the former `MVector` scratch did not compile on device backends).
    let pgl = set_ic!(make_pg_1d(200), x -> exp(-x^2 / 0.02)), i = 100
        nb_slice = pgl.neighbor.ranges[i]
        f_i, f_nb, dists = pgl.rhos[i], neighbour_states(pgl.rhos, pgl), pgl.neighbor.distances
        g = SVector{2, State{1, Float64}}(State{1, Float64}(-3.0), State{1, Float64}(0.5))
        for lim in (BarthJespersenLimiter(:hard), VenkatakrishnanLimiter(:soft), MinmodLimiter(:hard), SuperbeeLimiter(:soft))
            lg = HyperCloudCore._limit_slopes(lim, g, nb_slice, f_i, f_nb, pgl, dists, Val(2))
            @test lg isa SVector{2, State{1, Float64}} && all(isfinite, lg[1]) && lg[1][1] / g[1][1] >= 0
            @test (@allocated HyperCloudCore._limit_slopes(lim, g, nb_slice, f_i, f_nb, pgl, dists, Val(2))) == 0
        end
    end
end

@testset "Aligned dissipation: non-negative first-order coefficients with high-degree weights" begin
    # Linear advection with constant reconstructions (order 1 everywhere) and a fixed degree-5 divergence:
    # the RHS is linear, K_i = Σ_j a_ij (u_j - u_i), and a unit state at neighbour j returns a_ij directly.
    # Sorted by the weight signs (AlignedRusanovFlux) every a_ij is non-negative, so the first-order update
    # is local-extremum-diminishing for any zero-sum weights; sorted by the distance (RusanovFlux) the
    # high-degree weights make some a_ij negative.
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = true)
    dx   = 1 / 24
    pg   = ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(1.0, 5.5dx), 1; randomness = (0.2dx, 0.2dx), rng = MersenneTwister(9))
    N    = pg.meta.N
    eq   = LinearAdvection(((1.0,), (0.7,)), Float64)
    nb   = pg.neighbor
    function coefficients(flux, i)
        m  = MUSCL(Float64, 2, 1, 5, NoLimiter(), flux; div_order = 5)
        ts = GeneralRKTimeStepper(eq, m, MOOD(), RK3_SSP_Tableau(Float64))
        HyperCloudCore.update_size!(ts, N, length(nb.indices), false); HyperCloudCore.update_size!(m, N)
        fill!(view(pg.core.particle_orders, 1:N), 1)
        a = Float64[]
        for k in nb.ranges[i]
            j = nb.indices[k]
            fill!(view(pg.rhos, 1:N), zero(State{1, Float64})); pg.rhos[j] = State{1, Float64}(1.0)
            HyperCloudCore.compute_explicit_rhs!(ts, eq, pg, ts.K_E_stages[1], ts.K_X_stages[1], pg.rhos, 0.0, true)
            push!(a, parent(ts.K_E_stages[1])[i][1])
        end
        return a
    end
    particles = (1, N ÷ 4, N ÷ 2, N)
    a_aligned = vcat((coefficients(AlignedRusanovFlux(), i) for i in particles)...)
    a_geo     = vcat((coefficients(RusanovFlux(), i) for i in particles)...)
    @test length(a_aligned) > 300 && minimum(a_aligned) >= -1e-12
    @test minimum(a_geo) < -1e-6                                       # the geometric sorting is not monotone here
    @test length(MUSCL(Float64, 2, 1, 5, NoLimiter(), AlignedRusanovFlux(); div_order = 5).dweights) == 0   # filled by prepare_geometry!
    @test build_flux(Dict{Symbol, Any}(:name => :AlignedRusanov), Dict{Symbol, Any}()) isa AlignedRusanovFlux
end

@testset "Implicit stage: RHS from the solve" begin
    # One implicit source: K_I of a diagonal stage is (U_out - U_in) / (dt a_ii), which equals S(U_out) for an
    # exact solve. Constant state, so the explicit RHS is zero and the step is the ARS222 solution of u' = -λu.
    struct DecaySource <: AbstractImplicitSourceTerm; λ::Float64; end
    HyperCloudCore.evaluate_source(s::DecaySource, U, i::Int, pg, t::Real) = -s.λ * U
    HyperCloudCore.implicit_solve(s::DecaySource, U_in, γ::Real, i::Int, pg, t::Real) = U_in / (1 + γ * s.λ)
    eq  = make_advection_1d()
    pg  = set_ic!(make_pg_1d(40), x -> 1.0)
    src = DecaySource(3.0)
    ts  = GeneralRKTimeStepper(eq, MUSCL(Float64, 1, 1, 2, NoLimiter(), RusanovFlux()), MOOD(), (), (src,), IMEX_ARS222_Tableau(Float64))
    dt  = 0.1
    ts(eq, pg, 0.0, dt)
    bt  = ts.tableau
    checked = 0
    for i in 1:HyperCloudCore.num_stages(bt)
        abs(bt.a[i, i]) > 1e-14 || continue
        Y = parent(ts.Y_stages[i]); K = parent(ts.K_I_stages[i])
        @test all(j -> isapprox(K[j][1], -src.λ * Y[j][1]; rtol = 1e-12), interior_indices(pg))
        checked += 1
    end
    @test checked >= 1
    @test all(j -> abs(pg.rhos[j][1] - exp(-src.λ * dt)) < 5e-3, interior_indices(pg))     # second-order IMEX step
end

@testset "Streamed divergence" begin
    # The streamed flux pass (cached per-slot weights, dynamic per-slot weights for dropped particles) gives the
    # divergence of the buffered fit up to round-off, with both fluxes, and never touches the slot buffers.
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = true)
    dx   = 1 / 32
    pg   = ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(1.0, 5.5dx), 2; randomness = (0.2dx, 0.2dx), rng = MersenneTwister(21))
    N    = pg.meta.N
    for i in 1:N
        x = pg.core.positions[i]
        pg.rhos[i] = State{2, Float64}(exp(-sum(abs2, x - Space{2, Float64}(0.5, 0.5)) / 0.05), sin(2pi * x[1]) * cos(2pi * x[2]))
    end
    eq = LinearAdvection(SVector(SMatrix{2, 2, Float64}(1.0, 0.0, 0.0, 0.5), SMatrix{2, 2, Float64}(0.7, 0.0, 0.0, -0.3)), Float64)
    function rhs(flux, streamed, drop)
        m  = MUSCL(Float64, 2, 2, 5, NoLimiter(), flux; streamed = streamed)
        ts = GeneralRKTimeStepper(eq, m, MOOD(), RK3_SSP_Tableau(Float64))
        HyperCloudCore.update_size!(ts, N, length(pg.neighbor.indices), false); HyperCloudCore.update_size!(m, N)
        fill!(view(pg.core.particle_orders, 1:N), 5)
        drop && (pg.core.particle_orders[findall(i -> abs(pg.core.positions[i][1] - 0.5) < 0.1, 1:N)] .= 2)
        HyperCloudCore.compute_explicit_rhs!(ts, eq, pg, ts.K_E_stages[1], ts.K_X_stages[1], pg.rhos, 0.0, true)
        return copy(view(parent(ts.K_E_stages[1]), 1:N)), length(ts.int_buffer.df_flux)
    end
    for flux in (RusanovFlux(), AlignedRusanovFlux()), drop in (false, true)
        Kb, nb_b = rhs(flux, false, drop)
        Ks, nb_s = rhs(flux, true, drop)
        scale = maximum(maximum(abs, k) for k in Kb)
        @test maximum(maximum(abs, Kb[i] - Ks[i]) for i in 1:N) < 1e-11 * scale
        @test nb_b >= length(pg.neighbor.indices) && nb_s == 0      # the streamed scheme never sizes the slot buffers
    end
    fill!(view(pg.core.particle_orders, 1:N), 0)
end

@testset "Memory budget" begin
    N  = 400
    eq = make_advection_1d()
    pg = set_ic!(make_pg_1d(N), x -> exp(-x^2 / 0.02))
    interp = MUSCL(Float64, 1, 1, 3, VenkatakrishnanLimiter(:hard), RusanovFlux())
    ts = GeneralRKTimeStepper(eq, interp, MOOD(EPD1(), MOODu1(1e-6)), RK3_SSP_Tableau(Float64))
    h  = 0.4 * (2 / N)
    ts(eq, pg, 0.0, h)                                       # sizes the stepper and interpolator buffers
    S, F = sizeof(State{1, Float64}), sizeof(Flux{1, 1, Float64})

    @testset "estimate" begin
        est = memory_estimate(pg, ts, interp)
        @test est.particles == N && est.slots == sum(view(pg.neighbor.counts, 1:N))   # row-major: the neighbour total
        @test slots_per_particle(pg) == est.slots / N
        @test est.bytes == N * est.per_particle + est.slots * est.per_slot + est.fixed
        @test est.fixed == sizeof(Int) * (length(pg.bins.cell_start) + length(pg.bins.cell_count)) +
                           sizeof(eltype(pg.bins.bin_neighbors)) * length(pg.bins.bin_neighbors) + sizeof(UnitRange{Int})
        g = memory_estimate(pg)
        # Stepper: rho_n, rho_cand, 3 stages × (Y, K_E, K_I) = 11 states, 5 Bool masks, 1 Int snapshot; per slot
        # df_flux, df_scratch, mask. No position buffers on a static grid.
        @test memory_estimate(pg, ts).per_particle - g.per_particle == 11S + 5 + sizeof(Int)
        @test memory_estimate(pg, ts).per_slot - g.per_slot == F + S + 1
        # MUSCL3: gradients SVector{2, State} + the reconstruction cache (SMatrix{2,2} + Bool) + the divergence
        # cache (degree 3 with a constant term: SMatrix{4,4} + Bool)
        @test memory_estimate(pg, interp).per_particle - g.per_particle ==
              2S + sizeof(SMatrix{2, 2, Float64, 4}) + 1 + sizeof(SMatrix{4, 4, Float64, 16}) + 1
        @test memory_estimate(pg, interp).per_slot == g.per_slot                 # the weight signs count only for the aligned flux
        interp_a = MUSCL(Float64, 1, 1, 3, NoLimiter(), AlignedRusanovFlux())
        @test memory_estimate(pg, interp_a).per_slot == g.per_slot + sizeof(Space{1, Float64})
        # A streamed scheme keeps the per-slot weights and the stepper drops its per-slot flux buffers
        interp_s = MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux(); streamed = true)
        ts_s     = GeneralRKTimeStepper(eq, interp_s, MOOD(), RK3_SSP_Tableau(Float64))
        @test memory_estimate(pg, ts_s, interp_s).per_slot - g.per_slot == sizeof(Space{1, Float64})
        @test !uses_slot_buffers(interp_s) && uses_slot_buffers(interp)
        # A moving grid adds x_n, x_cand and 3 stages × (X, K_X): 8 position buffers
        pgm = make_pg_1d(N; mover = LagrangianMover())
        tsm = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64))
        @test memory_estimate(pgm, tsm).per_particle - memory_estimate(pgm).per_particle == 11S + 5 + sizeof(Int) + 8 * sizeof(Space{1, Float64})
        @test memory_estimate(pg, ts, interp; particles = 2N, slots = 3).bytes == 2N * est.per_particle + 3 * est.per_slot + est.fixed
    end

    @testset "reserve!" begin
        est = memory_estimate(pg, ts, interp)
        spp = slots_per_particle(pg)
        r = reserve!(pg, ts, interp; limit = 2 * est.bytes, tolerance = 0.1)
        @test r.limit == 2 * est.bytes
        @test r.particles == floor(Int, (floor(Int, 0.9 * r.limit) - est.fixed) / (est.per_particle + spp * est.per_slot))
        @test r.slots == ceil(Int, r.particles * spp)
        @test r.bytes == memory_estimate(pg, ts, interp; particles = r.particles, slots = r.slots).bytes <= 0.9 * r.limit
        @test length(ts.rho_n) == r.particles && length(interp.gradients) == r.particles
        @test length(pg.neighbor.indices) == r.slots && length(ts.int_buffer.df_flux) == r.slots
        @test length(pg.neighbor.ranges) == r.particles + 1 && all(isempty, view(pg.neighbor.ranges, (N + 2):(r.particles + 1)))
        @test ts.rho_n.fixed && pg.neighbor.indices.fixed && pg.rhos.fixed && !Growable(zeros(2)).fixed
        ts(eq, pg, h, h); ts(eq, pg, 2h, h)                  # steps run inside the reservation
        @test length(ts.rho_n) == r.particles && length(pg.neighbor.indices) == r.slots
        @test_throws CapacityError HyperCloudCore.ensure_capacity!(pg, r.particles + 1)
        @test_throws CapacityError HyperCloudCore.ensure_capacity!(pg.neighbor, r.slots + 1)
        @test_throws CapacityError HyperCloudCore.ensure_capacity!(ts.rho_n, r.particles + 1)
        HyperCloudCore.ensure_capacity!(ts.rho_n, r.particles)                    # within the reservation: no-op
        @test_throws ArgumentError reserve!(pg, ts, interp; limit = est.bytes ÷ 2)
        @test_throws ArgumentError reserve!(pg, ts, interp; limit = 2 * est.bytes, slots_per_particle = 0)   # lists do not fit
        @test_throws ArgumentError reserve!(pg, ts, interp; limit = 2 * est.bytes, tolerance = 1.0)
        @test_throws ArgumentError reserve!(pg, ts, interp; limit = est.fixed)
        gr = Growable(zeros(2))
        HyperCloudCore.ensure_capacity!(gr, 3)                                    # an unreserved buffer grows
        @test length(gr) == ceil(Int, 3 * 1.25)
    end

    @testset "exact fit and outgrowing the budget" begin
        pg2 = set_ic!(make_pg_1d(N), x -> 0.0)
        est = memory_estimate(pg2)
        r = reserve!(pg2; limit = est.bytes + 1, tolerance = 0)
        @test r.particles == N && r.bytes == est.bytes
        pg2.neighbor(pg2)                                    # a search with the same lists fits
        @test length(pg2.neighbor.indices) == r.slots
        pos = pg2.core.positions
        for i in 1:N; pos[i] = pos[i] / 2; end               # twice as dense: more neighbours than reserved
        @test_throws CapacityError pg2.neighbor(pg2)
    end
end

@testset "Moving grids" begin
    N  = 60
    c  = 0.7
    eq = make_advection_1d(c)
    ic = x -> 0.5 + 0.3 * sin(π * x)
    dx = 2 / N

    @testset "grid_velocity" begin
        pg = make_pg_1d(N)
        u  = State{1, Float64}(1.0)
        @test grid_velocity(NoGridMover(), eq, 1, u, pg, 0.0) == zero(Space{1, Float64})
        @test grid_velocity(LagrangianMover(), eq, 1, u, pg, 0.0) == Space{1, Float64}(c)
        @test grid_velocity(ScaledLagrangianMover(0.5), eq, 1, u, pg, 0.0) == Space{1, Float64}(0.5c)
        pm = PrescribedMover((x, t) -> Space{1, Float64}(2.0 * x[1] + t))
        @test grid_velocity(pm, eq, 3, u, pg, 1.0) == Space{1, Float64}(2.0 * pg.core.positions[3][1] + 1.0)
    end

    @testset "refresh_geometry! matches the full search and respects periodicity" begin
        pg = make_pg_1d(N)
        Np = pg.meta.N
        n_tot = last(pg.neighbor.ranges[Np])                  # buffers are over-allocated; compare active slots only
        d0, w0 = pg.neighbor.distances[1:n_tot], pg.neighbor.weights[1:n_tot]
        refresh_geometry!(pg)
        @test pg.neighbor.distances[1:n_tot] == d0 && pg.neighbor.weights[1:n_tot] == w0

        shift = Space{1, Float64}(0.37 * 2.0)                 # rigid translation, leaves the canvas
        for i in 1:Np; pg.core.positions[i] += shift; end
        refresh_geometry!(pg)
        @test all(isapprox.(pg.neighbor.distances[1:n_tot], d0; atol = 1e-12))   # minimal-image distances unchanged
        @test all(isapprox.(pg.neighbor.weights[1:n_tot], w0; atol = 1e-12))

        # Pairs that drift beyond R lose their weight
        pg.core.positions[1] += Space{1, Float64}(0.5)
        refresh_geometry!(pg)
        @test all(k -> pg.neighbor.weights[k] == 0.0, pg.neighbor.ranges[1])
    end

    @testset "ALEFrame: relative flux, shifted spectrum" begin
        pg = make_pg_1d(N)
        u  = State{1, Float64}(0.8)
        ale = ALEFrame(eq, Space{1, Float64}(c))                   # frame moving with the flow
        @test flux(ale, u)[1][1] == 0.0
        @test max_eigenvalue(ale, u, 1) == 0.0
        @test velocity(ale, u, 1)[1, 1] == 0.0
        ale2 = ALEFrame(eq, Space{1, Float64}(-c))
        @test flux(ale2, u)[1][1] ≈ 2c * 0.8
        @test max_eigenvalue(ale2, u, 1) ≈ 2c
        @test prim2cons(ale, u) == u && cons2prim(ale, u) == u
    end

    @testset "Lagrangian transport is exact for every scheme" begin
        # w = u freezes the relative flux F − w_i ρ to zero, so ρ is carried unchanged by the particles
        schemes = (CentralDivergence(Float64, 1, 1, 1),
                   MUSCL(Float64, 1, 1, 2, MinmodLimiter(:hard), RusanovFlux()),
                   MUSCL(Float64, 1, 1, 4, NoLimiter(), RusanovFlux()),
                   WENO(Float64, 1, 1, 2),
                   UpwindDivergence(Float64, 1, 1, 1, :Classic, RusanovFlux()))
        for interp in schemes
            pg = set_ic!(make_pg_1d(N; mover = LagrangianMover()), ic)
            Np = pg.meta.N
            x0 = copy(view(pg.core.positions, 1:Np))
            r0 = copy(view(pg.rhos, 1:Np))
            ts = GeneralRKTimeStepper(eq, interp, MOOD(), RK2_SSP_Tableau(Float64))
            dt = 0.4 * dx / c
            n_steps = 12
            for k in 0:n_steps-1; ts(eq, pg, k * dt, dt); end
            t = n_steps * dt
            wrap(x) = -1.0 + mod(x + 1.0, 2.0)
            @test all(i -> isapprox(pg.rhos[i][1], r0[i][1]; atol = 1e-10), 1:Np)
            @test all(i -> isapprox(pg.core.positions[i][1], wrap(x0[i][1] + c * t); atol = 1e-10), 1:Np)
        end
    end

    @testset "Prescribed motion keeps a 2nd-order MUSCL scheme accurate and conservative" begin
        mover = PrescribedMover((x, t) -> Space{1, Float64}(0.3 * sin(π * x[1])))
        errs = Float64[]
        for n in (60, 120)
            pg = set_ic!(make_pg_1d(n; mover = mover), ic)
            interp = MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux())
            ts = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64))
            dt = 0.3 * (2 / n) / (c + 0.3)
            t  = 0.0
            while t < 0.5 - 1e-12
                h = min(dt, 0.5 - t); ts(eq, pg, t, h); t += h
            end
            Np = pg.meta.N
            exact = i -> ic(pg.core.positions[i][1] - c * t)   # periodic, ic is 2-periodic on [-1,1]
            push!(errs, sqrt(sum((pg.rhos[i][1] - exact(i))^2 for i in 1:Np) / Np))
            @test all(r -> isfinite(r[1]), pg.rhos)
        end
        eoc = log2(errs[1] / errs[2])
        @test eoc > 1.5
    end
end

@testset "Particle volumes" begin
    @testset "Uniform periodic cloud: every V equals dx" begin
        pg = make_pg_1d(50)
        N  = pg.meta.N
        @test all(i -> isapprox(pg.core.volumes[i], 2 / 50; rtol = 1e-12), 1:N)
        @test sum(view(pg.core.volumes, 1:N)) ≈ domain_volume(pg)
        update_volumes!(pg, MLSVolumes(2))
        @test all(i -> isapprox(pg.core.volumes[i], 2 / 50; rtol = 1e-10), 1:N)
        update_volumes!(pg, ShepardVolumes(sweeps = 3))
        @test sum(view(pg.core.volumes, 1:N)) ≈ domain_volume(pg)
    end

    @testset "Quadrature order on a jittered cloud" begin
        geom = get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = false)
        dx   = 2 / 80
        pg   = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1;
                            randomness = (0.3dx,), rng = MersenneTwister(1))
        N    = pg.meta.N
        x    = i -> pg.core.positions[i][1]
        quad = f -> sum(pg.core.volumes[i] * f(x(i)) for i in 1:N)

        @test sum(view(pg.core.volumes, 1:N)) ≈ 2.0                 # Shepard: Σ V = |Ω| exactly
        err_shepard = abs(quad(x -> x^2) - 2 / 3)

        update_volumes!(pg, MLSVolumes(2; samples_per_dx = 4))        # reproduces quadratics
        @test sum(view(pg.core.volumes, 1:N)) ≈ 2.0
        @test all(v -> v >= 0, view(pg.core.volumes, 1:N))
        # Exterior (ghost) particles can get negative MLS volumes; they are clamped to zero before
        # the renormalisation, which costs some of the polynomial exactness near a non-periodic
        # boundary (the clamped mass is O(dx) per ghost). Exactness is tested on the periodic cloud below.
        err_mls = abs(quad(x -> x^2) - 2 / 3)
        @test err_mls < 5e-3
        @test err_mls < err_shepard
        @test abs(quad(x -> x) - 0.0) < 5e-3
    end

    @testset "Quadrature order on a jittered periodic cloud (no ghosts)" begin
        geom = get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true)
        dx   = 2 / 80
        pg   = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1;
                            randomness = (0.3dx,), rng = MersenneTwister(2))
        N    = pg.meta.N
        x    = i -> pg.core.positions[i][1]
        f    = x -> 1 + sin(π * x)                                   # ∫_{-1}^{1} f = 2
        quad = () -> sum(pg.core.volumes[i] * f(x(i)) for i in 1:N)

        err_shepard = abs(quad() - 2)
        update_volumes!(pg, MLSVolumes(2; samples_per_dx = 4))
        @test all(v -> v > 0, view(pg.core.volumes, 1:N))
        err_mls2 = abs(quad() - 2)
        update_volumes!(pg, MLSVolumes(3; samples_per_dx = 4))
        err_mls3 = abs(quad() - 2)
        @test err_mls2 < 1e-5
        @test err_mls2 < err_shepard
        @test err_mls3 <= 2 * err_mls2                                # degree 3 is not worse
    end

    @testset "Singular kernels fall back to uniform volumes" begin
        geom = get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true)
        pg = @test_logs (:warn, r"bounded at zero") ParticleGrid(geom, (0.05,), InverseWeightFunction(1.0, 0.125), 1)
        @test all(i -> pg.core.volumes[i] ≈ 2 / pg.meta.N, 1:pg.meta.N)
    end

    @testset "total_mass" begin
        pg = set_ic!(make_pg_1d(40), x -> 2.0)
        @test total_mass(pg)[1] ≈ 4.0
    end
end

@testset "Particle adaptivity" begin
    ic   = x -> 1.0 + 0.5 * sin(π * x)
    kmax = 3
    ad   = ParticleAdaptivity(Float64)

    # Remove particle `p` through the remap machinery (exactly conservative) and return the new mass
    function punch_hole!(pg, p)
        N0 = pg.meta.N
        ev = HyperCloudCore._AdaptEvent{1, Float64}(pg.core.positions[p], [p], Space{1, Float64}[])
        np, nr, nv = Space{1, Float64}[], State{1, Float64}[], Float64[]
        HyperCloudCore._remap_event!(pg, ad, ev, np, nr, nv)
        mask = pg.shared.bit_buffer; fill!(view(mask, 1:N0), false); mask[p] = true
        HyperCloudCore._apply_topology!(pg, mask, np, nr, nv, kmax)
        pg.neighbor(pg)
        return nothing
    end

    @testset "Removal + re-insertion are exactly conservative and bounded" begin
        pg = set_ic!(make_pg_1d(40), ic)
        N0 = pg.meta.N
        m0 = total_mass(pg)
        x_hole = pg.core.positions[10][1]

        punch_hole!(pg, 10)
        @test pg.meta.N == N0 - 1
        @test maximum(abs.(total_mass(pg) - m0)) < 1e-12
        @test sum(view(pg.core.volumes, 1:pg.meta.N)) ≈ domain_volume(pg)

        rep = adapt_particles!(pg, ad, kmax)
        @test rep.inserted == 1 && rep.removed == 0
        @test pg.meta.N == N0
        @test maximum(abs.(rep.mass_residual)) < 1e-12
        @test maximum(abs.(total_mass(pg) - m0)) < 1e-12
        @test all(r -> 0.5 - 1e-12 <= r[1] <= 1.5 + 1e-12, view(pg.rhos, 1:pg.meta.N))
        @test pg.core.particle_orders[pg.meta.N] == kmax                # new particle starts at max order

        # The new particle sits in the hole and carries a second-order accurate value
        x_new = pg.core.positions[pg.meta.N][1]
        @test abs(x_new - x_hole) < 0.3 * (2 / 40)
        @test abs(pg.rhos[pg.meta.N][1] - ic(x_new)) < 0.05

        rep2 = adapt_particles!(pg, ad, kmax)                             # nothing left to do
        @test rep2.inserted == 0 && rep2.removed == 0
    end

    @testset "Merge of a close pair" begin
        pg = set_ic!(make_pg_1d(40), ic)
        N0 = pg.meta.N
        pg.core.positions[21] = pg.core.positions[20] + Space{1, Float64}(0.2 * 2 / 40)
        pg.neighbor(pg)
        update_volumes!(pg, ShepardVolumes())
        m0 = total_mass(pg)
        rep = adapt_particles!(pg, ad, kmax)
        @test rep.removed == 2 && rep.inserted == 1
        @test pg.meta.N == N0 - 1
        @test maximum(abs.(rep.mass_residual)) < 1e-12
        @test all(r -> 0.5 - 1e-12 <= r[1] <= 1.5 + 1e-12, view(pg.rhos, 1:pg.meta.N))
    end

    @testset "No events inside the shock band" begin
        pg = set_ic!(make_pg_1d(40), ic)
        punch_hole!(pg, 10)
        pg.core.particle_orders[12] = 1                                   # pretend MOOD lowered a neighbour
        rep = adapt_particles!(pg, ad, kmax)
        @test rep.inserted == 0 && rep.removed == 0
        rep = adapt_particles!(pg, ParticleAdaptivity(Float64; exclude_reduced_order = false), kmax)
        @test rep.inserted == 1
    end

    @testset "DMP repair is conservative" begin
        U  = [State{1, Float64}(1.3), State{1, Float64}(0.4), State{1, Float64}(0.9)]
        V  = [1.0, 1.0, 2.0]
        lo = fill(State{1, Float64}(0.5), 3)
        hi = fill(State{1, Float64}(1.0), 3)
        m0 = sum(V .* getindex.(U, 1))
        HyperCloudCore._dmp_repair!(U, V, lo, hi)
        @test all(u -> 0.5 - 1e-12 <= u[1] <= 1.0 + 1e-12, U)
        @test sum(V .* getindex.(U, 1)) ≈ m0
    end

    # 2-D cloud with close pairs about 4.2 R apart: the old 4R spacing accepted them in one call although
    # the local set of each remap (radius ≈ 4.2 R in 2-D) reached the particle the other merge removes.
    function clustered_cloud_2d(; spacing_R = 4.2, n = 40)
        dx   = 2 / n
        geom = get_rectangular_domain(Float64, (-1.0, -1.0), (1.0, 1.0); is_periodic = true)
        pg   = ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(1.0, 2.5dx), 1;
                            randomness = (0.1dx, 0.1dx), rng = MersenneTwister(3))
        R, N = pg.meta.R, pg.meta.N
        L, Li = pg.domain.L_wrap, pg.domain.invL_wrap
        s  = spacing_R * R
        nc = floor(Int, 2 / s)
        for a in 1:nc, b in 1:nc
            c = Space{2, Float64}(-1 + (a - 0.5) * s, -1 + (b - 0.5) * s)
            p = argmin(i -> sum(abs2, HyperCloudCore._wrapped_diff(pg.core.positions[i], c, L, Li)), 1:N)
            q = argmin(i -> i == p ? Inf : sum(abs2, HyperCloudCore._wrapped_diff(pg.core.positions[i], pg.core.positions[p], L, Li)), 1:N)
            pg.core.positions[p] = HyperCloudCore._wrap_point(pg, pg.core.positions[q] + Space{2, Float64}(0.2dx, 0.0))
        end
        pg.neighbor(pg)
        update_volumes!(pg, ShepardVolumes())
        for i in 1:N
            x = pg.core.positions[i]
            pg.rhos[i] = State{1, Float64}(1 + 0.5 * sin(π * x[1]) * cos(π * x[2]))
        end
        return pg, nc^2
    end

    @testset "Simultaneous events in 2-D are independent (exactly conservative)" begin
        pg, n_pairs = clustered_cloud_2d()
        ad2 = ParticleAdaptivity(Float64; exclude_reduced_order = false)
        m0  = total_mass(pg)[1]
        evs = HyperCloudCore._detect_events(pg, ad2, kmax)
        @test length(evs) > 1                                              # several events per call
        spacing = HyperCloudCore._event_spacing(pg)
        L, Li = pg.domain.L_wrap, pg.domain.invL_wrap
        @test all(norm(HyperCloudCore._wrapped_diff(e1.center, e2.center, L, Li)) > spacing - pg.meta.R
                  for e1 in evs, e2 in evs if e1 !== e2)
        worst, merged, calls = 0.0, 0, 0
        while calls < 20
            rep = adapt_particles!(pg, ad2, kmax)
            rep.inserted + rep.removed == 0 && break
            worst   = max(worst, abs(rep.mass_residual[1]))
            merged += rep.removed
            calls  += 1
        end
        @test merged >= 2n_pairs                                           # every close pair was merged
        @test worst < 1e-13 * m0
        @test abs(total_mass(pg)[1] - m0) < 1e-12 * m0
        @test all(r -> 0.5 - 1e-12 <= r[1] <= 1.5 + 1e-12, view(pg.rhos, 1:pg.meta.N))
    end

    @testset "max_events" begin
        pg, _ = clustered_cloud_2d()
        @test length(HyperCloudCore._detect_events(pg, ParticleAdaptivity(Float64; exclude_reduced_order = false, max_events = 1), kmax)) == 1
        @test_throws AssertionError ParticleAdaptivity(Float64; max_events = 0)
    end

    @testset "Re-estimated volumes: the call is exact for the stepper's volumes ($(nameof(typeof(est))))" for est in (ShepardVolumes(), MLSVolumes(2))
        function merged_cloud()
            dx = 2 / 40
            pg = make_pg_1d(40)
            pg.core.positions[21] = pg.core.positions[20] + Space{1, Float64}(0.2dx)
            pg.neighbor(pg)
            update_volumes!(pg, est)
            return set_ic!(pg, ic)
        end
        # Without the estimator the remap's volumes stay, and the next re-estimation changes the mass
        pg = merged_cloud(); m0 = total_mass(pg)[1]
        rep = adapt_particles!(pg, ad, kmax)
        @test rep.removed == 2 && abs(rep.mass_residual[1]) < 1e-12
        U_remap = [pg.rhos[i][1] for i in 1:pg.meta.N]
        update_volumes!(pg, est)
        defect_plain = abs(total_mass(pg)[1] - m0)
        @test defect_plain > 1e-8
        # With it the call ends on the estimator's volumes: exact, and the next re-estimation changes nothing
        pg = merged_cloud(); m0 = total_mass(pg)[1]
        ad_v = ParticleAdaptivity(Float64; volume_estimator = est)
        rep = adapt_particles!(pg, ad_v, kmax)
        @test rep.removed == 2 && abs(rep.mass_residual[1]) < 1e-12
        @test abs(total_mass(pg)[1] - m0) < 1e-12
        V = copy(view(pg.core.volumes, 1:pg.meta.N))
        update_volumes!(pg, est)
        @test view(pg.core.volumes, 1:pg.meta.N) ≈ V rtol = 1e-12
        @test abs(total_mass(pg)[1] - m0) < 1e-12
        @test all(r -> 0.5 - 1e-12 <= r[1] <= 1.5 + 1e-12, view(pg.rhos, 1:pg.meta.N))
        # The states stay the remap's up to a correction of the size of the volume defect (spread over the remap's
        # region, whose volume is O(1) here): measured 3.2e-4 for a defect of 2.7e-4. Keeping each particle's
        # content instead changed states by up to 18 %.
        δU = maximum(abs, [pg.rhos[i][1] for i in 1:pg.meta.N] .- U_remap)
        @test δU < 5 * defect_plain
    end

    @testset "Builders: the grid's volume estimator reaches the stepper and the adaptivity" begin
        eq  = make_advection_1d()
        ctx = Dict{Symbol, Any}(:T => Float64, :D => 1, :M => 1, :Equation => eq,
                                :ExplicitSources => (), :ImplicitSources => ())
        ctx[:Domain]     = build_domain(Dict{Symbol, Any}(:domain => :rectangular, :mins => (-1.0,), :maxs => (1.0,), :periodic => true), ctx)
        ctx[:WeightConf] = Dict{Symbol, Any}(:name => :exponential, :range => 2.5, :alpha => 1.0)
        pg = build_particle_grid(Dict{Symbol, Any}(:Ns => (40,), :mover => :sinusoidal, :mover_amplitude => 0.5,
                                                   :volumes => :mls, :volumes_degree => 2), ctx)
        @test ctx[:VolumeEstimator] isa MLSVolumes{2}
        ctx[:Flux]    = build_flux(Dict{Symbol, Any}(:name => :Rusanov), ctx)
        ctx[:Limiter] = build_limiter(Dict{Symbol, Any}(:name => :none), ctx)
        ctx[:MOOD]    = build_mood(Dict{Symbol, Any}(:criterion => :none), ctx)
        ctx[:Scheme]  = build_scheme(Dict{Symbol, Any}(:name => :MUSCL, :order => 2), ctx)
        ctx[:Tableau] = build_tableau(Dict{Symbol, Any}(:stepper => :RK2), ctx)
        @test build_timestepper(ctx).volume_estimator isa MLSVolumes{2}

        ad_b = build_adaptivity(Dict{Symbol, Any}(:enabled => true, :max_events => 3), ctx)
        @test ad_b.max_events == 3
        @test ad_b.volume_estimator isa MLSVolumes{2}                     # moving grid: the stepper's estimator
        @test build_adaptivity(Dict{Symbol, Any}(:enabled => true), ctx).max_events == typemax(Int)
        @test build_adaptivity(Dict{Symbol, Any}(), ctx) === nothing
        ctx_static = merge(ctx, Dict{Symbol, Any}(:Grid => make_pg_1d(20)))
        @test build_adaptivity(Dict{Symbol, Any}(:enabled => true), ctx_static).volume_estimator === nothing
    end

    @testset "MLS volumes through a host copy (device-grid path)" begin
        geom = get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true)
        dx   = 2 / 60
        pg   = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1;
                            randomness = (0.3dx,), rng = MersenneTwister(5))
        update_volumes!(pg, MLSVolumes(2))
        V = copy(view(pg.core.volumes, 1:pg.meta.N))
        update_volumes!(pg, ShepardVolumes())
        HyperCloudCore._update_volumes_on_host!(pg, MLSVolumes(2))
        @test view(pg.core.volumes, 1:pg.meta.N) ≈ V rtol = 1e-13
    end
end
