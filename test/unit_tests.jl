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
              pg.bins.next)
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

    for (order, div_order) in ((3, 0), (4, 2), (2, 0))
        interp = MUSCL(Float64, 2, 2, order, NoLimiter(), RusanovFlux(); div_order = div_order)
        @test !HyperCloudCore.is_current(interp.grad_weights, pg)
        HyperCloudCore.prepare_geometry!(interp, pg)
        @test HyperCloudCore.is_current(interp.grad_weights, pg) && HyperCloudCore.is_current(interp.div_weights, pg)
        grad_deg = max(1, order - 1)
        div_idx  = div_order == 0 ? max(2, order) : max(2, div_order)
        @test (interp.div_weights === interp.grad_weights) == (div_idx - 1 == grad_deg)
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
    cases = [
        ("MUSCL3 VK MOODu2 RK3",  () -> MUSCL(Float64, 1, 1, 3, VenkatakrishnanLimiter(:hard), RusanovFlux()), MOOD(EPD1(), MOODu2(1e-6)), RK3_SSP_Tableau(Float64), NoGridMover()),
        ("MUSCL2 MOODu1 EPD2 RK2", () -> MUSCL(Float64, 1, 1, 2, NoLimiter(), RusanovFlux()), MOOD(EPD2(), MOODu1(1e-6)), RK2_SSP_Tableau(Float64), NoGridMover()),
        ("Central2 RK2",           () -> CentralDivergence(Float64, 1, 1, 2), MOOD(), RK2_SSP_Tableau(Float64), NoGridMover()),
        ("Upwind Euler",           () -> UpwindDivergence(Float64, 1, 1, 1, :Classic, UpwindFlux()), MOOD(), RK1_Euler_Tableau(Float64), NoGridMover()),
        ("MUSCL3 moving RK2",      () -> MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux()), MOOD(), RK2_SSP_Tableau(Float64), mover),
        ("MUSCL4 IMEX ARS222",     () -> MUSCL(Float64, 1, 1, 4, NoLimiter(), RusanovFlux()), MOOD(EPD1(), MOODu1(1e-6)), IMEX_ARS222_Tableau(Float64), NoGridMover()),
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
        @test kv.core.positions === pg.core.positions                    # no copy on the host
        dg = HyperCloudCore.device_grid(Array, pg)
        @test dg.meta isa HyperCloudCore.GridMetadata && dg.meta.geometry_version == pg.meta.geometry_version
        @test dg.core.positions == pg.core.positions && dg.geometry === pg.geometry
        bt = HyperCloudCore.Adapt.adapt(Array, RK3_SSP_Tableau(Float64))
        @test bt.a_t isa SMatrix && bt.b_t isa SVector && isbits(bt)
    end
end

@testset "Butcher tableaus" begin
    explicit = (RK1_Euler_Tableau, RK2_SSP_Tableau, RK2_Ralston_Tableau, RK3_SSP_Tableau, RK4_Classical_Tableau)
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
        @test _divergence_degree(interp) == 2
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
        snapshot = copy(orders); snapshot[p] = 1
        changed  = zeros(Bool, N)
        HyperCloudCore.trigger_halo!(StrictEPD0(), p, pg, orders, snapshot, changed)
        nbs = [nb_i[k] for k in nb_r[p]]
        @test all(orders[j] == 1 for j in nbs)
        @test all(changed[j] for j in nbs)
        @test orders[p + 3] == 3 && !changed[p + 3]           # two hops away: untouched
        @test count(changed) == n_nb
        # EPD* strategies never force
        fill!(orders, 3); fill!(changed, false)
        HyperCloudCore.trigger_halo!(EPD2(), p, pg, orders, snapshot, changed)
        @test all(==(3), orders) && !any(changed)
    end

    @testset "detect_and_drop! terminates at order 1" begin
        ts = GeneralRKTimeStepper(eq, interp, MOOD(EPD1(), OnlyMOOD()), RK3_SSP_Tableau(Float64))
        update_size!(ts, N, length(nb_i), false)
        update_size!(interp, N)
        fill!(orders, 3)
        fill!(ts.work.dirty, false); fill!(ts.work.triggered_step, false)   # resize! leaves them undefined
        f = (full) -> HyperCloudCore.detect_and_drop!(ts.mood, ts.work, interp, pg, ts.int_buffer, pg.rhos, pg.rhos, full)
        @test f(true)  == true  && all(==(2), view(orders, 1:N))
        @test f(false) == false                              # dirty mask empty → nobody checked
        fill!(ts.work.dirty, true)
        @test f(false) == true  && all(==(1), view(orders, 1:N))
        @test f(true)  == false && all(==(1), view(orders, 1:N))   # order-1 particles are never flagged
        @test all(view(ts.work.triggered_step, 1:N))
        # NoMOOD short-circuits
        ts0 = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64))
        update_size!(ts0, N, length(nb_i), false)
        @test HyperCloudCore.detect_and_drop!(ts0.mood, ts0.work, interp, pg, ts0.int_buffer, pg.rhos, pg.rhos, true) == false
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
            @test length(mood_flags(ts)) >= pg.meta.N
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
    function step_alloc(n)
        pgn = set_ic!(make_pg_1d(n), x -> exp(-x^2 / 0.02))
        interp = MUSCL(Float64, 1, 1, 3, VenkatakrishnanLimiter(:hard), RusanovFlux())
        ts = GeneralRKTimeStepper(eq, interp, MOOD(EPD1(), MOODu1(1e-6)), RK3_SSP_Tableau(Float64))
        h = 0.4 * (2 / n)
        ts(eq, pgn, 0.0, h); ts(eq, pgn, h, h)
        return @allocated ts(eq, pgn, 2h, h)
    end
    for use_threads in (false, true)
        set_threads!(use_threads)
        small, large = step_alloc(N ÷ 10), step_alloc(N)
        @test large <= small + 8192
        @test large < (use_threads ? 512 * 1024 : 128 * 1024)
        interp = MUSCL(Float64, 1, 1, 3, VenkatakrishnanLimiter(:hard), RusanovFlux())
        get_time_step(pg, eq, interp); update_volumes!(pg, ShepardVolumes())   # compile first
        @test (@allocated get_time_step(pg, eq, interp)) < (use_threads ? 32_768 : 8192)
        @test (@allocated update_volumes!(pg, ShepardVolumes())) < (use_threads ? 32_768 : 8192)
    end
    set_threads!(false)
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
end
