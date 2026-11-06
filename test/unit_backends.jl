# =========================================================================
# EXECUTION BACKENDS (groups `backends-a`, `backends-b`): the same steps on Polyester, Threads and the
# KernelAbstractions CPU backend (PoCL), kernel views, growable buffers. The parity cases are split over the
# two groups (`_backend_cases`), the remaining checks run in `backends-b`.
# =========================================================================

@testset "Execution backends" begin
    if TEST_GROUP != "backends-a"
        for D in 1:3, deg in 0:5
            expected = deg == 0 ? 0 : typeof(HyperCloudCore.basis_length(Val(D), Val(deg))).parameters[1]
            @test HyperCloudCore.basis_count(D, deg) == expected
        end
    end

    N  = 80
    eq = make_advection_1d()
    dt = 0.4 * (2 / N)
    ic = x -> exp(-x^2 / 0.02)
    mover = PrescribedMover((x, t) -> Space{1, Float64}(0.3 * sin(π * x[1])))
    # The config-built sinusoidal mover must be a bits type (a closure over `T` could not enter a GPU kernel).
    sinus = HyperCloudCore.build_mover(Dict{Symbol, Any}(:mover => :sinusoidal, :mover_amplitude => 0.3),
                                       Dict{Symbol, Any}(:T => Float64, :D => 1, :Domain => get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true)))
    if TEST_GROUP != "backends-a"
        @test sinus isa PrescribedMover{<:HyperCloudCore.SinusoidalVelocity{1, Float64}} && isbits(sinus)
        @test sinus.velocity(Space{1, Float64}(-0.5), 0.0) ≈ Space{1, Float64}(0.3 * sin(2π * 0.25))
    end
    # Each case is a new combination of scheme, MOOD and tableau types, i.e. a full compilation of the stepper
    # (~13 s, three times that on the CI runners): orthogonal features (limiter, MOOD variant, tableau, streamed,
    # mover) share cases, so that every kernel is still compiled for every backend.
    cases = [
        ("MUSCL3 VK MOODu2 RK3",          () -> MUSCL(Float64, 1, 1, 3, VenkatakrishnanLimiter(:hard), RusanovFlux()), MOOD(EPD1(), MOODu2(1e-6)), RK3_SSP_Tableau(Float64), NoGridMover()),
        ("MUSCL2 Minmod MOODu1 EPD2 SSPRK54", () -> MUSCL(Float64, 1, 1, 2, MinmodLimiter(:hard), RusanovFlux()), MOOD(EPD2(), MOODu1(1e-6)), RK4_SSP_Tableau(Float64), NoGridMover()),
        ("MUSCL3 BJ soft MOODu2 streamed", () -> MUSCL(Float64, 1, 1, 3, BarthJespersenLimiter(:soft), RusanovFlux(); streamed = true), MOOD(EPD1(), MOODu2(1e-6)), RK2_SSP_Tableau(Float64), NoGridMover()),
        ("MUSCL5 aligned streamed",       () -> MUSCL(Float64, 1, 1, 5, NoLimiter(), AlignedRusanovFlux(); div_order = 5, streamed = true), MOOD(EPD1(), MOODu2(1e-6)), RK3_SSP_Tableau(Float64), NoGridMover()),
        ("MUSCL4 IMEX ARS222",            () -> MUSCL(Float64, 1, 1, 4, NoLimiter(), RusanovFlux()), MOOD(EPD1(), MOODu1(1e-6)), IMEX_ARS222_Tableau(Float64), NoGridMover()),
        ("MUSCL3 sinusoidal RK2",         () -> MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux()), MOOD(), RK2_SSP_Tableau(Float64), sinus),
        ("MUSCL1 moment moving",          () -> MUSCL(Float64, 1, 1, 1, NoLimiter(), AlignedRusanovFlux(); moment_correction = true), MOOD(), RK2_SSP_Tableau(Float64), mover),
        ("Central2 RK2",                  () -> CentralDivergence(Float64, 1, 1, 2), MOOD(), RK2_SSP_Tableau(Float64), NoGridMover()),
        ("Upwind Euler",                  () -> UpwindDivergence(Float64, 1, 1, 1, :Classic, UpwindFlux()), MOOD(), RK1_Euler_Tableau(Float64), NoGridMover()),
        ("Upwind1 Tiwari RK2",            () -> UpwindDivergence(Float64, 1, 1, 1, :Tiwari, UpwindFlux()), MOOD(), RK2_SSP_Tableau(Float64), NoGridMover()),
        ("WENO2 RK2",                     () -> WENO(Float64, 1, 1, 2), MOOD(), RK2_SSP_Tableau(Float64), NoGridMover()),
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
        @testset "$name" for (name, make_interp, mood, tab, mv) in _backend_cases(cases)
            ref = run_case(PolyesterBackend(), make_interp, mood, tab, mv)
            thr = run_case(ThreadsBackend(), make_interp, mood, tab, mv)
            @test thr == ref                                          # same arithmetic, bitwise
            dev = run_case(ka, make_interp, mood, tab, mv)            # compiled as device kernels (PoCL)
            @test all(isapprox.(first.(dev[1]), first.(ref[1]); atol = 1e-12))
            @test all(isapprox.(first.(dev[2]), first.(ref[2]); atol = 1e-12))
            @test isapprox(dev[3], ref[3]; rtol = 1e-9)      # dt from per-slot weights computed in a device kernel (contraction differs)
            @test dev[4] == ref[4]
        end
    finally
        set_backend!(PolyesterBackend())
    end

    if TEST_GROUP != "backends-a"                       # the remaining checks run once, in `backends-b`
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

        @testset "KA kernels whose work-group limit is below the configured size" begin
            # Private memory grows with the state size (2-D BGK with 36 components: PoCL allows 231 work-items for
            # the flux kernel); a static work-group size of 256 used to throw. Which kernels hit their limit depends on
            # the CPU and the PoCL build, so the test configures a size above any device's limit (2^20 work-items):
            # every launch then exceeds the kernel's limit and takes the fallback.
            wide_body!(i, out, k) = (out[i] = Float64(i * k); nothing)
            out = zeros(1000)
            @test KABackend(HyperCloudCore.KA.CPU()).workgroupsize == 256   # the default
            ka  = KABackend(HyperCloudCore.KA.CPU(); workgroupsize = 2^20)
            for _ in 1:2                                                    # the fallback is remembered
                fill!(out, 0)
                foreach_particle(wide_body!, ka, length(out), (out, 7))
                @test out == [Float64(7i) for i in 1:length(out)]
            end
            @test any(k -> k[3] === typeof(wide_body!), HyperCloudCore._TUNED_WORKGROUP)
            @test !HyperCloudCore._is_workgroup_size_error(ArgumentError("other")) &&
                  !HyperCloudCore._is_workgroup_size_error(DomainError(1))     # other errors are rethrown
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
end
