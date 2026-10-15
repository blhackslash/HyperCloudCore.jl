# Device tests, included by runtests.jl when HYPERCLOUD_BACKEND=amdgpu (environment: test/gpu).
# The kernel passes of a time step run on a grid built with `device_grid`, with the interpolator and the
# stepper holding their buffers on the device, and must reproduce the host result.

const DeviceArray   = ROCArray
const device_backend = ROCBackend()

@testset "Device residency ($(nameof(DeviceArray)))" begin
    N  = 80
    eq = make_advection_1d()
    dt = 0.4 * (2 / N)
    gauss = x -> exp(-x^2 / 0.02)
    box   = x -> abs(x) < 0.3 ? 1.0 : 0.0                     # discontinuous: makes MOOD drop orders
    muscl(order, limiter) = st -> MUSCL(Float64, 1, 1, order, limiter, RusanovFlux(); storage = st)
    cases = [
        ("MUSCL3 NoLimiter RK3",        muscl(3, NoLimiter()),                     MOOD(),                           RK3_SSP_Tableau(Float64), gauss),
        ("MUSCL3 VK RK2",               muscl(3, VenkatakrishnanLimiter(:hard)),   MOOD(),                           RK2_SSP_Tableau(Float64), gauss),
        ("MUSCL2 Euler",                muscl(2, NoLimiter()),                     MOOD(),                           RK1_Euler_Tableau(Float64), gauss),
        ("Central2 RK2",                st -> CentralDivergence(Float64, 1, 1, 2), MOOD(),                           RK2_SSP_Tableau(Float64), gauss),
        ("MUSCL3 MOODu2 EPD1 RK3 box",  muscl(3, NoLimiter()),                     MOOD(EPD1(), MOODu2(1e-6)),       RK3_SSP_Tableau(Float64), box),
        ("MUSCL3 MOODu1 SEPD0 RK2 box", muscl(3, NoLimiter()),                     MOOD(StrictEPD0(), MOODu1(1e-6)), RK2_SSP_Tableau(Float64), box),
        ("MUSCL2 MOODu1 EPD2 RK2 box",  muscl(2, NoLimiter()),                     MOOD(EPD2(), MOODu1(1e-6)),       RK2_SSP_Tableau(Float64), box),
    ]
    try
        @testset "$name" for (name, make_interp, mood, tab, ic) in cases
            set_backend!(PolyesterBackend())                              # host reference
            pg_ref     = set_ic!(make_pg_1d(N), ic)
            interp_ref = make_interp(Array)
            ts_ref     = GeneralRKTimeStepper(eq, interp_ref, mood, tab)
            dt_ref     = get_time_step(pg_ref, eq, interp_ref)
            for k in 0:3; ts_ref(eq, pg_ref, k * dt, dt); end

            dg     = HyperCloudCore.device_grid(DeviceArray, set_ic!(make_pg_1d(N), ic))
            interp = make_interp(DeviceArray)
            ts     = GeneralRKTimeStepper(eq, interp, mood, tab; storage = DeviceArray)
            @test parent(ts.rho_n) isa DeviceArray && parent(ts.int_buffer.f) isa DeviceArray
            set_backend!(device_backend)
            @test isapprox(get_time_step(dg, eq, interp), dt_ref; rtol = 1e-12)
            for k in 0:3; ts(eq, dg, k * dt, dt); end
            out = Array(dg.rhos)
            @test all(isapprox.(first.(out[1:N]), first.(pg_ref.rhos[1:N]); atol = 1e-12))
            @test Array(dg.core.particle_orders)[1:N] == pg_ref.core.particle_orders[1:N]
            @test Array(mood_flags(ts))[1:N] == mood_flags(ts_ref)[1:N]
            ic === box && @test any(view(mood_flags(ts_ref), 1:N))        # the MOOD passes were exercised
        end

        @testset "boundary conditions: $(nameof(typeof(bc)))" for bc in (FixedDirichlet(), OutflowBC())
            function make_pg_bc(bc)
                dx   = 2.0 / N
                geom = get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = false,
                                              bc_map = Dict{Int, AbstractBoundaryCondition}(1 => bc))
                return set_ic!(ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1), x -> exp(-(x + 0.5)^2 / 0.02))
            end
            mood = MOOD(EPD1(), MOODu1(1e-6))
            tab  = RK3_SSP_Tableau(Float64)
            set_backend!(PolyesterBackend())
            pg_ref     = make_pg_bc(bc)
            interp_ref = muscl(3, NoLimiter())(Array)
            ts_ref     = GeneralRKTimeStepper(eq, interp_ref, mood, tab)
            n = pg_ref.meta.N
            for k in 0:5; ts_ref(eq, pg_ref, k * dt, dt); end

            dg     = HyperCloudCore.device_grid(DeviceArray, make_pg_bc(bc))
            interp = muscl(3, NoLimiter())(DeviceArray)
            ts     = GeneralRKTimeStepper(eq, interp, mood, tab; storage = DeviceArray)
            set_backend!(device_backend)
            for k in 0:5; ts(eq, dg, k * dt, dt); end
            out = Array(dg.rhos)
            @test all(isapprox.(first.(out[1:n]), first.(pg_ref.rhos[1:n]); atol = 1e-12))
            @test Array(dg.core.particle_orders)[1:n] == pg_ref.core.particle_orders[1:n]
            @test any(pg_ref.core.is_boundary[1:n])                        # the condition had particles to act on
        end

        @testset "moving grid: neighbour search on the device" begin
            mover = PrescribedMover((x, t) -> Space{1, Float64}(0.3 * sin(π * x[1])))
            tab   = RK2_SSP_Tableau(Float64)
            set_backend!(PolyesterBackend())
            pg_ref = set_ic!(make_pg_1d(N; mover = mover), gauss)
            ts_ref = GeneralRKTimeStepper(eq, muscl(3, NoLimiter())(Array), MOOD(), tab)
            for k in 0:3; ts_ref(eq, pg_ref, k * dt, dt); end

            dg = HyperCloudCore.device_grid(DeviceArray, set_ic!(make_pg_1d(N; mover = mover), gauss))
            @test parent(dg.neighbor.indices) isa DeviceArray && parent(dg.bins.cell_particles) isa DeviceArray
            ts = GeneralRKTimeStepper(eq, muscl(3, NoLimiter())(DeviceArray), MOOD(), tab; storage = DeviceArray)
            set_backend!(device_backend)
            for k in 0:3; ts(eq, dg, k * dt, dt); end

            n   = pg_ref.meta.N
            tot = first(pg_ref.neighbor.ranges[n + 1]) - 1
            @test dg.meta.max_nb == pg_ref.meta.max_nb && dg.meta.geometry_version == pg_ref.meta.geometry_version
            @test Array(dg.neighbor.ranges)[1:n + 1] == pg_ref.neighbor.ranges[1:n + 1]
            @test Array(parent(dg.neighbor.indices))[1:tot] == parent(pg_ref.neighbor.indices)[1:tot]   # same lists, same order
            @test Array(parent(dg.neighbor.weights))[1:tot] == parent(pg_ref.neighbor.weights)[1:tot]
            @test all(isapprox.(first.(Array(dg.core.positions)[1:n]), first.(pg_ref.core.positions[1:n]); atol = 1e-12))
            @test all(isapprox.(first.(Array(dg.rhos)[1:n]), first.(pg_ref.rhos[1:n]); atol = 1e-12))
            @test all(isapprox.(Array(dg.core.volumes)[1:n], pg_ref.core.volumes[1:n]; rtol = 1e-12))
        end

        @testset "builders with context[:Storage]" begin
            ctx = Dict{Symbol, Any}(:T => Float64, :D => 1, :M => 1, :Equation => eq, :Storage => DeviceArray,
                                    :ExplicitSources => (), :ImplicitSources => ())
            ctx[:Domain]     = build_domain(Dict{Symbol, Any}(:domain => :rectangular, :mins => (-1.0,), :maxs => (1.0,), :periodic => true), ctx)
            ctx[:WeightConf] = Dict{Symbol, Any}(:name => :exponential, :range => 2.5, :alpha => 1.0)
            set_backend!(PolyesterBackend())                              # the grid is built on the host, then moved
            pg = build_particle_grid(Dict{Symbol, Any}(:Ns => (N,)), ctx)
            @test parent(pg.rhos) isa DeviceArray && ctx[:Grid] === pg
            ctx[:Flux]    = build_flux(Dict{Symbol, Any}(:name => :Rusanov), ctx)
            ctx[:Limiter] = build_limiter(Dict{Symbol, Any}(:name => :none), ctx)
            ctx[:MOOD]    = build_mood(Dict{Symbol, Any}(:criterion => :U1, :strategy => :EPD1, :delta_relax => 1e-6), ctx)
            ctx[:Scheme]  = build_scheme(Dict{Symbol, Any}(:name => :MUSCL, :order => 3), ctx)
            ctx[:Tableau] = build_tableau(Dict{Symbol, Any}(:stepper => :RK3), ctx)
            ts = build_timestepper(ctx)
            @test parent(ctx[:Scheme].gradients) isa DeviceArray && parent(ts.rho_n) isa DeviceArray
            n  = pg.meta.N
            copyto!(pg.rhos, 1, DeviceArray([State{1, Float64}(gauss(x[1])) for x in Array(pg.core.positions)[1:n]]), 1, n)
            set_backend!(device_backend)
            for k in 0:3; ts(eq, pg, k * dt, dt); end
            @test all(isfinite, first.(Array(pg.rhos)[1:n]))
        end

        @testset "adaptivity: host remap, device grid updated in place" begin
            ic_s = x -> 1.0 + 0.5 * sin(π * x)
            ad   = ParticleAdaptivity(Float64)
            kmax = 3
            function punch_hole!(pg, p)                                   # as in the host unit test
                N0 = pg.meta.N
                ev = HyperCloudCore._AdaptEvent{1, Float64}(pg.core.positions[p], [p], Space{1, Float64}[])
                np, nr, nv = Space{1, Float64}[], State{1, Float64}[], Float64[]
                HyperCloudCore._remap_event!(pg, ad, ev, np, nr, nv)
                mask = pg.shared.bit_buffer; fill!(view(mask, 1:N0), false); mask[p] = true
                HyperCloudCore._apply_topology!(pg, mask, np, nr, nv, kmax)
                pg.neighbor(pg)
            end
            set_backend!(PolyesterBackend())
            pg_ref = set_ic!(make_pg_1d(40), ic_s); punch_hole!(pg_ref, 10)
            dg     = HyperCloudCore.device_grid(DeviceArray, pg_ref)       # same hole on the device
            N0     = pg_ref.meta.N
            rep_ref = adapt_particles!(pg_ref, ad, kmax)
            set_backend!(device_backend)
            m0  = total_mass(dg)
            rep = adapt_particles!(dg, ad, kmax)
            @test (rep.inserted, rep.removed) == (rep_ref.inserted, rep_ref.removed) == (1, 0)
            @test dg.meta.N == pg_ref.meta.N == N0 + 1 && dg.meta.N_interior == pg_ref.meta.N_interior
            @test maximum(abs.(rep.mass_residual)) < 1e-12 && maximum(abs.(total_mass(dg) - m0)) < 1e-12
            n = dg.meta.N
            @test Array(dg.core.positions)[1:n] == pg_ref.core.positions[1:n]
            @test Array(dg.rhos)[1:n] == pg_ref.rhos[1:n] && Array(dg.core.volumes)[1:n] == pg_ref.core.volumes[1:n]
            @test Array(dg.neighbor.ranges)[1:n + 1] == pg_ref.neighbor.ranges[1:n + 1]
            @test parent(dg.rhos) isa DeviceArray && length(dg.rhos) >= n
            rep2 = adapt_particles!(dg, ad, kmax)                          # well spaced: the pre-check alone
            @test rep2.inserted == 0 && rep2.removed == 0

            set_backend!(PolyesterBackend())
            pg_m = set_ic!(make_pg_1d(40), ic_s)
            pg_m.core.positions[21] = pg_m.core.positions[20] + Space{1, Float64}(0.2 * 2 / 40)
            pg_m.neighbor(pg_m); update_volumes!(pg_m, ShepardVolumes())
            dg_m = HyperCloudCore.device_grid(DeviceArray, pg_m)
            rep_ref = adapt_particles!(pg_m, ad, kmax)
            set_backend!(device_backend)
            rep = adapt_particles!(dg_m, ad, kmax)
            @test (rep.inserted, rep.removed) == (rep_ref.inserted, rep_ref.removed) == (1, 2)
            @test dg_m.meta.N == pg_m.meta.N && maximum(abs.(rep.mass_residual)) < 1e-12
            @test Array(dg_m.rhos)[1:dg_m.meta.N] == pg_m.rhos[1:pg_m.meta.N]
        end

        @testset "host buffers are rejected" begin
            set_backend!(PolyesterBackend())                              # grids are built on the host
            pg_host = set_ic!(make_pg_1d(N), gauss)
            dg = HyperCloudCore.device_grid(DeviceArray, set_ic!(make_pg_1d(N), gauss))
            interp_host = MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux())
            interp_dev  = MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux(); storage = DeviceArray)
            ts_host = GeneralRKTimeStepper(eq, interp_host, MOOD(), RK3_SSP_Tableau(Float64))
            set_backend!(device_backend)
            @test_throws ArgumentError ts_host(eq, dg, 0.0, dt)                      # stepper buffers on the host
            @test_throws ArgumentError get_time_step(pg_host, eq, interp_dev)        # grid on the host
        end

        @testset "Growable on the device" begin
            g = Growable{Float64}(DeviceArray)
            @test parent(g) isa DeviceArray && length(g) == 0
            HyperCloudCore.ensure_capacity!(g, 10); fill!(g, 1.0)
            n = length(g)
            HyperCloudCore.ensure_capacity!(g, 100)
            @test length(g) >= 100 && parent(g) isa DeviceArray
            @test all(==(1.0), Array(parent(g))[1:n])                                   # contents survive growth
            @test HyperCloudCore.kernel_view(g) === parent(g)
        end
    finally
        set_backend!(PolyesterBackend())
    end
end
