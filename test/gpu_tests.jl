# Device tests, included by runtests.jl when HYPERCLOUD_BACKEND=amdgpu (environment: test/gpu).
# The kernel passes of a time step run on a grid built with `device_grid`, with the interpolator and the
# stepper holding their buffers on the device, and must reproduce the host result.

const DeviceArray   = ROCArray
const device_backend = ROCBackend()

@testset "Device residency ($(nameof(DeviceArray)))" begin
    N  = 80
    eq = make_advection_1d()
    dt = 0.4 * (2 / N)
    ic = x -> exp(-x^2 / 0.02)
    cases = [
        ("MUSCL3 NoLimiter RK3", st -> MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux(); storage = st), RK3_SSP_Tableau(Float64)),
        ("MUSCL3 VK RK2",        st -> MUSCL(Float64, 1, 1, 3, VenkatakrishnanLimiter(:hard), RusanovFlux(); storage = st), RK2_SSP_Tableau(Float64)),
        ("MUSCL2 Euler",         st -> MUSCL(Float64, 1, 1, 2, NoLimiter(), RusanovFlux(); storage = st), RK1_Euler_Tableau(Float64)),
        ("Central2 RK2",         st -> CentralDivergence(Float64, 1, 1, 2), RK2_SSP_Tableau(Float64)),
    ]
    try
        @testset "$name" for (name, make_interp, tab) in cases
            set_backend!(PolyesterBackend())                              # host reference
            pg_ref     = set_ic!(make_pg_1d(N), ic)
            interp_ref = make_interp(Array)
            ts_ref     = GeneralRKTimeStepper(eq, interp_ref, MOOD(), tab)
            dt_ref     = get_time_step(pg_ref, eq, interp_ref)
            for k in 0:3; ts_ref(eq, pg_ref, k * dt, dt); end

            dg     = HyperCloudCore.device_grid(DeviceArray, set_ic!(make_pg_1d(N), ic))
            interp = make_interp(DeviceArray)
            ts     = GeneralRKTimeStepper(eq, interp, MOOD(), tab; storage = DeviceArray)
            @test parent(ts.rho_n) isa DeviceArray && parent(ts.int_buffer.f) isa DeviceArray
            set_backend!(device_backend)
            @test isapprox(get_time_step(dg, eq, interp), dt_ref; rtol = 1e-12)
            for k in 0:3; ts(eq, dg, k * dt, dt); end
            out = Array(dg.rhos)
            @test all(isapprox.(first.(out[1:N]), first.(pg_ref.rhos[1:N]); atol = 1e-12))
            @test Array(dg.core.particle_orders)[1:N] == pg_ref.core.particle_orders[1:N]
        end

        @testset "host buffers are rejected" begin
            set_backend!(PolyesterBackend())                              # grids are built on the host
            pg_host = set_ic!(make_pg_1d(N), ic)
            dg = HyperCloudCore.device_grid(DeviceArray, set_ic!(make_pg_1d(N), ic))
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
