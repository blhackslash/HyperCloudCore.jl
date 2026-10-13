# =========================================================================
# PIPELINE TESTS: full simulations through PDEStudioCore (slow).
# Each block runs one `run_all_simulations` and asserts on the stored results.
# =========================================================================

# ---- helpers -------------------------------------------------------------

const BASE_1D = Dict{Symbol, Any}(
    :sim_func_name => :run_direct_simulation,
    :snapshots => 5, :tmax => 2.0,
    :Grid_domain => :rectangular, :Grid_mins => (-1.0,), :Grid_maxs => (1.0,), :Grid_Ns => (100,),
    :Grid_periodic => true, :Grid_SEED => 42, :Grid_randomness_factor => (0.0,),
    :Weight_name => :exponential, :Weight_range => 2.5, :Weight_alpha => 1.0,
    :PDE_name => :linear, :PDE_velocities => ((1.0,),),
    :IC_name => :gauss, :IC_a => (1.0,), :IC_b => (0.0,), :IC_width => 0.1,
    :Time_stepper => :RK3, :Time_CFL => 0.4,
    :Scheme_MLS_order => 0,
)

const BOX_IC = Dict{Symbol, Any}(:IC_name => :box, :IC_u_bg => (0.0,), :IC_u_box => (1.0,), :IC_mins => (-0.5,), :IC_maxs => (0.5,))

muscl(order; limiter = :none, mood = nothing, extra...) = begin
    d = Dict{Symbol, Any}(:Scheme_name => :MUSCL, :Flux_name => :Rusanov, :Scheme_order => order, :Limiter_name => limiter)
    limiter == :none || (d[:Limiter_mode] = :hard)
    mood === nothing || merge!(d, Dict(:MOOD_criterion => mood[1], :MOOD_strategy => mood[2], :MOOD_delta_relax => 1e-4))
    merge!(d, Dict(extra))
end

"Run every method in `methods` over `varied` and return a loader `sim(method_key; overrides...)`."
function run_suite(shared, methods; varied = Dict{Symbol, Any}(), stats = true)
    config = SimulationConfig(shared, methods, collect(keys(methods)); varied = varied, reference = :analytical_solution)
    run_all_simulations(config; force_overwrite = true, calculate_stats = stats)
    return (key; overrides...) -> load_sim_data(create_param_dict(merge(shared, methods[key])..., overrides...))
end

final_stat(sim, key) = sim.stats[key][end][1]
eoc(sim_coarse, sim_fine, key = :relative_l2error) = log2(final_stat(sim_coarse, key) / final_stat(sim_fine, key))
u_extrema(sim) = extrema(v[1] for v in sim.u[end])
check_runs(sim) = (@test sim isa LSimData; @test isfinite(final_stat(sim, :l1error)))

# ---- 1. Every spatial scheme converges and conserves mass -----------------

@testset "Spatial schemes: sanity, mass, convergence" begin
    methods = Dict(
        :upwind  => Dict(:Scheme_name => :Upwind,  :Flux_name => :Rusanov, :Scheme_upwind_alg_nd => :Classic, :Scheme_order => 1),
        :muscl   => muscl(2; limiter = :minmod),
        :weno    => Dict(:Scheme_name => :WENO,    :Flux_name => :Rusanov, :Scheme_order => 2),
        :central => Dict(:Scheme_name => :Central, :Flux_name => :Rusanov, :Scheme_order => 2),
    )
    sim = run_suite(BASE_1D, methods; varied = Dict(:Grid_Ns => [(200,), (400,)]))

    for key in keys(methods)
        @testset "$key" begin
            coarse, fine = sim(key; Grid_Ns = (200,)), sim(key; Grid_Ns = (400,))
            check_runs(coarse)
            @test length(coarse.stats[:l1error]) == length(coarse.t)
            m0 = fine.stats[:mass][1][1]
            @test all(m -> isapprox(m[1], m0; rtol = 1e-2), fine.stats[:mass])
            @test final_stat(fine, :l1error) < final_stat(coarse, :l1error)
        end
    end
end

# ---- 2. MUSCL orders 2–5, with and without MOOD, keep their EOC -----------

@testset "MUSCL high-order EOC (orders 2–5, ± MOOD)" begin
    shared  = merge(BASE_1D, Dict(:Time_stepper => :RK4))
    methods = Dict{Symbol, Any}()
    for o in 2:5
        methods[Symbol("plain$o")] = muscl(o)
        methods[Symbol("mood$o")]  = muscl(o; mood = (:U2, :EPD1))
    end
    sim = run_suite(shared, methods; varied = Dict(:Grid_Ns => [(200,), (400,)]))

    errs = Dict{Symbol, Float64}()
    for o in 2:5, tag in (:plain, :mood)
        key = Symbol("$tag$o")
        @testset "$key" begin
            coarse, fine = sim(key; Grid_Ns = (200,)), sim(key; Grid_Ns = (400,))
            errs[key] = final_stat(fine, :relative_l2error)
            r = eoc(coarse, fine)
            @test final_stat(fine, :relative_l2error) < final_stat(coarse, :relative_l2error)
            # MOODu2 needs curvature coefficients for its smooth-extremum override; at order 2 it
            # degrades to a plain DMP and keeps flagging the Gaussian peak, which costs some EOC.
            lo = (tag == :mood && o == 2) ? 1.2 : (o <= 3 ? 1.5 : 3.5)
            @test lo < r < (o <= 3 ? 2.5 : 5.0)
        end
    end
    @testset "Error hierarchy" begin
        @test errs[:plain4] < errs[:plain2] && errs[:plain5] < errs[:plain3]
        @test errs[:plain3] <= errs[:plain2] && errs[:plain5] <= errs[:plain4]
    end
    @testset "MOOD does not degrade a smooth solution" begin
        for o in 2:5
            @test errs[Symbol("mood$o")] <= 3 * errs[Symbol("plain$o")]
        end
    end
end

# ---- 3. Boundedness on a discontinuity: MOOD and VK bound, unlimited does not

@testset "Boundedness on a box (MOOD, VK limiter vs unlimited)" begin
    shared  = merge(BASE_1D, BOX_IC)
    methods = Dict{Symbol, Any}()
    for o in 2:5
        methods[Symbol("mood$o")]  = muscl(o; mood = (:U2, :EPD1))
        methods[Symbol("vk$o")]    = muscl(o; limiter = :VK)
        methods[Symbol("unlim$o")] = muscl(o)
    end
    sim = run_suite(shared, methods; stats = false)

    for o in 2:5
        @testset "order $o" begin
            lo, hi = u_extrema(sim(Symbol("mood$o")))
            @test lo >= -1e-2 && hi <= 1 + 1e-2
            lo, hi = u_extrema(sim(Symbol("vk$o")))
            @test lo >= -3e-2 && hi <= 1 + 3e-2
            lo, hi = u_extrema(sim(Symbol("unlim$o")))
            @test lo < -1e-2 || hi > 1 + 1e-2
        end
    end
end

# ---- 4. Execution smoke tests: every strategy × criterion, stepper, mover ---

@testset "Execution smoke tests" begin
    short = merge(BASE_1D, Dict(:tmax => 0.05, :snapshots => 2, :Grid_Ns => (50,), :Time_CFL => 0.3))

    @testset "MOOD strategies × criteria" begin
        methods = Dict(Symbol("$(s)_$(c)") => muscl(2; mood = (c, s))
                       for s in (:EPD0, :SEPD0, :EPD1, :EPD2), c in (:U1, :U2))
        sim = run_suite(merge(short, BOX_IC, Dict(:Time_stepper => :RK2)), methods)
        for key in keys(methods); check_runs(sim(key)); end
    end

    @testset "Explicit Runge-Kutta tableaus" begin
        methods = Dict(:euler => Dict(:Time_stepper => :Euler, :Time_CFL => 0.05),
                       :rk2   => Dict(:Time_stepper => :RK2),
                       :rk3   => Dict(:Time_stepper => :RK3),
                       :rk4   => Dict(:Time_stepper => :RK4))
        sim = run_suite(merge(short, muscl(5), Dict(:IC_width => 1.0)), methods)
        for key in keys(methods); check_runs(sim(key)); end
    end

    @testset "IMEX kinetic relaxation" begin
        shared = merge(short, muscl(5), Dict(:IC_width => 1.0,
            :sim_func_name => :run_kinetic_simulation,
            :Kinetic_velocities => ((-2.0, 2.0),), :Kinetic_indices => [1, 3], :Kinetic_epsilon => 1e-4))
        methods = Dict(
            :ars233      => Dict(:Time_stepper => :ARS233),
            :ars233_mood => merge(Dict(:Time_stepper => :ARS233), muscl(5; mood = (:U2, :EPD1))),
            :ars222      => Dict(:Time_stepper => :ARS222),
            :prssp3      => Dict(:Time_stepper => :PRSSP3),
            :ssp332      => Dict(:Time_stepper => :SSP332),
            :imex_euler  => Dict(:Time_stepper => :IMEXEuler, :Time_CFL => 0.05))
        sim = run_suite(shared, methods)
        for key in keys(methods); check_runs(sim(key)); end
    end

    @testset "Moving grid with every spatial scheme" begin
        shared  = merge(short, Dict(:Grid_mover => :sinusoidal, :Grid_mover_amplitude => 0.2))
        methods = Dict(
            :upwind  => Dict(:Scheme_name => :Upwind,  :Flux_name => :Rusanov, :Scheme_upwind_alg_nd => :Classic, :Scheme_order => 1),
            :muscl   => muscl(3; mood = (:U2, :EPD2)),
            :weno    => Dict(:Scheme_name => :WENO,    :Flux_name => :Rusanov, :Scheme_order => 2),
            :central => Dict(:Scheme_name => :Central, :Flux_name => :Rusanov, :Scheme_order => 2))
        sim = run_suite(shared, methods)
        for key in keys(methods)
            s = sim(key)
            check_runs(s)
            @test any(s.x[end][i] != s.x[1][i] for i in eachindex(s.x[1]))     # particles actually moved
        end
    end
end

# ---- 5. Moving grids keep accuracy; Lagrangian motion beats Eulerian ------

@testset "Moving grid accuracy" begin
    # Reference solutions are evaluated at the stored (moved) particle positions.
    @testset "Prescribed motion: MUSCL3 keeps its EOC" begin
        shared  = merge(BASE_1D, Dict(:tmax => 0.5, :Grid_mover => :sinusoidal, :Grid_mover_amplitude => 0.2))
        methods = Dict(:moving => muscl(3), :static => merge(muscl(3), Dict(:Grid_mover => :none)))
        sim = run_suite(shared, methods; varied = Dict(:Grid_Ns => [(100,), (200,)]))
        r_moving = eoc(sim(:moving; Grid_Ns = (100,)), sim(:moving; Grid_Ns = (200,)))
        r_static = eoc(sim(:static; Grid_Ns = (100,)), sim(:static; Grid_Ns = (200,)))
        @test r_moving > 1.5
        @test abs(r_moving - r_static) < 1.0
    end

    @testset "Adaptivity on a compressing/expanding grid" begin
        # The sinusoidal mover clusters particles in some regions and starves others, which
        # triggers merges and insertions. The remap must not spoil mass or stability.
        shared  = merge(BASE_1D, Dict(:tmax => 0.5, :Grid_mover => :sinusoidal, :Grid_mover_amplitude => 0.4))
        methods = Dict(:moving       => muscl(3; mood = (:U2, :EPD1)),
                       :moving_adapt => merge(muscl(3; mood = (:U2, :EPD1)), Dict(:Adapt_enabled => true)))
        sim = run_suite(shared, methods)
        scalar(v) = v isa Number ? v : scalar(first(v))
        s0, s1 = sim(:moving), sim(:moving_adapt)
        check_runs(s0); check_runs(s1)
        @test scalar(s1.stats[:final_particle_count]) != scalar(s0.stats[:final_particle_count])
        e0, e1 = scalar(s0.stats[:discrete_mass_error]), scalar(s1.stats[:discrete_mass_error])
        @test e1 < max(2e0, 1e-6)
        @test final_stat(s1, :l1error) < 2 * final_stat(s0, :l1error) + 1e-3
    end

    @testset "Lagrangian motion makes linear advection exact" begin
        shared  = merge(BASE_1D, Dict(:tmax => 1.0))
        methods = Dict(:lagrangian => merge(muscl(2; limiter = :minmod), Dict(:Grid_mover => :lagrangian)),
                       :eulerian   => muscl(2; limiter = :minmod))
        sim = run_suite(shared, methods)
        @test final_stat(sim(:lagrangian), :l1error) < 1e-8
        @test final_stat(sim(:lagrangian), :l1error) < final_stat(sim(:eulerian), :l1error)
    end
end

# ---- 6. 2-D upwind algorithms converge at first order ----------------------

@testset "2-D upwind algorithms (Classic, Tiwari, Praveen)" begin
    shared = Dict{Symbol, Any}(
        :sim_func_name => :run_direct_simulation, :snapshots => 3, :tmax => 0.1,
        :PDE_name => :linear, :PDE_velocities => ((1.0,), (1.0,)),
        :Grid_domain => :rectangular, :Grid_mins => (-1.0, -1.0), :Grid_maxs => (1.0, 1.0), :Grid_Ns => (20, 20),
        :Grid_periodic => true, :Grid_SEED => 42, :Grid_randomness_factor => (0.0, 0.0),
        :IC_name => :gauss, :IC_a => (1.0,), :IC_b => (0.0, 0.0), :IC_width => 0.1,
        :Weight_name => :exponential, :Weight_range => 2.5, :Weight_alpha => 1.0,
        :Time_stepper => :RK2, :Time_CFL => 0.4)
    methods = Dict(alg => Dict(:Scheme_name => :Upwind, :Flux_name => :Rusanov, :Scheme_upwind_alg_nd => alg, :Scheme_order => 1)
                   for alg in (:Classic, :Tiwari, :Praveen))
    sim = run_suite(shared, methods; varied = Dict(:Grid_Ns => [(50, 50), (100, 100)]))

    for alg in keys(methods)
        @testset "$alg" begin
            coarse, fine = sim(alg; Grid_Ns = (50, 50)), sim(alg; Grid_Ns = (100, 100))
            @test length(coarse.x[1][1]) == 2
            @test final_stat(fine, :relative_l2error) < final_stat(coarse, :relative_l2error)
            @test 0.5 < eoc(coarse, fine) < 1.5
        end
    end
end

# ---- 7. Non-periodic spherical domain with boundary conditions -------------

@testset "Spherical domain & boundary conditions" begin
    shared = Dict{Symbol, Any}(
        :sim_func_name => :run_direct_simulation, :snapshots => 2, :tmax => 0.1,
        :PDE_name => :linear, :PDE_velocities => ((1.0,), (1.0,)),
        :Grid_domain => :spherical, :Grid_mins => (-1.0, -1.0), :Grid_maxs => (1.0, 1.0), :Grid_Ns => (30, 30),
        :Grid_periodic => false, :Grid_SEED => 42, :Grid_randomness_factor => (0.2, 0.2),
        :IC_name => :gauss, :IC_a => (1.0,), :IC_b => (0.0, 0.0), :IC_width => 0.5,
        :Scheme_name => :Upwind, :Flux_name => :Upwind, :Scheme_upwind_alg_nd => :Classic, :Scheme_order => 1,
        :Weight_name => :exponential, :Weight_range => 2.5, :Weight_alpha => 1.0,
        :Time_stepper => :RK2, :Time_CFL => 0.4)
    methods = Dict(:fixed => Dict(:Grid_bc => Dict(1 => :fixed_dirichlet)), :outflow => Dict(:Grid_bc => Dict(1 => :outflow)))
    sim = run_suite(shared, methods; stats = false)

    for key in keys(methods)
        s = sim(key)
        @test s isa LSimData
        @test length(s.x[1][1]) == 2
        @test all(isfinite(v[1]) for v in s.u[end])
    end
end
