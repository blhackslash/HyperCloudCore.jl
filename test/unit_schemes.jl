# =========================================================================
# SCHEME TESTS (group `schemes`): aligned dissipation, moment correction, SSP substep checks, implicit
# stage, streamed divergence, CFL estimate, MOODPAD, memory budget, moving grids, volumes, adaptivity.
# =========================================================================

@testset "Aligned dissipation: non-negative first-order coefficients with high-degree weights" begin
    # Linear advection with constant reconstructions (order 1 everywhere) and a fixed degree-3 divergence:
    # the RHS is linear, K_i = Σ_j a_ij (u_j - u_i), and a unit state at neighbour j returns a_ij directly.
    # Sorted by the weight signs (AlignedRusanovFlux) every a_ij is non-negative, so the first-order update
    # is local-extremum-diminishing for any zero-sum weights; sorted by the distance (RusanovFlux) the
    # high-degree weights make some a_ij negative.
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = true)
    dx   = 1 / 16
    pg   = ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(1.0, 3.5dx), 1; randomness = (0.2dx, 0.2dx), rng = MersenneTwister(9))
    N    = pg.meta.N
    eq   = LinearAdvection(((1.0,), (0.7,)), Float64)
    nb   = pg.neighbor
    function coefficients(flux, i)
        m  = MUSCL(Float64, 2, 1, 3, NoLimiter(), flux; div_order = 3)
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
    @test length(a_aligned) > 100 && minimum(a_aligned) >= -1e-12
    @test minimum(a_geo) < -1e-6                                       # the geometric sorting is not monotone here
    @test length(MUSCL(Float64, 2, 1, 3, NoLimiter(), AlignedRusanovFlux(); div_order = 3).dweights) == 0   # filled by prepare_geometry!
    @test build_flux(Dict{Symbol, Any}(:name => :AlignedRusanov), Dict{Symbol, Any}()) isa AlignedRusanovFlux
end

@testset "Aligned dissipation also for a first-order scheme (K = 1)" begin
    # MUSCL order 1 has no particle below the full order, so the aligned orientation used to be inactive and the
    # degree-1 divergence weights of a jittered cloud with κ = 2.5 gave some negative first-order coefficients.
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = true)
    dx   = 1 / 24
    pg   = ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(1.0, 2.5dx), 1; randomness = (0.2dx, 0.2dx), rng = MersenneTwister(9))
    N    = pg.meta.N
    eq   = LinearAdvection(((1.0,), (0.7,)), Float64)
    nb   = pg.neighbor
    function coefficients(flux, i)
        m  = MUSCL(Float64, 2, 1, 1, NoLimiter(), flux)
        ts = GeneralRKTimeStepper(eq, m, MOOD(), RK2_SSP_Tableau(Float64))
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
    a_aligned = vcat((coefficients(AlignedRusanovFlux(), i) for i in 1:N)...)
    a_geo     = vcat((coefficients(RusanovFlux(), i) for i in 1:N)...)
    @test minimum(a_aligned) >= -1e-12
    @test count(<(-1e-12), a_geo) > 0                                  # geometric sorting: some negative coefficients
end

@testset "Aligned dissipation uses the signs of the weights in use" begin
    # Particles off the cached path multiply their flux differences with a dynamic fit; the orientation used to
    # come from the cached full-degree weights anyway (G1: the safeguard lowers the degree; G3: coupled mode), or
    # was geometric without a cached factorisation (G2). First-order coefficients a_ij as above, with every
    # particle lowered to order 1 so that the aligned orientation is active on every edge.
    function coefficients(pg, eq, make_muscl, i; prepare = identity)
        N, nb = pg.meta.N, pg.neighbor
        m  = make_muscl()
        ts = GeneralRKTimeStepper(eq, m, MOOD(), RK3_SSP_Tableau(Float64))
        HyperCloudCore.update_size!(ts, N, length(nb.indices), false); HyperCloudCore.update_size!(m, N)
        fill!(view(pg.core.particle_orders, 1:N), 1)
        HyperCloudCore.prepare_geometry!(m, pg); prepare(m)
        a = Float64[]
        for k in nb.ranges[i]
            j = nb.indices[k]
            fill!(view(pg.rhos, 1:N), zero(State{1, Float64})); pg.rhos[j] = State{1, Float64}(1.0)
            HyperCloudCore.compute_explicit_rhs!(ts, eq, pg, ts.K_E_stages[1], ts.K_X_stages[1], pg.rhos, 0.0, true)
            push!(a, parent(ts.K_E_stages[1])[i][1])
        end
        return a
    end
    eq1 = make_advection_1d()

    @testset "G1: safeguard-lowered degree (1-D, R = 2.5Δx, div_order = 3), $(jit ? "jittered" : "lattice"), streamed = $streamed" for
            jit in (false, true), streamed in (false, true)
        dx = 2 / 40
        geom = get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true)
        pg = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1; randomness = (jit ? 0.2dx : 0.0,), rng = MersenneTwister(4))
        N = pg.meta.N
        mk = () -> MUSCL(Float64, 1, 1, 3, NoLimiter(), AlignedRusanovFlux(); div_order = 3, streamed = streamed)
        a = vcat((coefficients(pg, eq1, mk, i) for i in 1:N)...)
        @test minimum(a) >= -1e-12
    end

    @testset "G2: no cached factorisation (2-D jittered, K = 1), streamed = $streamed" for streamed in (false, true)
        dx = 1 / 24
        geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = true)
        pg = ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(1.0, 2.5dx), 1; randomness = (0.2dx, 0.2dx), rng = MersenneTwister(9))
        eq2 = LinearAdvection(((1.0,), (0.7,)), Float64)
        mk = () -> MUSCL(Float64, 2, 1, 1, NoLimiter(), AlignedRusanovFlux(); streamed = streamed)
        idx = 1:3:pg.meta.N
        a_cached  = vcat((coefficients(pg, eq2, mk, i) for i in idx)...)
        a_dynamic = vcat((coefficients(pg, eq2, mk, i; prepare = m -> fill!(m.div_weights.ok, false)) for i in idx)...)
        @test minimum(a_dynamic) >= -1e-12
        @test a_dynamic ≈ a_cached rtol = 1e-10                  # the dynamic fit of the same degree, solved per slot
    end

    @testset "G3: coupled mode (div_order = 0), lowered particles fit a lower degree (2-D jittered)" begin
        dx = 1 / 20
        geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = true)
        pg = ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(1.0, 2.5dx), 1; randomness = (0.2dx, 0.2dx), rng = MersenneTwister(5))
        eq2 = LinearAdvection(((1.0,), (0.7,)), Float64)
        mk = () -> MUSCL(Float64, 2, 1, 4, NoLimiter(), AlignedRusanovFlux())
        a = vcat((coefficients(pg, eq2, mk, i) for i in 1:5:pg.meta.N)...)
        @test minimum(a) >= -1e-12
    end
end

@testset "Moment-corrected first-order dissipation" begin
    # The first-order RHS of linear advection is Σ_j a_ij (u_j - u_i) ≈ (Σ_j a_ij x_ij) · ∇u: consistency needs
    # Σ_j a_ij x_ij = -a for every particle. The Rusanov dissipation adds Σ_d s |c^d_ij| x_ij, which only vanishes
    # by symmetry; the correction cancels it.
    function velocity_errors(D, jit, flux; moment, n = D == 1 ? 60 : 16)
        dx   = 1 / n
        geom = get_rectangular_domain(Float64, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D); is_periodic = true)
        pg   = ParticleGrid(geom, ntuple(_ -> dx, D), ExponentialWeightFunction(1.0, 2.5dx), 1; randomness = ntuple(_ -> jit * dx, D), rng = MersenneTwister(4))
        N, nb = pg.meta.N, pg.neighbor
        a    = D == 1 ? (1.0,) : (1.0, -0.6)
        eq   = LinearAdvection(ntuple(d -> (a[d],), D), Float64)
        m    = MUSCL(Float64, D, 1, 1, NoLimiter(), flux; moment_correction = moment)
        ts   = GeneralRKTimeStepper(eq, m, MOOD(), RK2_SSP_Tableau(Float64))
        HyperCloudCore.update_size!(ts, N, length(nb.indices), false); HyperCloudCore.update_size!(m, N)
        fill!(view(pg.core.particle_orders, 1:N), 1)
        errs, amin = Float64[], Inf
        for i in 1:(D == 1 ? 1 : 3):N
            v = zero(Space{D, Float64})
            for k in nb.ranges[i]
                j = nb.indices[k]
                fill!(view(pg.rhos, 1:N), zero(State{1, Float64})); pg.rhos[j] = State{1, Float64}(1.0)
                HyperCloudCore.compute_explicit_rhs!(ts, eq, pg, ts.K_E_stages[1], ts.K_X_stages[1], pg.rhos, 0.0, true)
                a_ij = parent(ts.K_E_stages[1])[i][1]
                v   += a_ij * nb.distances[k]
                amin = min(amin, a_ij)
            end
            push!(errs, maximum(abs, v + Space{D, Float64}(a)))
        end
        return errs, amin, m, nb, N
    end
    # the slots in use (the neighbour arrays keep spare capacity after the last range)
    slots(nb, N) = (k for i in 1:N for k in nb.ranges[i])
    @testset "$(D)-D, jitter $jit, $(nameof(typeof(flux)))" for D in (1, 2), jit in (0.2, 0.4), flux in (AlignedRusanovFlux(), RusanovFlux())
        e0, _, _, _, _      = velocity_errors(D, jit, flux; moment = false)
        e1, amin, m, nb, N  = velocity_errors(D, jit, flux; moment = true)
        @test maximum(e0) > 1e-2                                  # O(1) local inconsistency without the correction
        @test maximum(e1) < 1e-10                                 # exact first moment with it
        flux isa AlignedRusanovFlux && @test amin >= -1e-12       # coefficients stay non-negative
        @test all(k -> all(>=(0), m.mweights[k]), slots(nb, N))
    end

    @testset "boundaries: capped, finite; Float32" begin
        for (T, D) in ((Float64, 2), (Float32, 2), (Float64, 1))
            dx   = T(1 / 20)
            geom = get_rectangular_domain(T, ntuple(_ -> zero(T), D), ntuple(_ -> one(T), D); is_periodic = false)
            pg   = ParticleGrid(geom, ntuple(_ -> dx, D), ExponentialWeightFunction(one(T), T(2.5) * dx), 1; randomness = ntuple(_ -> T(0.3) * dx, D), rng = MersenneTwister(3))
            m    = MUSCL(T, D, 1, 1, NoLimiter(), AlignedRusanovFlux(); moment_correction = true)
            HyperCloudCore.update_size!(m, pg.meta.N); HyperCloudCore.prepare_geometry!(m, pg)
            nb   = pg.neighbor
            @test all(k -> all(isfinite, m.mweights[k]) && all(>=(0), m.mweights[k]), slots(nb, pg.meta.N))
            @test all(i -> sum(k -> sum(m.mweights[k]), nb.ranges[i]; init = zero(T)) <=
                           (1 + 10eps(T)) * sum(k -> sum(abs, m.dweights[k]), nb.ranges[i]; init = zero(T)) + 100floatmin(T), 1:pg.meta.N)
        end
    end

    @testset "off, higher orders and the builder key" begin
        pg = make_pg_1d(30); set_ic!(pg, x -> sin(π * x))
        eq = make_advection_1d()
        @test length(MUSCL(Float64, 1, 1, 1, NoLimiter(), AlignedRusanovFlux()).mweights) == 0
        function rhs(m)
            ts = GeneralRKTimeStepper(eq, m, MOOD(), RK2_SSP_Tableau(Float64))
            HyperCloudCore.update_size!(ts, pg.meta.N, length(pg.neighbor.indices), false); HyperCloudCore.update_size!(m, pg.meta.N)
            fill!(view(pg.core.particle_orders, 1:pg.meta.N), 2)
            HyperCloudCore.compute_explicit_rhs!(ts, eq, pg, ts.K_E_stages[1], ts.K_X_stages[1], pg.rhos, 0.0, true)
            return copy(view(parent(ts.K_E_stages[1]), 1:pg.meta.N))
        end
        # ignored above order 1 (bitwise)
        @test rhs(MUSCL(Float64, 1, 1, 2, NoLimiter(), AlignedRusanovFlux(); moment_correction = true)) ==
              rhs(MUSCL(Float64, 1, 1, 2, NoLimiter(), AlignedRusanovFlux()))
        ctx = Dict{Symbol, Any}(:T => Float64, :D => 1, :M => 1, :Flux => AlignedRusanovFlux(), :Limiter => NoLimiter())
        @test build_scheme(Dict{Symbol, Any}(:name => :MUSCL, :order => 1, :moment_correction => true), ctx).moment === Val(true)
        @test build_scheme(Dict{Symbol, Any}(:name => :MUSCL, :order => 1), ctx).moment === Val(false)
    end
end

@testset "SSP coefficients and the canonical Shu–Osher form" begin
    @test ssp_coefficient(RK1_Euler_Tableau(Float64)) ≈ 1 atol = 1e-9
    @test ssp_coefficient(RK2_SSP_Tableau(Float64)) ≈ 1 atol = 1e-9
    @test ssp_coefficient(RK2_Ralston_Tableau(Float64)) ≈ 0.5 atol = 1e-9
    @test ssp_coefficient(RK3_SSP_Tableau(Float64)) ≈ 1 atol = 1e-9
    @test ssp_coefficient(RK4_SSP_Tableau(Float64)) ≈ 1.50818004918983 atol = 1e-8   # Spiteri & Ruuth
    @test ssp_coefficient(RK4_Classical_Tableau(Float64)) == 0
    @test ssp_coefficient(IMEX_ARS222_Tableau(Float64)) == 0
    # Rounded copies of the built-in tableaus get the Float64 coefficient (the rounded SSPRK(5,4) alone gave 1.5032);
    # other Float32 tableaus are computed from their own coefficients
    for f in (RK1_Euler_Tableau, RK2_Ralston_Tableau, RK3_SSP_Tableau, RK4_SSP_Tableau, RK4_Classical_Tableau)
        @test ssp_coefficient(f(Float32)) == ssp_coefficient(f(Float64))
    end
    @test ssp_coefficient(IMEX_ARS222_Tableau(Float32)) == 0
    own32 = RKButcherTableau(Float32[0 0; 0.5 0], Float32[0, 1], Float32[0, 0.5])          # explicit midpoint: not SSP
    @test ssp_coefficient(own32) == 0
    @test HyperCloudCore._ssp_coefficient(RK4_SSP_Tableau(Float32)) < 1.504                  # what the rounded coefficients give
    # At C the canonical coefficients are non-negative, above C some are negative
    bt = RK4_SSP_Tableau(Float64); C = ssp_coefficient(bt); s = num_stages(bt)
    K = zeros(s + 1, s + 1); K[1:s, 1:s] .= bt.a_t; K[s + 1, 1:s] .= bt.b_t
    canon(r) = (α = r * K / (I + r * K); (α, (I - α) * ones(s + 1)))
    α, v = canon(C)
    @test minimum(α) >= -1e-10 && minimum(v) >= -1e-10
    α2, v2 = canon(1.01C)
    @test min(minimum(α2), minimum(v2)) < 0
    # The final update reuses the substeps of stages 2 and 4: why checking the stage candidates is not enough
    @test α[s + 1, 2] > 0.1 && α[s + 1, 4] > 0.1
end

# A criterion that records the candidate it is shown for one particle and never flags
struct RecordCriterion <: MOODCriterion end
const RECORDED = Tuple{Float64, Float64}[]                              # (U_prev, candidate) of particle 7
function (m::MOOD{<:MOODStrategy, RecordCriterion})(g, p, rho_i, nb_slice, cand, pg, U_prev)
    p == 7 && push!(RECORDED, (rho_i[1], cand[1]))
    return false
end

@testset "MOOD checks the Euler substeps of SSP tableaus" begin
    for (bt, C) in ((RK3_SSP_Tableau(Float64), 1.0), (RK4_SSP_Tableau(Float64), 1.50818004918983))
        s = num_stages(bt)
        for substeps in (true, false)
            pg = set_ic!(make_pg_1d(40), x -> 1 + 0.5 * sin(π * x))
            eq = make_advection_1d()
            ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux()), MOOD(EPD1(), RecordCriterion()), bt;
                                      mood_substeps = substeps)
            @test ts.substep_ratio ≈ (substeps ? 1 / C : 0.0) atol = 1e-9
            empty!(RECORDED)
            dt = 0.01
            u_n = pg.rhos[7][1]
            ts(eq, pg, 0.0, dt)
            @test length(RECORDED) == s                                  # one check per checkpoint 2 … s+1
            for i in 2:(s + 1)
                Yp, cand = RECORDED[i - 1]
                @test Yp ≈ ts.Y_stages[i - 1][7][1]
                if substeps
                    @test cand ≈ Yp + dt / C * ts.K_E_stages[i - 1][7][1]
                else
                    @test cand ≈ (i <= s ? ts.Y_stages[i][7][1] : pg.rhos[7][1])
                end
            end
        end
    end
    # IMEX and non-SSP tableaus keep checking the stage candidates
    eq = make_advection_1d()
    @test GeneralRKTimeStepper(eq, MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux()), MOOD(EPD1(), MOODu1(1e-6)), RK4_Classical_Tableau(Float64)).substep_ratio == 0
    ctx = Dict{Symbol, Any}(:T => Float64, :Equation => eq, :Scheme => MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux()),
                            :MOOD => MOOD(EPD1(), MOODu1(1e-6)), :Tableau => RK3_SSP_Tableau(Float64),
                            :ExplicitSources => (), :ImplicitSources => ())
    @test build_timestepper(ctx).substep_ratio ≈ 1 atol = 1e-9
    @test build_timestepper(ctx; time_conf = Dict{Symbol, Any}(:mood_substeps => false)).substep_ratio == 0
end

@testset "Implicit stage: RHS from the solve" begin
    # One implicit source: K_I of a diagonal stage is (U_out - U_in) / (dt a_ii), which equals S(U_out) for an
    # exact solve. Constant state, so the explicit RHS is zero and the step is the ARS222 solution of u' = -λu.
    # Decay rate λ (1 + κ sin(π x_i)); κ ≠ 0 makes it vary in space, so it does not commute with the advection
    struct DecaySource <: AbstractImplicitSourceTerm; λ::Float64; κ::Float64; end
    DecaySource(λ) = DecaySource(λ, 0.0)
    decay_rate(s::DecaySource, i, pg) = s.λ * (1 + s.κ * sinpi(pg.core.positions[i][1]))
    HyperCloudCore.evaluate_source(s::DecaySource, U, i::Int, pg, t::Real) = -decay_rate(s, i, pg) * U
    HyperCloudCore.implicit_solve(s::DecaySource, U_in, γ::Real, i::Int, pg, t::Real) = U_in / (1 + γ * decay_rate(s, i, pg))
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

@testset "IMEX tableaus: observed order through the stepper" begin
    # Advection (explicit) coupled with a decay (implicit) on a fixed periodic lattice: the error against a fine-Δt
    # run of the same spatial scheme is the temporal error. All four tableaus have three stages, so they share the
    # stepper type of the test above (one compilation). The old two-stage ARS222 and the inconsistent PRSSP3 had
    # EOC 1.05 / 1.07 here. The decay rate varies in space: with a constant rate the two operators commute and only
    # the sum b̃·c + b·c̃ of the coupling conditions enters (a pair with b̃·c = 0.7, b·c̃ = 0.3 passed at EOC 2.02;
    # with κ = 0.8 it gives 1.07).
    eq  = make_advection_1d()
    src = DecaySource(3.0, 0.8)
    function imex_run(bt, n; tend = 0.4)
        pg = set_ic!(make_pg_1d(20), x -> sin(π * x))
        ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 1, 1, 2, NoLimiter(), RusanovFlux()), MOOD(), (), (src,), bt)
        dt = tend / n
        for k in 0:(n - 1); ts(eq, pg, k * dt, dt); end
        return [pg.rhos[i][1] for i in interior_indices(pg)]
    end
    for (make, order) in ((IMEX_ARS222_Tableau, 2), (IMEX_ARS233_Tableau, 3), (IMEX_PRSSP3_Tableau, 2), (IMEX_SSP2332_Tableau, 2))
        bt  = make(Float64)
        ref = imex_run(bt, 640)
        e1, e2 = maximum(abs, imex_run(bt, 20) - ref), maximum(abs, imex_run(bt, 40) - ref)
        @test log2(e1 / e2) > order - 0.2
    end
end

@testset "Implicit sources with an explicit tableau are applied explicitly" begin
    # Used to be dropped (b = 0): one RK3 step of u' = -λu from u = 1 is the cubic Taylor polynomial of exp(-λΔt)
    eq  = make_advection_1d()
    pg  = set_ic!(make_pg_1d(20), x -> 1.0)
    src = DecaySource(3.0)
    ts  = @test_logs (:warn, r"applied explicitly") GeneralRKTimeStepper(eq, MUSCL(Float64, 1, 1, 2, NoLimiter(), RusanovFlux()),
                                                                          MOOD(), (), (src,), RK3_SSP_Tableau(Float64))
    @test ts.substep_ratio ≈ 1 atol = 1e-9                          # still the SSP coefficient of RK3
    # Only a tableau without an implicit part (a = 0 and b = 0) is converted: a hand-made a = 0, b ≠ 0 keeps its b,
    # with a warning that nothing is solved implicitly either
    own = IMEXButcherTableau(zeros(2, 2), [0.0 0.0; 1.0 0.0], [0.0, 1.0], [0.0, 1.0], [0.25, 0.75], [0.5, 0.5])
    ts2 = @test_logs (:warn, r"zero diagonal") GeneralRKTimeStepper(eq, MUSCL(Float64, 1, 1, 2, NoLimiter(), RusanovFlux()), MOOD(), (), (src,), own)
    @test ts2.tableau.b == [0.25, 0.75] && iszero(ts2.tableau.a)
    dt = 0.1; z = -src.λ * dt
    ts(eq, pg, 0.0, dt)
    @test all(j -> isapprox(pg.rhos[j][1], 1 + z + z^2 / 2 + z^3 / 6; rtol = 1e-13), interior_indices(pg))
    # The Euler substep MOOD would check includes the source (not sized here: no MOOD criterion)
    HyperCloudCore.ensure_capacity!(ts.rho_sub, pg.meta.N)
    W = HyperCloudCore.form_substep!(ts, pg, 2, dt)
    Y, KE, KI = ts.Y_stages[1], ts.K_E_stages[1], ts.K_I_stages[1]
    @test all(j -> W[j] ≈ Y[j] + dt * (KE[j] + KI[j]) && KI[j][1] ≈ -src.λ * Y[j][1], interior_indices(pg))
end

@testset "Central/Upwind: the neighbour-count safeguard counts the intercept" begin
    # The divergence fit has basis_count(D, q) + 1 unknowns. On a jittered 1-D cloud with R = 2.5Δx, particles with
    # exactly 4 neighbours used to fit degree 4 through the Tikhonov retry: div(x) off by up to 8e-3 (and Central4
    # did not converge, errors 4.5e-3, 5.7e-3, 3.8e-3 for N = 40, 80, 160). Now they fit degree 3: exact for F = x.
    geom = get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true)
    dx   = 2 / 40
    pg   = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1; randomness = (0.2dx,), rng = MersenneTwister(1))
    N    = pg.meta.N
    eq   = make_advection_1d()
    for i in 1:N; pg.rhos[i] = State{1, Float64}(pg.core.positions[i][1]); end    # F = u = x: div F = 1
    ib   = HyperCloudCore.InteractionBuffer{1, 1, Float64}()
    update_size!(ib, length(pg.neighbor.indices))
    central = CentralDivergence(Float64, 1, 1, 4)
    fill!(view(pg.core.particle_orders, 1:N), 4)
    inner = [i for i in 1:N if abs(pg.core.positions[i][1]) < 0.7]               # away from the periodic seam of x
    @test any(i -> length(pg.neighbor.ranges[i]) == HyperCloudCore.basis_count(1, 4), inner)
    err = maximum(i -> abs(central(eq, i, pg.rhos[i], pg.neighbor.ranges[i], pg, pg.rhos, ib, EPD1())[1] - 1), inner)
    @test err < 1e-10
    # Upwind (degree 1, D + 1 = 2 unknowns): a single neighbour is not enough
    upwind = UpwindDivergence(Float64, 1, 1, 1, :Classic, RusanovFlux())
    fill!(view(pg.core.particle_orders, 1:N), 1)
    i  = inner[1]
    sl = first(pg.neighbor.ranges[i]):first(pg.neighbor.ranges[i])
    @test upwind(eq, i, pg.rhos[i], sl, pg, pg.rhos, ib, EPD1()) == zero(State{1, Float64})
end

@testset "MUSCL: a starved particle keeps the highest order its neighbours support" begin
    # MUSCL4 reconstructs with degree 3 (9 unknowns in 2-D). A particle with 5-8 neighbours went straight to
    # order 1 (no slope at all); now it reconstructs with degree 2 (order 3), with 2-4 neighbours degree 1.
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = true)
    dx   = 1 / 16
    pg   = ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(1.0, 1.6dx), 1; randomness = (0.25dx, 0.25dx), rng = MersenneTwister(3))
    N    = pg.meta.N
    for i in 1:N; x = pg.core.positions[i]; pg.rhos[i] = State{1, Float64}(x[1] + 2x[2]); end    # linear: exact for degree ≥ 1
    m  = MUSCL(Float64, 2, 1, 4, NoLimiter(), AlignedRusanovFlux())
    update_size!(m, N)
    ib = HyperCloudCore.InteractionBuffer{2, 1, Float64}()
    update_size!(ib, length(pg.neighbor.indices))
    fill!(view(pg.core.particle_orders, 1:N), 4)
    nnb(i) = length(pg.neighbor.ranges[i])
    inner = [i for i in 1:N if all(d -> 3dx < pg.core.positions[i][d] < 1 - 3dx, 1:2)]   # away from the periodic seam of x + 2y
    for i in inner; update_content!(m, i, pg.rhos[i], pg.neighbor.ranges[i], pg, pg.rhos, ib); end
    expected(n) = n >= 9 ? 4 : n >= 5 ? 3 : n >= 2 ? 2 : 1
    starved = filter(i -> 2 <= nnb(i) < 9, inner)
    @test !isempty(starved) && any(i -> nnb(i) >= 5, starved)
    @test all(i -> pg.core.particle_orders[i] == expected(nnb(i)), inner)
    @test all(i -> isapprox(m.gradients[i][1][1], 1; atol = 1e-8) && isapprox(m.gradients[i][2][1], 2; atol = 1e-8), starved)
end

@testset "Streamed divergence" begin
    # The streamed flux pass (cached per-slot weights, dynamic per-slot weights for dropped particles) gives the
    # divergence of the buffered fit up to round-off, with both fluxes, and never touches the slot buffers.
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = true)
    dx   = 1 / 24
    pg   = ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(1.0, 3.5dx), 2; randomness = (0.2dx, 0.2dx), rng = MersenneTwister(21))
    N    = pg.meta.N
    for i in 1:N
        x = pg.core.positions[i]
        pg.rhos[i] = State{2, Float64}(exp(-sum(abs2, x - Space{2, Float64}(0.5, 0.5)) / 0.05), sin(2pi * x[1]) * cos(2pi * x[2]))
    end
    eq = LinearAdvection(SVector(SMatrix{2, 2, Float64}(1.0, 0.0, 0.0, 0.5), SMatrix{2, 2, Float64}(0.7, 0.0, 0.0, -0.3)), Float64)
    function rhs(flux, streamed, drop)
        m  = MUSCL(Float64, 2, 2, 3, NoLimiter(), flux; streamed = streamed)
        ts = GeneralRKTimeStepper(eq, m, MOOD(), RK3_SSP_Tableau(Float64))
        HyperCloudCore.update_size!(ts, N, length(pg.neighbor.indices), false); HyperCloudCore.update_size!(m, N)
        fill!(view(pg.core.particle_orders, 1:N), 3)
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

@testset "MUSCL: the divergence cache holds the degree each particle uses" begin
    # The neighbour-count safeguard lowers the divergence degree on small stencils (1-D MUSCL3, R = 2.5Δx: degree 2
    # for every particle); the cache and its per-slot weights were built for the full degree (3) regardless, so the
    # cache was never hit and the CFL estimate summed weights the scheme does not use (1.500 vs 0.644). Now the
    # per-slot weights are those of the fit in use.
    for (o, R, used_expected) in ((3, 2.5, 2), (5, 2.5, 3), (5, 3.5, 4), (3, 3.5, 3))
        pg = make_pg_1d(40; range = R); nb = pg.neighbor
        m  = MUSCL(Float64, 1, 1, o, NoLimiter(), RusanovFlux())
        used = HyperCloudCore._divergence_index(Val(0), Val(1), o, length(nb.ranges[1])) - 1
        @test used == used_expected
        HyperCloudCore.prepare_geometry!(m, pg)
        S = zeros(Space{1, Float64}, length(nb.indices))
        HyperCloudCore._face_vectors!(S, nb.ranges[1], nb.distances, nb.weights, 1 / pg.meta.dx[1], 1.0, used, Val(1))
        @test all(k -> isapprox(m.dweights[k][1], S[k][1]; rtol = 1e-10, atol = 1e-10 / pg.meta.dx[1]), nb.ranges[1])
    end
end

@testset "CFL estimate: pair speeds, boundary states, cached weights" begin
    # A state-dependent wave speed 1 + 9u: a fast state on a boundary particle used to be invisible (boundary
    # particles are skipped), now it limits the step of its interior neighbours through the pair speed.
    struct FastWhereHigh <: HyperbolicPDE{1, 1, Float64} end
    HyperCloudCore.max_eigenvalue(::FastWhereHigh, U::State{1, Float64}, d::Int) = 1 + 9 * U[1]
    Nc = 60; dxc = 2.0 / Nc
    pgc = ParticleGrid(get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = false,
                                              bc_map = Dict{Int, AbstractBoundaryCondition}(1 => FixedDirichlet())),
                       (dxc,), ExponentialWeightFunction(1.0, 2.5dxc), 1)
    eqc = FastWhereHigh()
    interp = MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux())
    fill!(view(pgc.rhos, 1:pgc.meta.N), zero(State{1, Float64}))
    tstep(pg, eq) = get_time_step(pg, eq, interp; cached_weights = false)    # the assembled fit (no geometry caches)
    dt_slow = tstep(pgc, eqc)
    # the boundary particle next to the interior (the outermost ones are beyond the cutoff of every interior particle)
    b = findlast(i -> pgc.core.is_boundary[i] && pgc.core.positions[i][1] < 0, 1:pgc.meta.N)
    pgc.rhos[b] = State{1, Float64}(1.0)
    dt_b = tstep(pgc, eqc)
    @test dt_b < 0.6 * dt_slow && dt_b >= 0.99 * dt_slow / 10
    pgc.rhos[b] = zero(State{1, Float64})
    i_fast = findfirst(i -> !pgc.core.is_boundary[i] && abs(pgc.core.positions[i][1]) < dxc, 1:pgc.meta.N)
    pgc.rhos[i_fast] = State{1, Float64}(1.0)
    dt_i = tstep(pgc, eqc)
    @test 0.99 * dt_slow / 10 <= dt_i <= dt_slow
    # With the divergence cache the cached per-slot weights are used (no factorisation): `get_time_step` refreshes
    # the cache itself, the kernel equals the host sum over the weights
    @test HyperCloudCore._cfl_slot_weights(interp, pgc) === nothing
    dt_c = get_time_step(pgc, eqc, interp)
    dw = HyperCloudCore._cfl_slot_weights(interp, pgc)
    @test dw !== nothing
    nbc = pgc.neighbor
    dt_host = minimum(begin
            lam_i = 1 + 9 * pgc.rhos[i][1]
            s = sum(abs(dw[k][1]) * max(lam_i, 1 + 9 * pgc.rhos[nbc.indices[k]][1]) for k in nbc.ranges[i])
            s > floatmin() ? 1 / s : Inf
        end for i in 1:pgc.meta.N if !pgc.core.is_boundary[i])
    @test isapprox(dt_c, dt_host; rtol = 1e-12)
    # They are the weights of the fit in use: with R = 2.5Δx (4 neighbours) the safeguard lowers MUSCL3's divergence
    # to degree 2 (Σ|c|Δx = 0.644), while the assembled fit is the full degree 3 (1.500), a 2.3× smaller step
    @test 2 * dt_i <= dt_c <= 2.6 * dt_i
    @test get_time_step(pgc, eqc, interp; cached_weights = false) == dt_i     # the opt-out assembles the fit as before
    @test (@allocated get_time_step(pgc, eqc, interp)) < 8192
    # No particle restricts the step (zero wave speeds): Inf, without a warning (a field at rest; drivers decide);
    # very slow waves still restrict it (the former absolute cut-off 1e-14 of Σ|c|λ gave Inf for λ ≲ 1e-15)
    struct StandStill <: HyperbolicPDE{1, 1, Float64} end
    HyperCloudCore.max_eigenvalue(::StandStill, U::State{1, Float64}, d::Int) = 0.0
    @test (@test_logs get_time_step(pgc, StandStill(), interp)) == Inf
    struct Creeping <: HyperbolicPDE{1, 1, Float64} end
    HyperCloudCore.max_eigenvalue(::Creeping, U::State{1, Float64}, d::Int) = 1e-20
    struct UnitSpeed <: HyperbolicPDE{1, 1, Float64} end
    HyperCloudCore.max_eigenvalue(::UnitSpeed, U::State{1, Float64}, d::Int) = 1.0
    @test get_time_step(pgc, Creeping(), interp) ≈ 1e20 * get_time_step(pgc, UnitSpeed(), interp) rtol = 1e-12
end

@testset "Limiters, WENO weights and the upwind speed are scale invariant" begin
    # The round-off thresholds were absolute (1e-12, 1e-14): a box of height 1e-15 was left unlimited (Venkatakrishnan
    # increments below 1e-12, Minmod slopes below 1e-12 taken as r = 1) and WENO blended it half/half, while the same
    # box of height 1 was limited. Relative to the local data a power-of-two scale now gives bitwise scaled results.
    eq  = make_advection_1d()
    box = x -> abs(x) < 0.3 ? 1.0 : 0.0
    sc  = 2.0^-50
    @testset "$name" for (name, mk) in (("MUSCL3 VK", () -> MUSCL(Float64, 1, 1, 3, VenkatakrishnanLimiter(:hard), RusanovFlux())),
                                        ("MUSCL2 Minmod", () -> MUSCL(Float64, 1, 1, 2, MinmodLimiter(:hard), RusanovFlux())),
                                        ("WENO2", () -> WENO(Float64, 1, 1, 2)))
        function run_box(a)
            pg = set_ic!(make_pg_1d(60), x -> a * box(x))
            m  = mk()
            ts = GeneralRKTimeStepper(eq, m, MOOD(), RK2_SSP_Tableau(Float64))
            dt = 0.4 * 2 / 60
            for k in 0:9; ts(eq, pg, k * dt, dt); end
            return [pg.rhos[i][1] for i in 1:pg.meta.N]
        end
        u1 = run_box(1.0)
        us = run_box(sc)
        @test us == sc .* u1
        name == "WENO2" || @test minimum(u1) >= -1e-12 && maximum(u1) <= 1 + 1e-12
    end
    # UpwindFlux (Murman–Roe) for Burgers: the secant speed for states 1e-20 apart near 1e-20 (the absolute 1e-14
    # took the Jacobian at u_L, off by half), the Jacobian only where the secant has lost its digits
    struct TinyBurgers <: HyperbolicPDE{1, 1, Float64} end
    HyperCloudCore.flux(::TinyBurgers, U::State{1, Float64}) = Flux{1, 1, Float64}((State{1, Float64}(U[1]^2 / 2),))
    HyperCloudCore.velocity(::TinyBurgers, U::State{1, Float64}, d::Int) = SMatrix{1, 1, Float64}(U[1])
    eqb = TinyBurgers()
    upw(uL, uR) = begin
        fL = Flux{1, 1, Float64}((State{1, Float64}(uL),)); fR = Flux{1, 1, Float64}((State{1, Float64}(uR),))
        UpwindFlux()(fL, fR, flux(eqb, fL[1]), flux(eqb, fR[1]), eqb)[1][1]
    end
    @test upw(1e-20, 3e-20) ≈ (0.5e-40 + 4.5e-40) / 2 - 2e-20 * 2e-20 / 2 rtol = 1e-12    # s = 2e-20 (secant)
    @test upw(1e-20, 3e-20) == 2.0^-264 * upw(2.0^132 * 1e-20, 2.0^132 * 3e-20)
    # Which speed is taken: a flux F = u whose Jacobian is reported as 7 (only to tell the branches apart). States
    # 1e-10 apart near 1 are below √ε |u|: Jacobian (s = 7); 1e-6 apart: secant (s = 1)
    struct TwoSpeeds <: HyperbolicPDE{1, 1, Float64} end
    HyperCloudCore.flux(::TwoSpeeds, U::State{1, Float64}) = Flux{1, 1, Float64}((U,))
    HyperCloudCore.velocity(::TwoSpeeds, U::State{1, Float64}, d::Int) = SMatrix{1, 1, Float64}(7.0)
    upw2(uL, uR) = (fL = Flux{1, 1, Float64}((State{1, Float64}(uL),)); fR = Flux{1, 1, Float64}((State{1, Float64}(uR),));
                    UpwindFlux()(fL, fR, fL, fR, TwoSpeeds())[1][1])
    @test upw2(1.0, 1.0 + 1e-10) ≈ (2 + 1e-10) / 2 - 7 * 1e-10 / 2 rtol = 1e-15
    @test upw2(1.0, 1.0 + 1e-6) ≈ (2 + 1e-6) / 2 - 1e-6 / 2 rtol = 1e-15
    # Float32 WENO: the offset relative to the indicators normalised by max(u_i², IS_C) keeps the weights scale free
    # (an offset 1e-12 max(u_i², IS_C) underflows below |u| ≈ 1e-13 in Float32: deviations of 1.4e-4 to 4.2e-4 of the
    # peak at 2^-50). Bitwise while u² and the indicators stay normal numbers (box, wide Gaussian); within 5e-8 where
    # they underflow (Gaussian tails below 1e-21, the zero-slope points of a sine)
    pg32 = ParticleGrid(get_rectangular_domain(Float32, (-1.0,), (1.0,); is_periodic = true), (2f0 / 60,),
                        ExponentialWeightFunction(1f0, 2.5f0 * 2f0 / 60), 1)
    eq32 = LinearAdvection(((1f0,),), Float32)
    w32  = WENO(Float32, 1, 1, 2)
    ib32 = HyperCloudCore.InteractionBuffer{1, 1, Float32}()
    update_size!(ib32, length(pg32.neighbor.indices))
    fill!(view(pg32.core.particle_orders, 1:pg32.meta.N), 2)
    function weno_div(a, f)
        for i in 1:pg32.meta.N; pg32.rhos[i] = State{1, Float32}(a * Float32(f(pg32.core.positions[i][1]))); end
        return [w32(eq32, i, pg32.rhos[i], pg32.neighbor.ranges[i], pg32, pg32.rhos, ib32, EPD1())[1] for i in 1:pg32.meta.N]
    end
    s50 = Float32(2.0^-50)
    for f in (box, x -> exp(-x^2 / 0.2))
        @test weno_div(s50, f) == s50 .* weno_div(1f0, f)
    end
    for f in (x -> sinpi(x), x -> exp(-x^2 / 0.02))
        d1 = weno_div(1f0, f)
        @test maximum(abs.(weno_div(s50, f) ./ s50 .- d1)) <= 1f-6 * maximum(abs.(d1))
    end
end

@testset "Limiters keep Float32 reconstructions positive at near-vacuum minima" begin
    # u = 1e-9 at every 7th particle, 1 elsewhere: at such a minimum the gradient is at round-off level. A cut-off
    # relative to the whole local range (16 eps · 1) let it through and took 22 interface values below zero (down to
    # -4e-7); relative to u_i and the bound the increment points to (u_min ≈ 0), it is limited
    N  = 100
    pg = ParticleGrid(get_rectangular_domain(Float32, (-1.0,), (1.0,); is_periodic = true), (2f0 / N,),
                      ExponentialWeightFunction(1f0, 2.5f0 * 2f0 / N), 1)
    n  = pg.meta.N
    for i in 1:n
        pg.rhos[i] = State{1, Float32}(mod(round(Int, (pg.core.positions[i][1] + 1) * N / 2), 7) == 3 ? 1f-9 : 1f0)
    end
    nb = pg.neighbor
    ib = HyperCloudCore.InteractionBuffer{1, 1, Float32}()
    update_size!(ib, length(nb.indices))
    for lim in (BarthJespersenLimiter(:hard), VenkatakrishnanLimiter(:hard))
        m = MUSCL(Float32, 1, 1, 2, lim, RusanovFlux())
        update_size!(m, n)
        fill!(view(pg.core.particle_orders, 1:n), 2)
        for i in 1:n; update_content!(m, i, pg.rhos[i], nb.ranges[i], pg, pg.rhos, ib); end
        @test all(i -> all(k -> pg.rhos[i][1] + m.gradients[i][1][1] * 0.5f0 * nb.distances[k][1] >= 0, nb.ranges[i]), 1:n)
    end
end

@testset "Limited reconstructions stay in the local bounds" begin
    # The interface (midpoint) reconstructions of a limited MUSCL3 lie within [min, max] of the neighbourhood, for
    # steps, kinks, narrow peaks and ramps; unlimited they overshoot (16 % on the box)
    eq = make_advection_1d()
    function max_overshoot(limiter, ic; order = 3, N = 60)
        pg = set_ic!(make_pg_1d(N), ic)
        m  = MUSCL(Float64, 1, 1, order, limiter, RusanovFlux())
        ts = GeneralRKTimeStepper(eq, m, MOOD(), RK3_SSP_Tableau(Float64))
        HyperCloudCore.update_size!(ts, N, length(pg.neighbor.indices), false); HyperCloudCore.update_size!(m, N)
        fill!(view(pg.core.particle_orders, 1:N), order)
        HyperCloudCore.compute_explicit_rhs!(ts, eq, pg, ts.K_E_stages[1], ts.K_X_stages[1], pg.rhos, 0.0, true)
        nb = pg.neighbor; worst = 0.0
        for i in 1:N
            lo, hi = extrema(push!([pg.rhos[nb.indices[k]][1] for k in nb.ranges[i]], pg.rhos[i][1]))
            for k in nb.ranges[i]
                p = HyperCloudCore.build_basis(Val(order - 1), 0.5 * nb.distances[k])
                r = pg.rhos[i][1] + sum(m.gradients[i][b][1] * p[b] for b in eachindex(p))
                worst = max(worst, r - hi, lo - r)
            end
        end
        return worst
    end
    ics = (x -> 0.5 * (1 + tanh(x / 0.04)), x -> abs(x) < 0.3 ? 1.0 : 0.0, x -> abs(x), x -> exp(-x^2 / 0.002),
           x -> x < 0 ? 0.0 : x < 0.2 ? 5x : 0.5)
    for lim in (BarthJespersenLimiter(:soft), VenkatakrishnanLimiter(:hard), MinmodLimiter(:hard)), ic in ics
        @test max_overshoot(lim, ic) <= 1e-12
    end
    @test max_overshoot(NoLimiter(), ics[2]) > 0.1
end

@testset "MOODPAD: admissibility only" begin
    # Scalar advection with the admissible set u ≥ 0. An unlimited MUSCL3 undershoots a box; with MOODPAD the
    # inadmissible candidates drop their order and the solution stays admissible, with no DMP involved. On a
    # smooth profile MOODPAD never triggers and reproduces the run without MOOD.
    struct PositiveAdvection <: HyperbolicPDE{1, 1, Float64} end
    HyperCloudCore.flux(::PositiveAdvection, U::State{1, Float64}) = Flux{1, 1, Float64}((U,))
    HyperCloudCore.max_eigenvalue(::PositiveAdvection, U::State{1, Float64}, d::Int) = 1.0
    HyperCloudCore.velocity(::PositiveAdvection, U::State{1, Float64}, d::Int) = SMatrix{1, 1, Float64}(1.0)
    HyperCloudCore.prim2cons(::PositiveAdvection, U::State) = U
    HyperCloudCore.cons2prim(::PositiveAdvection, U::State) = U
    HyperCloudCore.is_admissible(::PositiveAdvection, U::State{1, Float64}) = U[1] >= 0
    eqp = PositiveAdvection()
    function run_pad(ic, mood, steps)
        pg = set_ic!(make_pg_1d(100), ic)
        m  = MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux())
        ts = GeneralRKTimeStepper(eqp, m, mood, RK3_SSP_Tableau(Float64))
        drops = 0; umin = Inf
        for k in 1:steps
            dt = 0.4 * get_time_step(pg, eqp, m)
            ts(eqp, pg, (k - 1) * dt, dt)
            drops += count(<(3), view(pg.core.particle_orders, 1:pg.meta.N))
            umin = min(umin, minimum(pg.rhos[i][1] for i in 1:pg.meta.N))
        end
        return [pg.rhos[i][1] for i in 1:pg.meta.N], umin, drops
    end
    box = x -> abs(x) < 0.3 ? 1.0 : 0.0
    _, umin_none, _ = run_pad(box, MOOD(), 30)
    _, umin_pad, drops_pad = run_pad(box, MOOD(EPD1(), MOODPAD()), 30)
    @test umin_none < -1e-3                      # the unlimited scheme undershoots
    @test umin_pad >= -1e-12 && drops_pad > 0    # PAD alone keeps the solution admissible
    gauss = x -> 1 + exp(-x^2 / 0.02)           # strictly positive: the unlimited scheme's small undershoots stay admissible
    u_none, _, _ = run_pad(gauss, MOOD(), 20)
    u_pad, _, drops_smooth = run_pad(gauss, MOOD(EPD1(), MOODPAD()), 20)
    @test drops_smooth == 0 && all(isapprox.(u_none, u_pad; atol = 1e-14))
    # Above the positivity limit of the first-order update a candidate at order 1 stays inadmissible: MOOD accepts it
    # (it cannot drop further), counts it and warns once per stepper; admissible steps count zero
    pg = set_ic!(make_pg_1d(100), box)
    m  = MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eqp, m, MOOD(EPD1(), MOODPAD()), RK3_SSP_Tableau(Float64))
    dt = get_time_step(pg, eqp, m)
    @test_logs ts(eqp, pg, 0.0, 0.4dt)
    @test mood_inadmissible(ts) == 0
    @test_logs (:warn, r"inadmissible candidate") ts(eqp, pg, 0.4dt, 3dt)
    n_bad = mood_inadmissible(ts)
    @test n_bad > 0                       # (the substeps are checked: the RK3 combination itself may be positive again)
    @test_logs ts(eqp, pg, 3.4dt, 0.4dt)
    @test build_mood_criterion(Val(:PAD), Dict{Symbol, Any}(), Dict{Symbol, Any}()) isa MOODPAD
    @test MOOD(EPD1(), MOODPAD())(nothing, 1, State{1, Float64}(0.0), 1:0, State{1, Float64}(-1.0), nothing, nothing) == false
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
        # Stepper: rho_n, rho_cand, 3 stages × (Y, K_E, K_I) = 11 states, plus the Euler substep MOOD checks with an SSP
        # tableau (rho_sub) if MOOD is on; 5 Bool masks, 1 Int snapshot; per slot df_flux, df_scratch, mask. No position
        # buffers on a static grid.
        @test memory_estimate(pg, ts).per_particle - g.per_particle ==
              (11 + HyperCloudCore._checks_substeps(ts)) * S + 5 + sizeof(Int)
        @test memory_estimate(pg, ts).per_slot - g.per_slot == F + S + 1
        # MUSCL3: gradients SVector{2, State} + the reconstruction cache (SMatrix{2,2} + Bool) + the divergence
        # cache (degree 3 with a constant term: SMatrix{4,4} + Bool)
        @test memory_estimate(pg, interp).per_particle - g.per_particle ==
              2S + sizeof(SMatrix{2, 2, Float64, 4}) + 1 + sizeof(SMatrix{4, 4, Float64, 16}) + 1
        @test memory_estimate(pg, interp).per_slot == g.per_slot + sizeof(Space{1, Float64})   # the per-slot divergence weights (CFL, aligned flux, streaming)
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

    @testset "No insertion between an interior particle and a ghost" begin
        # The interior compresses towards 0, the ghosts stay: a gap opens at the right wall. The insertion used to
        # put an interior-flagged particle at x = 1.001, outside the domain.
        dx   = 0.01
        geom = get_rectangular_domain(Float64, (0.0,), (1.0,))
        pg   = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1)
        for i in 1:pg.meta.N
            pg.rhos[i] = State{1, Float64}(1.0)
            pg.core.is_boundary[i] || (pg.core.positions[i] = 0.992 * pg.core.positions[i])
        end
        pg.neighbor(pg)
        update_volumes!(pg, ShepardVolumes())
        adapt_particles!(pg, ad, kmax)
        @test all(i -> pg.core.is_boundary[i] || geom.is_interior(pg.core.positions[i]), 1:pg.meta.N)
        @test HyperCloudCore._has_adapt_candidates(pg, ad) == false       # the kernel pre-check agrees
    end

    @testset "A CapacityError leaves the grid untouched" begin
        pg = set_ic!(make_pg_1d(40), ic)
        punch_hole!(pg, 10)
        N  = pg.meta.N
        reserve!(pg; limit = memory_estimate(pg).bytes + 1, tolerance = 0)   # exactly N particles
        U0, V0, m0 = copy(view(pg.rhos, 1:N)), copy(view(pg.core.volumes, 1:N)), total_mass(pg)
        @test_throws CapacityError adapt_particles!(pg, ad, kmax)
        @test pg.meta.N == N && view(pg.rhos, 1:N) == U0 && view(pg.core.volumes, 1:N) == V0 && total_mass(pg) == m0
    end

    @testset "A CapacityError of the neighbour slots also leaves the grid untouched" begin
        # One gap of 1.7Δx; room for 51 particles but only for the current slots per particle (200 slots): the
        # insertion needs 204, which used to throw in the neighbour search after the topology change.
        function gap_grid()
            geom = get_rectangular_domain(Float64, (0.0,), (1.0,); is_periodic = true)
            dx   = 0.02
            g    = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1)   # not `pg`: the closure
            n    = g.meta.N                                                               # would assign the outer one
            for i in 1:n; g.rhos[i] = State{1, Float64}(sin(2π * g.core.positions[i][1])); end
            order = sortperm([g.core.positions[i][1] for i in 1:n])
            s = (1 - 1.7dx) / (n - 1)
            for (k, i) in enumerate(order); g.core.positions[i] = Space{1, Float64}(0.001 + (k - 1) * s); end
            g.neighbor(g); update_volumes!(g, ShepardVolumes())
            return g
        end
        pg  = gap_grid()
        N   = pg.meta.N
        # the slot count of the new topology is predicted exactly (here, without a reservation)
        evs = HyperCloudCore._detect_events(pg, ad, kmax)
        tot, mx = HyperCloudCore._slots_after_events(pg, evs)
        pg2 = gap_grid()
        adapt_particles!(pg2, ad, kmax)
        @test pg2.meta.N == N + 1
        @test tot == sum(i -> length(pg2.neighbor.ranges[i]), 1:pg2.meta.N) && mx == pg2.meta.max_nb
        spp = slots_per_particle(pg)
        est = memory_estimate(pg; particles = 0, slots = 0)
        reserve!(pg; limit = est.fixed + ceil(Int, (N + 1.5) * (est.per_particle + spp * est.per_slot)), tolerance = 0,
                 slots_per_particle = spp)
        U0, V0, X0 = copy(view(pg.rhos, 1:N)), copy(view(pg.core.volumes, 1:N)), copy(view(pg.core.positions, 1:N))
        @test_throws CapacityError adapt_particles!(pg, ad, kmax)
        @test pg.meta.N == N && view(pg.rhos, 1:N) == U0 && view(pg.core.volumes, 1:N) == V0 && view(pg.core.positions, 1:N) == X0
        @test maximum(i -> last(pg.neighbor.ranges[i]), 1:N) <= length(pg.neighbor.indices)
    end

    @testset "Events next to a wall leave the boundary particles alone" begin
        # A merge with the wall particle of a FixedDirichlet box: the remap used to write the ghosts too (0.95 →
        # 0.9529 at x = -0.05), which changed the Dirichlet data for good. Ghost states and volumes stay now, and
        # Σ V U over the interior is exact.
        bc   = Dict{Int, AbstractBoundaryCondition}(1 => FixedDirichlet())
        geom = get_rectangular_domain(Float64, (0.0,), (1.0,); bc_map = bc)
        dx   = 0.05
        pg   = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1)
        N    = pg.meta.N
        for i in 1:N; pg.rhos[i] = State{1, Float64}(1 + pg.core.positions[i][1]); end
        i5 = argmin(i -> abs(pg.core.positions[i][1] - 0.05), 1:N)
        pg.core.positions[i5] = Space{1, Float64}(0.015); pg.rhos[i5] = State{1, Float64}(1.015)
        pg.neighbor(pg); update_volumes!(pg, ShepardVolumes())
        ghost(i) = pg.core.is_boundary[i]
        g0 = Dict(pg.core.positions[i] => (pg.rhos[i], pg.core.volumes[i]) for i in 1:N if ghost(i))
        content(g) = sum(i -> pg.core.volumes[i] * pg.rhos[i][1], filter(i -> ghost(i) == g, 1:pg.meta.N))
        c_in, c_gh = content(false), content(true)
        rep = adapt_particles!(pg, ad, kmax)
        @test rep.removed == 2 && rep.inserted == 1
        @test Dict(pg.core.positions[i] => (pg.rhos[i], pg.core.volumes[i]) for i in 1:pg.meta.N if ghost(i)) == g0
        @test abs(content(false) - c_in) < 1e-14 && content(true) == c_gh
        @test all(i -> ghost(i) || 1 - 1e-12 <= pg.rhos[i][1] <= 2 + 1e-12, 1:pg.meta.N)
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
