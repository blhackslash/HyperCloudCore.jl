# =========================================================================
# UNIT TESTS (group `core`): API contracts, utilities, grid buffers, MLS caches, boundary conditions,
# tableaus, the MOOD kernel and criteria, the time stepper.
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
    @test param2uvec((1, 2.0)) isa SVector{2, Float64}              # mixed tuples are promoted (was SVector{2, Real})
    @test ExponentialWeightFunction(1.0, 0.25f0) isa ExponentialWeightFunction{Float64}   # mixed types promoted
    @test ExponentialWeightFunction(1, 2) isa ExponentialWeightFunction{Float64} && InverseWeightFunction(1, 2)(1.0) ≈ 1   # integers: Float64
    @test InverseWeightFunction(1, 0.5) isa InverseWeightFunction{Float64}

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
    # Exact for infinite arguments and wide magnitudes (the former 0.5 (a + b ± |a - b|) gave NaN, 0, Inf, ...)
    @test math_min(1.0, Inf) == 1.0 && math_max(-Inf, 1.0) == 1.0 && math_max(1.0, Inf) == Inf
    @test math_min(1.0, 1e-17) == 1e-17 && math_min(1e20, 1.0) == 1.0 && math_max(-1e20, -1.0) == -1.0
    @test math_min(1.0f0, 1.0f8) == 1.0f0 && math_min(1f3, 1f-3) == 1f-3 && math_max(3.0f38, 3.0f38) == 3.0f38
    @test math_min(0.1, 0.3) == 0.1 && math_max(0.1, 0.3) == 0.3
    @test isnan(math_min(NaN, 1.0)) && isnan(math_max(1.0, NaN))
    @test math_min(SVector(1.0, 1f8), SVector(1e-17, 1.0)) == SVector(1e-17, 1.0)
    # ... so the limiters do not switch off at a large ratio r (unlimited: φ = 1, superbee 2)
    for (T, r) in ((Float64, 1e17), (Float32, 1.0f8))
        @test HyperCloudCore.limiter_phi(BarthJespersenLimiter(:hard), r) == one(T)
        @test HyperCloudCore.limiter_phi(MinmodLimiter(:hard), r) == one(T)
        @test HyperCloudCore.limiter_phi(SuperbeeLimiter(:hard), r) == T(2)
    end
    # a tiny increment inside wide bounds is not limited (Float32: φ was 0)
    f32(x) = State{1, Float32}(x)
    @test HyperCloudCore._update_phi(BarthJespersenLimiter(:hard), ones(State{1, Float32}), f32(1f-8), f32(0), f32(1), f32(-1))[1] == 1f0
    # Venkatakrishnan: r² overflowed Float32 from r ≈ 1.8e19 (NaN); the same values as the textbook form otherwise
    vk(r) = HyperCloudCore.limiter_phi(VenkatakrishnanLimiter(:hard), r)
    @test all(r -> isapprox(vk(r), (r^2 + 2r) / (r^2 + r + 2); rtol = 4eps()), (0.0, 1e-3, 0.5, 1.0, 1.5, 7.0, 1e3, 1e7))
    @test vk(-2.0) == 0 && vk(1f20) ≈ 1f0 && vk(Inf32) == 1f0 && vk(1f20) ≈ Float32(vk(1e20))
    @test !isnan(HyperCloudCore._update_phi(VenkatakrishnanLimiter(:hard), ones(State{1, Float32}), f32(1f-11), f32(0), f32(1f9), f32(-1f9))[1])
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
              pg.reorder.permutation,
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

    @testset "Wall lattice points are interior on both sides" begin
        # canvas_mins + (I - 1) dx missed `maxs` by an ulp for many spacings: the right wall point became a ghost
        for T in (Float64, Float32), (lo, hi, Ns) in ((-1, 1, 20), (-1, 1, 22), (-1, 1, 25), (-1, 1, 30), (-1, 1, 46), (0, 0.7, 10))
            dx   = (T(hi) - T(lo)) / Ns
            pg   = ParticleGrid(get_rectangular_domain(T, (lo,), (hi,)), (dx,), ExponentialWeightFunction(one(T), T(2.5) * dx), 1)
            xs   = sort([pg.core.positions[i][1] for i in 1:pg.meta.N if !pg.core.is_boundary[i]])
            @test length(xs) == Ns + 1 && xs[1] == T(lo) && xs[end] == T(hi)
            ghosts = [pg.core.positions[i][1] for i in 1:pg.meta.N if pg.core.is_boundary[i]]
            @test count(<(T(lo)), ghosts) == count(>(T(hi)), ghosts)
        end
        # Float32 at fine resolutions: the round-off of the lattice coordinate grows with |x|/dx (lost from Ns ≈ 3900)
        for (lo, hi, Ns) in ((0, 1, 3900), (0, 1, 5900), (-1, 1, 3900), (-1, 1, 6500), (0, 1, 20000))
            dx   = (Float32(hi) - Float32(lo)) / Ns
            geom = get_rectangular_domain(Float32, (lo,), (hi,))
            cd   = HyperCloudCore.ComputationalDomain(geom, (dx,), 2.5f0)
            pos, isb, _, _, _ = HyperCloudCore.get_points(cd, geom; nominal_dx = (dx,), interp_range_factor = 2.5f0)
            xs   = [p[1] for (p, b) in zip(pos, isb) if !b]
            @test length(xs) == Ns + 1 && minimum(xs) == lo && maximum(xs) == hi
        end
        # Wall points are chosen by index: far from the origin in Float32 a distance tolerance grew to almost dx
        # and could put the wall point's neighbour on the wall as well (old code: 5003 and 10003 interior points, a
        # duplicated wall point). A wall at -1 in Float32 has its canvas minimum in the next binade: the index
        # tolerance needs that rounding (a fixed share of dx left the wall without a point).
        for (lo, hi, Ns) in ((61.6f0, 61.6f0 + 0.24f0, 5000), (-54.181f0, -54.181f0 + 0.39f0, 10_000), (-1f0, -0.7f0, 100_000))
            dx   = (Float32(hi) - Float32(lo)) / Ns
            geom = get_rectangular_domain(Float32, (lo,), (hi,))
            cd   = HyperCloudCore.ComputationalDomain(geom, (dx,), 2.5f0)
            pos, isb, _, _, _ = HyperCloudCore.get_points(cd, geom; nominal_dx = (dx,), interp_range_factor = 2.5f0)
            xs   = sort([p[1] for (p, b) in zip(pos, isb) if !b])
            @test length(xs) == Ns + 1 && xs[1] == lo && xs[end] == hi
            @test minimum(diff(xs)) > dx / 2                        # no duplicated wall point
        end
        # A wall between two lattice points (extent not a multiple of dx) is not snapped to, also within 1 % of a
        # multiple (only round-off is): the points stay on the lattice
        for dxv in (0.3, 1 / 30.009, 1 / 29.991)
            geom = get_rectangular_domain(Float64, (0.0,), (1.0,))
            cd   = HyperCloudCore.ComputationalDomain(geom, (dxv,), 2.5)
            pos, isb, _, _, _ = HyperCloudCore.get_points(cd, geom; nominal_dx = (dxv,), interp_range_factor = 2.5)
            xs   = sort([p[1] for (p, b) in zip(pos, isb) if !b])
            @test xs[1] == 0 && maximum(abs.(xs ./ dxv .- round.(xs ./ dxv))) < 1e-12 && xs[end] < 1
        end
        dx = 2 / 30                                                   # 2-D: no wall row or column lost
        pg = ParticleGrid(get_rectangular_domain(Float64, (-1.0, -1.0), (1.0, 1.0)), (dx, dx), ExponentialWeightFunction(1.0, 2.5dx), 1)
        @test pg.meta.N_interior == 31^2
    end

    @testset "Interior particles without a linear fit are reported" begin
        # 1-D, R = 1.2 dx with 20 % jitter: some particles have a single neighbour (frozen by Central/Upwind)
        geom = get_rectangular_domain(Float64, (0.0,), (1.0,); is_periodic = true)
        @test_logs (:warn, r"fewer than 2 neighbours") match_mode = :any ParticleGrid(geom, (0.025,),
            ExponentialWeightFunction(1.0, 1.2 * 0.025), 1; randomness = (0.2 * 0.025,), rng = MersenneTwister(3))
        @test_logs ParticleGrid(geom, (0.025,), ExponentialWeightFunction(1.0, 2.5 * 0.025), 1)    # normal radius: quiet
    end

    @testset "Anisotropic spacing: the ghost band is R thick on every axis" begin
        # Every axis used to get ceil(R / max dx) layers: on the finer axis the band was thinner than R (it reached
        # y = -0.03 for R = 0.05 with dx = (0.02, 0.01)), and wall particles there had truncated stencils.
        geom = get_rectangular_domain(Float64, (0.0, 0.0), (2.0, 1.0))
        for dx in ((0.02, 0.01), (0.1, 0.05))
            R  = 2.5 * maximum(dx)
            pg = ParticleGrid(geom, dx, ExponentialWeightFunction(1.0, R), 1)
            ghosts = [pg.core.positions[i] for i in 1:pg.meta.N if pg.core.is_boundary[i]]
            @test minimum(p -> p[2], ghosts) <= -R + 1e-12 && minimum(p -> p[1], ghosts) <= -R + 1e-12
            pos = pg.core.positions
            near(x) = argmin(i -> pg.core.is_boundary[i] ? Inf : sum(abs2, pos[i] - Space{2, Float64}(x)), 1:pg.meta.N)
            dx == (0.1, 0.05) && @test length(pg.neighbor.ranges[near((1.0, 0.0))]) == length(pg.neighbor.ranges[near((1.0, 0.5))])
        end
        # isotropic grids keep their layers
        pg = ParticleGrid(get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0)), (0.1, 0.1), ExponentialWeightFunction(1.0, 0.3), 1)
        @test pg.domain.canvas_mins ≈ Space{2, Float64}(-0.3, -0.3)
        # Complete stencils at every wall and corner, 2-D and 3-D: each interior particle of an unjittered lattice sees
        # the bulk count (R = 2.37 max dx avoids distance ties)
        function lattice_count(dx, R)
            K = ntuple(d -> floor(Int, R / dx[d]) + 1, length(dx))
            return count(I -> !all(==(0), Tuple(I)) && sum(abs2, ntuple(d -> I[d] * dx[d], length(dx))) <= R^2,
                         CartesianIndices(ntuple(d -> -K[d]:K[d], length(dx))))
        end
        for (mins, maxs, dx) in (((0.0, 0.0), (2.0, 1.0), (0.1, 0.05)), ((0.0, 0.0, 0.0), (1.0, 0.5, 0.5), (0.1, 0.05, 0.025)))
            R  = 2.37 * maximum(dx)
            pg = ParticleGrid(get_rectangular_domain(Float64, mins, maxs), dx, ExponentialWeightFunction(1.0, R), 1)
            nref = lattice_count(dx, R)
            @test all(i -> pg.core.is_boundary[i] || length(pg.neighbor.ranges[i]) == nref, 1:pg.meta.N)
        end
        # Float32: the layer count of the finer axis is ⌈R/dx_d⌉ = 6, not one more from round-off (the guard was a
        # Float64 factor that cannot pull a Float32 product back: 7 layers here). The spacings are given in Float64, so
        # their Float32 ratio is inexact.
        cd = HyperCloudCore.ComputationalDomain(get_rectangular_domain(Float32, (0.0, 0.0), (1.0, 1.0)), (1 / 53, 1 / 159), 2.0)
        @test round(Int, -cd.canvas_mins[2] * 159) == 6 && round(Int, -cd.canvas_mins[1] * 53) == 2
    end

    @testset "Boundary tags: periodic edges ignored, bc_map mismatches warned" begin
        bc   = Dict{Int, AbstractBoundaryCondition}(3 => OutflowBC(), 4 => OutflowBC())
        geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = (true, false), bc_map = bc)
        pg   = @test_logs ParticleGrid(geom, (0.05, 0.05), ExponentialWeightFunction(1.0, 0.125), 1)   # no warning
        ghosts = [i for i in 1:pg.meta.N if pg.core.is_boundary[i]]
        @test !isempty(ghosts)
        @test all(i -> pg.core.tags[i] == (pg.core.positions[i][2] < 0 ? 3 : 4), ghosts)   # 29 of 120 were 1 or 2
        # 1-D: the left wall carries tag 1, the right wall 2 (both carried 1 up to 0.7.9, so a tag-2 entry, e.g. an
        # outflow at the right end of a Sod tube, was dead); a map with only tag 1 leaves the right wall uncovered
        bc1 = Dict{Int, AbstractBoundaryCondition}(1 => FixedDirichlet(), 2 => OutflowBC())
        geom1 = get_rectangular_domain(Float64, (0.0,), (1.0,); bc_map = bc1)
        pg1 = @test_logs ParticleGrid(geom1, (0.1,), ExponentialWeightFunction(1.0, 0.25), 1)
        @test all(i -> !pg1.core.is_boundary[i] || pg1.core.tags[i] == (pg1.core.positions[i][1] < 0.5 ? 1 : 2), 1:pg1.meta.N)
        geom1b = get_rectangular_domain(Float64, (0.0,), (1.0,); bc_map = Dict{Int, AbstractBoundaryCondition}(1 => OutflowBC()))
        @test_logs (:warn, r"tag\(s\) \[2\] have no condition") ParticleGrid(geom1b, (0.1,), ExponentialWeightFunction(1.0, 0.25), 1)
        @test_logs ParticleGrid(geom1b, (0.1,), ExponentialWeightFunction(1.0, 0.25), 1)       # once per session
        # 3-D: faces 1/2 (x), 3/4 (y), 5/6 (z), the nearest face, the lower tag at edges and corners; periodic z: no 5, 6
        for per in (false, (false, false, true))
            g3  = get_rectangular_domain(Float64, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0); is_periodic = per)
            pg3 = ParticleGrid(g3, (0.125, 0.125, 0.125), ExponentialWeightFunction(1.0, 0.3), 1)
            function face(x)
                best, bd = 1, Inf
                for d in 1:3
                    per isa Tuple && per[d] && continue
                    abs(x[d]) < bd && (best = 2d - 1; bd = abs(x[d]))
                    abs(x[d] - 1) < bd && (best = 2d; bd = abs(x[d] - 1))
                end
                return best
            end
            gh = [i for i in 1:pg3.meta.N if pg3.core.is_boundary[i]]
            @test all(i -> pg3.core.tags[i] == face(pg3.core.positions[i]), gh)
            @test sort(unique(pg3.core.tags[gh])) == (per === false ? collect(1:6) : collect(1:4))
        end
        # 2-D box with only the left/right walls mapped: the bottom/top particles keep their state
        geom2 = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0);
                                       bc_map = Dict{Int, AbstractBoundaryCondition}(1 => OutflowBC(), 2 => OutflowBC()))
        @test_logs (:warn, r"tag\(s\) \[3, 4\] have no condition") ParticleGrid(geom2, (0.1, 0.1), ExponentialWeightFunction(1.0, 0.25), 1)
    end

    @testset "Reordering after buffer growth" begin
        # permute! needed buffers of length N: after any growth (adaptivity) it threw a BoundsError
        geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = true)
        pg   = ParticleGrid(geom, (0.1, 0.1), ExponentialWeightFunction(1.0, 0.25), 1)
        N    = pg.meta.N
        pos0 = pg.core.positions[1:N]                         # the reordered layout of the constructor
        f(x) = State{1, Float64}(x[1] + 10x[2])
        HyperCloudCore.ensure_capacity!(pg, N + 5)
        perm = shuffle(MersenneTwister(1), 1:N)
        # every per-particle array must follow its particle: tags carry the original index, the others a function of it
        for i in 1:N
            k = perm[i]
            pg.core.positions[i] = pos0[k]; pg.rhos[i] = f(pos0[k]); pg.core.tags[i] = k
            pg.core.volumes[i] = k / N; pg.core.particle_orders[i] = k; pg.core.is_boundary[i] = isodd(k)
            pg.curvatures[i] = f(pos0[k])
        end
        v0 = pg.meta.geometry_version
        pg.reorder(pg)
        @test pg.core.positions[1:N] == pos0
        @test all(i -> pg.rhos[i] == f(pg.core.positions[i]) && pos0[pg.core.tags[i]] == pg.core.positions[i], 1:N)
        @test all(i -> (k = pg.core.tags[i]; pg.core.volumes[i] == k / N && pg.core.particle_orders[i] == k &&
                       pg.core.is_boundary[i] == isodd(k) && pg.curvatures[i] == f(pos0[k])), 1:N)
        fill!(view(pg.core.is_boundary, 1:N), false)
        @test pg.meta.geometry_version > v0
        fill!(view(pg.core.tags, 1:N), 0)
        pg.neighbor(pg)
        @test all(i -> length(pg.neighbor.ranges[i]) > 0, 1:N)
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

@testset "Boundary conditions as kernels" begin
    N  = 60
    dx = 2.0 / N
    make_pg_bc(bc) = ParticleGrid(get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = false,
                                                         bc_map = Dict{Int, AbstractBoundaryCondition}(1 => bc, 2 => bc)),
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
    # Order conditions of the IMEX pairs (partitioned RK conditions: every combination of the explicit and
    # implicit weights, nodes and matrices), up to the claimed order, and the nodes are the row sums
    function imex_residuals(bt)
        A, At = Matrix(bt.a), Matrix(bt.a_t)
        bs, cs, As = (Vector(bt.b_t), Vector(bt.b)), (Vector(bt.c_t), Vector(bt.c)), (At, A)
        o1 = [sum(b) - 1 for b in bs]
        o2 = [b' * c - 1/2 for b in bs for c in cs]
        o3 = vcat([b' * (c .* c2) - 1/3 for b in bs for c in cs for c2 in cs],
                  [b' * (Am * c) - 1/6 for b in bs for Am in As for c in cs])
        nodes = max(maximum(abs, vec(sum(A; dims = 2)) - cs[2]), maximum(abs, vec(sum(At; dims = 2)) - cs[1]))
        return maximum(abs, o1), maximum(abs, o2), maximum(abs, o3), nodes
    end
    for (make, order) in ((IMEX_Euler_Tableau, 1), (IMEX_ARS222_Tableau, 2), (IMEX_ARS233_Tableau, 3),
                          (IMEX_PRSSP3_Tableau, 2), (IMEX_SSP2332_Tableau, 2)), T in (Float64, Float32)
        r1, r2, r3, nodes = imex_residuals(make(T))
        tol = 10eps(T)
        @test nodes < tol
        @test r1 < tol
        order >= 2 && @test r2 < tol
        order >= 3 && @test r3 < tol
        order == 2 && @test r3 > 1e-3                       # the claimed order is the actual one
    end
    @test num_stages(IMEX_ARS222_Tableau(Float64)) == 3 && iszero(IMEX_ARS222_Tableau(Float64).a[1, 1])
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

@testset "MOODu2: smooth extrema, plateaus and jumps (2-D)" begin
    # A 2-D periodic lattice with hand-set states and curvature coefficients: the criterion is tested on its own.
    n    = 20
    dx   = 1 / n
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = true)
    pg   = ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(1.0, 2.5dx), 2)
    N    = pg.meta.N
    muscl = MUSCL(Float64, 2, 2, 3, NoLimiter(), RusanovFlux())
    update_size!(muscl, N)
    fill!(pg.core.particle_orders, 3)
    pos(i) = pg.core.positions[i]
    nearest(c) = argmin(i -> sum(abs2, pos(i) - Space{2, Float64}(c)), 1:N)

    # Component fields with their second derivatives (∂xx, ∂yy); the cached coefficients D+1, D+2 are curvatures
    function set_fields!(f1, d1, f2, d2)
        for i in 1:N
            x = pos(i)
            pg.rhos[i] = State{2, Float64}(f1(x), f2(x))
            c1, c2 = d1(x), d2(x)
            muscl.gradients[i] = SVector{5, State{2, Float64}}(zero(State{2, Float64}), zero(State{2, Float64}),
                State{2, Float64}(c1[1], c2[1]), State{2, Float64}(c1[2], c2[2]), zero(State{2, Float64}))
        end
    end
    U = pg.rhos
    function flags(crit, p, cand)
        mood  = MOOD(EPD2(), crit)
        range = HyperCloudCore._global_range(crit, U, N)
        return HyperCloudCore._criterion_flags(mood, muscl, p, U[p], pg.neighbor.ranges[p], cand, pg, U, range)
    end
    local_extrema(p, m) = extrema(U[k][m] for k in (p, (pg.neighbor.indices[s] for s in pg.neighbor.ranges[p])...))

    δ = 1e-8
    w = 0.1                                                     # Gaussian width²: the stencil (2.5 dx) is well inside it
    c0 = (0.5, 0.5)
    gauss(x)   = exp(-((x[1] - c0[1])^2 + (x[2] - c0[2])^2) / w)
    f1(x)      = 1 - 0.5 * gauss(x)
    d1(x)      = (-0.5 * gauss(x) * (4 * (x[1] - c0[1])^2 / w^2 - 2 / w), -0.5 * gauss(x) * (4 * (x[2] - c0[2])^2 / w^2 - 2 / w))
    f2(x)      = sin(2π * x[1])                                 # inflection line x = 1/2 through the minimum of f1
    d2(x)      = (-(2π)^2 * sin(2π * x[1]), 0.0)

    @testset "Smooth minimum next to another component's inflection line" begin
        set_fields!(f1, d1, f2, d2)
        p = nearest(c0)
        lo1, _ = local_extrema(p, 1)
        cand = State{2, Float64}(lo1 - 10δ, U[p][2])           # the discrete minimum is undershot, component 2 fine
        @test flags(MOODu2(δ), p, cand) == false                # accepted: only component 1 is tested
        # The curvature test is still there: the same undershoot of component 2 at its inflection line is rejected
        lo2, _ = local_extrema(p, 2)
        @test flags(MOODu2(δ), p, State{2, Float64}(U[p][1], lo2 - 10δ)) == true
        # Below order 3 there are no curvature coefficients: plain DMP
        pg.core.particle_orders[p] = 2
        @test flags(MOODu2(δ), p, cand) == true
        pg.core.particle_orders[p] = 3
    end

    @testset "Plateau: a component that barely varies across the stencil" begin
        # Component 1: a large bump elsewhere (global range 0.5) and a ripple of amplitude 1e-6 around p, whose
        # curvature changes sign across the stencil; the candidate overshoots the ripple by 5δ.
        q = (0.1, 0.1)
        narrow(x) = exp(-((x[1] - c0[1])^2 + (x[2] - c0[2])^2) / 0.01)   # e^-32 at q: the bump does not reach it
        ripple(x) = 1e-6 * sin(20π * x[1] + 0.7)                 # phase: not zero on the lattice
        g1(x)  = 1 - 0.5 * narrow(x) + ripple(x)
        dg1(x) = (-1e-6 * (20π)^2 * sin(20π * x[1] + 0.7), 0.0)
        set_fields!(g1, dg1, x -> 0.0, x -> (0.0, 0.0))
        p = nearest(q)
        lo, hi = local_extrema(p, 1)
        @test hi - lo < 1e-5                                    # the stencil sees the ripple only
        cand = State{2, Float64}(hi + 5δ, 0.0)
        @test flags(MOODu2(δ; plateau = 0), p, cand) == true    # no plateau test: the ripple's curvature changes sign
        @test flags(MOODu2(δ; plateau = 1e-3), p, cand) == false   # local range 2e-6 < 1e-3 × global range 0.5
        pg.core.particle_orders[p] = 1                          # a plateau needs no curvature: also below order 3
        @test flags(MOODu2(δ; plateau = 1e-3), p, cand) == false
        pg.core.particle_orders[p] = 3
    end

    @testset "A jump is still flagged with the plateau test on" begin
        step(x)  = x[1] < 0.5 ? 0.0 : 1.0
        dstep(x) = (abs(x[1] - 0.5) < 2.5dx ? (x[1] < 0.5 ? 1 / dx^2 : -1 / dx^2) : 0.0, 0.0)   # Gibbs-like curvature
        set_fields!(step, dstep, x -> 0.0, x -> (0.0, 0.0))
        p = nearest((0.5 - dx / 2, 0.5))
        _, hi = local_extrema(p, 1)
        cand = State{2, Float64}(hi + 1e-3, 0.0)
        for crit in (MOODu2(δ; plateau = 0), MOODu2(δ), MOODu2(δ; plateau = 1e-2))
            @test flags(crit, p, cand) == true
        end
        # The corner of a jump on a lattice: the plateau next to it has exactly zero curvature, so the sign test
        # passes (c_min c_max = 0); only the magnitude test among the nearest neighbours sees the collapse.
        corner(x) = (0.5 <= x[1] < 0.5 + dx / 2 ? -1 / dx^2 : 0.0, 0.0)
        set_fields!(step, corner, x -> 0.0, x -> (0.0, 0.0))
        p = nearest((0.5 + dx, 0.5))
        _, hi = local_extrema(p, 1)
        cand = State{2, Float64}(hi + 1e-3, 0.0)
        @test flags(MOODu2(δ), p, cand) == true
        @test flags(MOODu2(δ; ratio = 0), p, cand) == false    # without it a 1-D box rings by 10 %
    end

    @testset "Float32: no Float64 arithmetic in the curvature test" begin
        pg32 = make_pg_1d(20; T = Float32)
        g32  = MUSCL(Float32, 1, 1, 3, NoLimiter(), RusanovFlux())
        c32  = MOODu2(1f-4)
        U32  = pg32.rhos
        rng  = HyperCloudCore._global_range(c32, U32, pg32.meta.N)
        argt = (typeof(c32), typeof(g32), Int, eltype(U32), typeof(pg32.neighbor.ranges[1]), eltype(U32), typeof(pg32),
                typeof(U32), typeof(rng))
        ci, rt = only(code_typed(HyperCloudCore._u2_fails, argt))
        @test rt === Bool
        @test !any(t -> Core.Compiler.widenconst(t) === Float64, ci.ssavaluetypes)   # was (1.5 h)², a Float64
    end

    @testset "Public functor and builder" begin
        set_fields!(f1, d1, f2, d2)
        p = nearest(c0)
        lo1, _ = local_extrema(p, 1)
        cand = State{2, Float64}(lo1 - 10δ, U[p][2])
        @test MOOD(EPD2(), MOODu2(δ))(muscl, p, U[p], pg.neighbor.ranges[p], cand, pg, U) == false
        ctx = Dict{Symbol, Any}(:T => Float64, :Grid => pg)
        c = build_mood_criterion(Val(:U2), Dict{Symbol, Any}(:delta_relax => 1e-4, :plateau => 1e-3, :curvature_ratio => 0.25), ctx)
        @test c isa MOODu2{Float64} && c.d ≈ 1e-4 * dx^2 && c.plateau == 1e-3 && c.ratio == 0.25
        c0_ = build_mood_criterion(Val(:U2), Dict{Symbol, Any}(:delta_relax => 1e-4), ctx)
        @test c0_.plateau == HyperCloudCore.U2_PLATEAU_DEFAULT == MOODu2(δ).plateau && c0_.ratio == 0.5 == MOODu2(δ).ratio
        @test HyperCloudCore._global_range(MOODu2(δ; plateau = 0), U, N) === nothing   # plateau off: no reduction
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
    # the extra launches of a step with a trigger. The aligned flux with `div_order = 3` takes the per-slot
    # dynamic fit for every particle (the safeguard lowers the degree at R = 2.5Δx).
    function step_alloc(n, flux = RusanovFlux(); kw...)
        pgn = set_ic!(make_pg_1d(n), x -> exp(-x^2 / 0.02))
        interp = MUSCL(Float64, 1, 1, 3, VenkatakrishnanLimiter(:hard), flux; kw...)
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
        small, large = step_alloc(N ÷ 10, AlignedRusanovFlux(); div_order = 3), step_alloc(N, AlignedRusanovFlux(); div_order = 3)
        @test large <= small + 8192
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
