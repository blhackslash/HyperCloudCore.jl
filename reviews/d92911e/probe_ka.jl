# KA CPU (PoCL) vs Polyester for the kernels this diff touches: Central/Upwind with the new safeguard on jittered
# clouds (the lowered-degree path), the Venkatakrishnan limiter in MUSCL (Float64 and Float32, discontinuous data,
# large ratios), and the adaptivity pre-check on a jittered box with walls.
include(joinpath(@__DIR__, "header.jl"))

function grid(D, T; n, R = 2.5, jit = 0.2, per = true, seed = 1)
    geom = get_rectangular_domain(T, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D); is_periodic = per)
    dx = T(1 / n)
    ParticleGrid(geom, ntuple(_ -> dx, D), ExponentialWeightFunction(one(T), T(R) * dx), 1;
                 randomness = ntuple(_ -> T(jit) * dx, D), rng = MersenneTwister(seed))
end
function ic!(pg, f)
    T = eltype(pg.meta.dx)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, T}(T(f(pg.core.positions[i]))); end
    pg
end

function run(backend, D, T, mk, f; n, steps = 3, R = 2.5, tab = RK3_SSP_Tableau)
    set_backend!(backend)
    try
        pg = ic!(grid(D, T; n, R), f)
        eq = LinearAdvection(ntuple(d -> (T(d == 1 ? 1 : 0.5),), D), T)
        sc = mk(T)
        ts = GeneralRKTimeStepper(eq, sc, MOOD(), tab(T))
        dt = T(0.2 / n)
        for k in 0:(steps - 1); ts(eq, pg, k * dt, dt); end
        return [pg.rhos[i][1] for i in 1:pg.meta.N]
    finally
        set_backend!(PolyesterBackend())
    end
end

ka = KABackend(HC.KA.CPU(); workgroupsize = 32)
box(x) = all(abs.(x .- 0.5) .< 0.2) ? 1e7 : 1.0                         # large jump: VK ratios up to ~1e7..1e19
smooth(x) = sinpi(2x[1]) * (length(x) > 1 ? cospi(2x[2]) : 1.0)
cases = Any[
    ("Central4 1-D jitter", 1, x -> CentralDivergence(x, 1, 1, 4), smooth, 40),
    ("Central3 2-D jitter", 2, x -> CentralDivergence(x, 2, 1, 3), smooth, 12),
    ("Upwind Classic 1-D", 1, x -> UpwindDivergence(x, 1, 1, 1, :Classic, RusanovFlux()), smooth, 40),
    ("MUSCL3 VK 2-D box", 2, x -> MUSCL(x, 2, 1, 3, VenkatakrishnanLimiter(:hard), RusanovFlux()), box, 12),
    ("MUSCL2 VK 1-D box", 1, x -> MUSCL(x, 1, 1, 2, VenkatakrishnanLimiter(:hard), RusanovFlux()), box, 40),
]
@testset "KA vs Polyester: $name $T" for (name, D, mk, f, n) in cases, T in (Float64, Float32)
    ref = run(PolyesterBackend(), D, T, mk, f; n)
    dev = run(ka, D, T, mk, f; n)
    scale = maximum(abs, ref)
    err = maximum(abs.(dev .- ref)) / scale
    println(rpad("$name $T", 32), " max rel diff KA vs Polyester = ", err, ", finite: ", all(isfinite, dev))
    @test all(isfinite, ref) && all(isfinite, dev)
    @test err < 1000eps(T)
end

@testset "KA adaptivity pre-check, walls, $T" for T in (Float64, Float32)
    bc = Dict{Int, AbstractBoundaryCondition}(t => OutflowBC() for t in 1:4)
    geom = get_rectangular_domain(T, (0.0, 0.0), (1.0, 1.0); bc_map = bc)
    flags = Bool[]
    for backend in (PolyesterBackend(), ka)
        set_backend!(backend)
        try
            pg = ParticleGrid(geom, (T(0.05), T(0.05)), ExponentialWeightFunction(one(T), T(0.125)), 1;
                              randomness = (T(0.45 * 0.05), T(0.45 * 0.05)), rng = MersenneTwister(4))
            push!(flags, HC._has_adapt_candidates(pg, ParticleAdaptivity(T)))
        finally
            set_backend!(PolyesterBackend())
        end
    end
    @test flags[1] == flags[2] == true
end

# Venkatakrishnan φ: new form vs textbook (in BigFloat), monotonicity, Float32 extremes
vk(r) = HC.limiter_phi(VenkatakrishnanLimiter(:hard), r)
textbook(r) = (r_ = max(big(0), big(r)); Float64((r_^2 + 2r_) / (r_^2 + r_ + 2)))
@testset "VK φ: accuracy and the bound property, $T" for T in (Float64, Float32)
    # The textbook φ itself is not monotone and exceeds 1 for r > 2 (max ≈ 1.094 at r = 2 + 2√2); what matters for
    # the limiter is φ(r) ≤ r (the limited increment stays within the bounds) and agreement with the textbook form.
    rs = sort(unique(vcat(T[0, 1, prevfloat(one(T)), nextfloat(one(T))], T.(exp10.(range(-8, 30; length = 20001))))))
    filter!(isfinite, rs)
    φ  = vk.(rs)
    rel = maximum(abs.(Float64.(φ) .- textbook.(rs)) ./ max.(textbook.(rs), floatmin(Float64)))
    println("VK $T: max rel error vs BigFloat = $(rel) (= $(rel / eps(T)) eps), φ ≤ r: $(all(φ .<= rs .* (1 + 4eps(T)))), max φ = $(maximum(φ)), ",
            "around r = 1: ", (vk(prevfloat(one(T))), vk(one(T)), vk(nextfloat(one(T)))))
    @test rel < 4eps(T) && all(φ .<= rs .* (1 + 4eps(T))) && all(>=(0), φ)
    @test vk(T(Inf)) == 1 && vk(-T(Inf)) == 0 && isnan(vk(T(NaN))) && vk(-zero(T)) == 0
    @test vk(floatmax(T)) == 1
end
