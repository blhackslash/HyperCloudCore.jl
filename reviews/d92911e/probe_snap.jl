# 6be4daf: wall snapping tolerance sqrt(eps)·dx + 8 eps (|canvas_min| + canvas length).
# (a) walls found at fine Float32 resolutions (the round-2 sweep), (b) can it snap a point that is not the wall
# point (two particles on one wall = duplicate positions)? relative tolerance tol/dx and minimum spacing,
# (c) offset domains, (d) jittered grids keep Ns+1 interior points on average and no duplicates.
include(joinpath(@__DIR__, "header.jl"))

function pts(T, lo, hi, Ns; jit = 0.0, seed = 1)
    dx   = (T(hi) - T(lo)) / Ns
    geom = get_rectangular_domain(T, (lo,), (hi,))
    cd   = HC.ComputationalDomain(geom, (dx,), T(2.5))
    pos, isb, _, _, _ = HC.get_points(cd, geom; nominal_dx = (dx,), interp_range_factor = T(2.5),
                                      randomness = (T(jit) * dx,), rng = MersenneTwister(seed))
    xs  = sort([p[1] for p in pos])
    tol = sqrt(eps(T)) * dx + 8 * eps(T) * (abs(cd.canvas_mins[1]) + abs(cd.canvas_maxs[1] - cd.canvas_mins[1]))
    gap = minimum(diff(Float64.(xs)))
    nin = count(!, isb)
    xin = [p[1] for (p, b) in zip(pos, isb) if !b]
    return (; nin, hit_lo = minimum(xin) == T(lo), hit_hi = maximum(xin) == T(hi), tol_dx = Float64(tol / dx),
            min_gap_dx = gap / Float64(dx))
end

@testset "(a) Float32 sweep [0,1] and [-1,1], Ns = 100:100:20000" begin
    lost = 0
    for (lo, hi) in ((0, 1), (-1, 1)), Ns in 100:100:20000
        r = pts(Float32, lo, hi, Ns)
        (r.nin == Ns + 1 && r.hit_lo && r.hit_hi) || (lost += 1)
    end
    println("(a) Float32 resolutions losing a wall point: $lost of 400")
    @test lost == 0
end

println("(b) fine / offset grids: tolerance in units of dx and the smallest spacing")
for (T, lo, hi, Ns) in ((Float32, 0, 1, 100_000), (Float32, 0, 1, 300_000), (Float32, 0, 1, 600_000), (Float32, 0, 1, 1_000_000),
                        (Float32, 10, 11, 1000), (Float32, 10, 11, 10_000), (Float32, 100, 101, 1000), (Float32, 100, 101, 10_000),
                        (Float32, 1000, 1001, 1000),
                        (Float64, 0, 1, 1_000_000), (Float64, 1e6, 1e6 + 1, 1000))
    r = pts(T, lo, hi, Ns)
    println(rpad("  $T [$lo, $hi] Ns=$Ns", 42), " interior $(r.nin) (want $(Ns + 1)), walls $(r.hit_lo)/$(r.hit_hi), tol = $(round(r.tol_dx, sigdigits = 3)) dx, min spacing $(round(r.min_gap_dx, sigdigits = 4)) dx")
end

@testset "(d) jittered: no duplicates, walls not snapped twice" begin
    for T in (Float64, Float32), Ns in (30, 1000, 7000), seed in 1:3
        r = pts(T, -1, 1, Ns; jit = 0.3, seed)
        @test r.min_gap_dx > 0.3
    end
end
