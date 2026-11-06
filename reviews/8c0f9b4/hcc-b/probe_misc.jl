include(joinpath(@__DIR__, "header.jl"))
const HC = HyperCloudCore
@testset "Central/Upwind safeguard counts the intercept" begin
    # 1-D jittered cloud, R = 2.5Δx: particles with exactly B(1, q) neighbours get a singular intercept fit
    pg = grid1d(40; range = 2.5, jit = 0.2, seed = 1)
    nb = pg.neighbor
    df_flux = Vector{Flux{1, 1, Float64}}(undef, length(nb.indices)); scratch = Vector{State{1, Float64}}(undef, length(nb.indices))
    for k in eachindex(nb.indices); df_flux[k] = Flux{1, 1, Float64}((State{1, Float64}(nb.distances[k][1]),)); end   # F = x: div = 1
    counts = Dict{Int, Vector{Float64}}()
    for i in interior_indices(pg)
        sl = nb.ranges[i]; n = length(sl)
        for q in 1:4
            HC.basis_count(1, q) == n || continue                       # passes Central's check `num_nb >= B(D, q)` exactly
            d = HC.Interpolator{1, q, 1}()(sl, nb.distances, nb.weights, df_flux, scratch; scale = pg.meta.dx)
            push!(get!(counts, q, Float64[]), abs(d[1] - 1))
        end
    end
    for (q, e) in sort(collect(counts))
        println("degree $q with exactly B = $q neighbours: $(length(e)) particles, max |div(x) - 1| = ", maximum(e))
    end
    @test all(maximum(e) > 1e-3 for e in values(counts))   # demonstrates the inconsistency (expected to FAIL after a fix)
end
@testset "Venkatakrishnan phi in Float32 overflows for large r" begin
    v = HC.limiter_phi(VenkatakrishnanLimiter(:hard), 1f20)
    println("Venkat phi(1f20) = ", v, "  phi(1e20) = ", HC.limiter_phi(VenkatakrishnanLimiter(:hard), 1e20))
    f32(x) = State{1, Float32}(x)
    p = HC._update_phi(VenkatakrishnanLimiter(:hard), ones(State{1, Float32}), f32(1f-11), f32(0), f32(1f9), f32(-1f9))
    println("_update_phi Float32, increment 1e-11, bounds ±1e9: ", p)
    @test isnan(v)
end
