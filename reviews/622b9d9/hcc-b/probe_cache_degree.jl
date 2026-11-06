# Probe: MUSCL3 in 1-D with R = 2.5Δx (the pipeline's BASE_1D). The neighbour-count safeguard of
# `_divergence_index` lowers the divergence degree to 2 for every particle, but the cache (`div_weights`,
# `dweights`) and the CFL degree (`_divergence_degree`) are built for degree 3: the cache is never used and the
# CFL estimate sums the weights of a fit the scheme does not use.
include(joinpath(@__DIR__, "_prelude.jl"))
@testset "cache degree vs used degree" begin
    for (o, R) in ((3, 2.5), (5, 2.5), (5, 3.5), (3, 3.5))
        pg = make_pg_1d(40; range = R); nb = pg.neighbor
        m = MUSCL(Float64, 1, 1, o, NoLimiter(), RusanovFlux())
        used = HyperCloudCore._divergence_index(Val(0), Val(1), o, length(nb.ranges[1])) - 1
        cache = HyperCloudCore._divergence_degree(m)
        S = zeros(Space{1, Float64}, length(nb.indices))
        sumabs(q) = (HyperCloudCore._face_vectors!(S, nb.ranges[1], nb.distances, nb.weights, 1 / pg.meta.dx[1], 1.0, q, Val(1));
                     sum(k -> abs(S[k][1]), nb.ranges[1]))
        @printf("MUSCL%d R=%.1fdx: %d neighbours, degree used %d, cached/CFL degree %d; Σ|c_ij|·Δx used %.3f vs cached %.3f\n",
                o, R, length(nb.ranges[1]), used, cache, sumabs(used) * pg.meta.dx[1], sumabs(cache) * pg.meta.dx[1])
        @test used == cache
    end
end
