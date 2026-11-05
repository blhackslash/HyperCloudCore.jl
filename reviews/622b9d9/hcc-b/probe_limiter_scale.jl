# Probe: the BJ/VK limiters skip interfaces with |reconstructed increment| <= 1e-12 (absolute, Limiter.jl:72), so a
# state of small magnitude is not limited at all. Same box, scaled by 1, 1e-9, 1e-13: relative overshoot.
include(joinpath(@__DIR__, "probe_limiters.jl"))
@testset "limiter scale invariance" begin
    for lim in (BarthJespersenLimiter(:hard), VenkatakrishnanLimiter(:hard)), s in (1.0, 1e-9, 1e-13)
        w = max_overshoot(lim; ic = x -> s * (abs(x) < 0.3 ? 1.0 : 0.0))
        @printf("%-30s scale %.0e: relative overshoot %.3e\n", string(typeof(lim)), s, w / s)
        @test w / s <= 1e-10
    end
end
