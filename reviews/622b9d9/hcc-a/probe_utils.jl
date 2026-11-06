# Probe: branchless math_min/math_max (src/Utils.jl) and their use in the slope limiters.
using Test, HyperCloudCore, StaticArrays
import HyperCloudCore: math_min, math_max, limiter_phi

@testset "probe_utils" begin
@testset "math_min / math_max edge cases" begin
    # Exact results expected from min/max
    @test math_min(1.0, Inf) == 1.0          # 0.5 * (1 + Inf - Inf) = NaN
    @test math_max(-Inf, 1.0) == 1.0
    @test math_min(1.0, 1e-17) == 1e-17      # cancellation: returns 0.0
    @test math_min(1.0f0, 1.0f8) == 1.0f0    # Float32: 1 + 1e8 == 1e8, returns 0
    @test math_max(3.0f38, 3.0f38) == 3.0f38 # a + b overflows to Inf
    @test math_min(0.1, 0.3) == 0.1          # rounding
    @test math_max(0.1, 0.3) == 0.3
end

@testset "limiters on a large ratio r (should be 1: no limiting)" begin
    for T in (Float64, Float32)
        r = T == Float64 ? 1e17 : 1.0f8
        @test limiter_phi(BarthJespersenLimiter(:hard), r) == one(T)
        @test limiter_phi(MinmodLimiter(:hard), r) == one(T)
        @test limiter_phi(SuperbeeLimiter(:hard), r) == T(2)
    end
end

end # probe_utils
# Show the actual values
for (a, b) in ((1.0, Inf), (1.0, 1e-17), (1.0f0, 1.0f8), (3.0f38, 3.0f38), (0.1, 0.3))
    println("math_min($a, $b) = ", math_min(a, b), "   math_max = ", math_max(a, b))
end
for r in (1.0f7, 2.0f7, 1.0f8)
    println("Float32 r = $r: BJ phi = ", limiter_phi(BarthJespersenLimiter(:hard), r),
            "  minmod phi = ", limiter_phi(MinmodLimiter(:hard), r))
end
