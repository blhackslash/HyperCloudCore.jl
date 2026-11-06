# Scale invariance beyond the test (Float32, 2-D, BJ soft, deep scales) and O(1) results (compare old/new outputs)
include(joinpath(@__DIR__, "common.jl"))
box1(x) = abs(x[1]) < 0.3 ? 1.0 : 0.0
gauss1(x) = exp(-x[1]^2 / 0.01)
box2(x) = (abs(x[1] - 0.5) < 0.2 && abs(x[2] - 0.5) < 0.2) ? 1.0 : 0.0
schemes1(T) = (("MUSCL3 VK", MUSCL(T, 1, 1, 3, VenkatakrishnanLimiter(:hard), RusanovFlux())),
               ("MUSCL3 BJsoft", MUSCL(T, 1, 1, 3, BarthJespersenLimiter(:soft), RusanovFlux())),
               ("MUSCL2 Minmod", MUSCL(T, 1, 1, 2, MinmodLimiter(:hard), RusanovFlux())),
               ("MUSCL2 Superbee", MUSCL(T, 1, 1, 2, SuperbeeLimiter(:hard), RusanovFlux())),
               ("WENO2", WENO(T, 1, 1, 2)))
function run!(pg, m, eq, T; steps = 10, jit = false)
    ts = GeneralRKTimeStepper(eq, m, MOOD(), RK2_SSP_Tableau(T))
    dt = T(0.4) * get_time_step(pg, eq, m; cached_weights = false)
    for k in 0:steps-1; ts(eq, pg, k * dt, dt); end
    [pg.rhos[i][1] for i in 1:pg.meta.N]
end
for T in (Float64, Float32)
    eq = Adv{1, T}(SVector(one(T)))
    for (name, m) in schemes1(T), (icn, ic) in (("box", box1), ("gauss", gauss1)), jit in (0.0, 0.3)
        u1 = run!(setic!(grid1d(60; T = T, jitter = jit), x -> T(ic(x))), m, eq, T)
        res = String[]
        for k in (-50, -100, 40)
            T == Float32 && k == -100 && continue
            s = T(2.0^k)
            us = run!(setic!(grid1d(60; T = T, jitter = jit), x -> s * T(ic(x))), m, eq, T)
            push!(res, "2^$k: " * (us == s .* u1 ? "bitwise" : "max rel dev $(Float64(maximum(abs.(us ./ s .- u1))))"))
        end
        println("$T $name $icn jitter=$jit: ", join(res, ", "), "; O(1) sum|u| = $(sum(abs, u1)), min $(minimum(u1)), max $(maximum(u1))")
    end
end
# 2-D VK hard, jittered
eq2 = Adv{2, Float64}(SVector(1.0, 0.5))
m2() = MUSCL(Float64, 2, 1, 3, VenkatakrishnanLimiter(:hard), RusanovFlux())
u1 = run!(setic!(grid2d(30; jitter = 0.25), box2), m2(), eq2, Float64; steps = 5)
us = run!(setic!(grid2d(30; jitter = 0.25), x -> 2.0^-50 * box2(x)), m2(), eq2, Float64; steps = 5)
println("2-D MUSCL3 VK jittered box 2^-50: ", us == 2.0^-50 .* u1 ? "bitwise" : "max rel dev $(maximum(abs.(us ./ 2.0^-50 .- u1)))", "; O(1) min $(minimum(u1)) max $(maximum(u1))")
