# _dmp_repair! threshold: small volumes (Float32, V = 1e-6), a 10 % overshoot; large states (Float64, U ~ 1e8).
using HyperCloudCore, StaticArrays, Printf
const HC = HyperCloudCore
for (T, Vs, base) in ((Float32, 1f-6, 1f0), (Float32, 1f-3, 1f0), (Float64, 1e-12, 1.0), (Float64, 1.0, 1e8))
    U  = [State{1, T}(base * T(1.1)), State{1, T}(base * T(0.6)), State{1, T}(base * T(0.8))]
    V  = T[Vs, Vs, 2Vs]
    lo = fill(State{1, T}(base * T(0.5)), 3); hi = fill(State{1, T}(base * T(1.0)), 3)
    m0 = sum(V .* getindex.(U, 1))
    HC._dmp_repair!(U, V, lo, hi)
    m1 = sum(V .* getindex.(U, 1))
    @printf("%-7s V=%-8g U~%-6g: relative content change %+.2e, max U/base %.4f\n", T, Vs, base, (m1 - m0) / m0, maximum(u -> u[1], U) / base)
end
