# Float32 ghost-layer guard: layers on the finer axis vs ⌈R/dx_d⌉ (exact ratio), and the test's configuration.
using HyperCloudCore
const HC = HyperCloudCore
dxm = 1.0f0 / 197; dx32 = (dxm, dxm / 6)
println("test config: max_dx/dx = ", dxm / dx32[2], ", 3.5f0 * ratio = ", 3.5f0 * (dxm / dx32[2]))
cd = HC.ComputationalDomain(get_rectangular_domain(Float32, (0.0, 0.0), (1.0, 1.0)), dx32, 3.5f0)
println("test config: layers = ", round(Int, -cd.canvas_mins[2] / dx32[2]))
let bad = 0, tot = 0, ex = []
    for n in 50:400, ratio in (2, 3, 4, 5, 6), Rf in (2.0f0, 2.5f0, 3.0f0, 3.5f0)
        dxa = 1.0f0 / n; dxb = dxa / ratio
        cd = HC.ComputationalDomain(get_rectangular_domain(Float32, (0.0, 0.0), (1.0, 1.0)), (dxa, dxb), Rf)
        layers = round(Int, -cd.canvas_mins[2] / dxb)
        expect = ceil(Int, Float64(Rf) * ratio - 1e-9)
        tot += 1
        layers != expect && (bad += 1; length(ex) < 5 && push!(ex, (n, ratio, Rf, layers, expect)))
    end
    println("Float32 sweep: $bad / $tot with an extra layer; examples (n, ratio, R/dx, layers, expected): ", ex)
end
# spacings given in Float64 and converted per axis (what a config with Ns per axis does)
let bad = 0, tot = 0, ex = []
    for n in 50:400, ratio in (2, 3, 4, 5, 6), Rf in (2.0, 2.5, 3.0, 3.5)
        dxa = 1 / n; dxb = 1 / (n * ratio)
        cd = HC.ComputationalDomain(get_rectangular_domain(Float32, (0.0, 0.0), (1.0, 1.0)), (dxa, dxb), Rf)
        layers = round(Int, -cd.canvas_mins[2] / Float32(dxb))
        expect = ceil(Int, Rf * ratio - 1e-9)
        tot += 1
        layers != expect && (bad += 1; length(ex) < 5 && push!(ex, (n, ratio, Rf, layers, expect)))
    end
    println("Float32 sweep, Float64 spacings: $bad / $tot with an extra layer; examples (n, ratio, R/dx, layers, expected): ", ex)
end
