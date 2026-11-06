include(joinpath(@__DIR__, "header.jl"))
println("== 1-D MUSCL4 jit 0.2, R = 3.5, larger N")
eq1 = LinearAdvection(((1.0,),), Float64)
errs = Float64[]
for N in (80, 160, 320, 640)
    push!(errs, advect_err(grid1d(N; range = 3.5, jit = 0.2), eq1, MUSCL(Float64, 1, 1, 4, NoLimiter(), RusanovFlux()), RK4_Classical_Tableau(Float64); cfl = 0.1))
end
println("MUSCL4 1D jit .2 errs=", round.(errs, sigdigits = 3), " EOC=", round.(log2.(errs[1:end-1] ./ errs[2:end]), digits = 2))
flush(stdout)
println("== 2-D, jit 0.2, advection (1, 0.5), sin(2πx)sin(2πy)")
eq = LinearAdvection(((1.0,), (0.5,)), Float64)
f0 = x -> sin(2pi * x[1]) * sin(2pi * x[2])
function exact_err2(pg, eq, scheme; tend = 0.25, cfl = 0.1)
    setic!(pg, f0)
    ts = GeneralRKTimeStepper(eq, scheme, MOOD(), RK4_Classical_Tableau(Float64))
    dx = minimum(pg.meta.dx); n = ceil(Int, tend / (cfl * dx)); dt = tend / n; t = 0.0
    for k in 1:n; ts(eq, pg, t, dt); t += dt; end
    maximum(i -> abs(pg.rhos[i][1] - f0(pg.core.positions[i] .- (1.0, 0.5) .* tend)), interior_indices(pg))
end
for (name, K, R, kw) in (("MUSCL2", 2, 2.5, ()), ("MUSCL3", 3, 2.5, ()), ("MUSCL3 R3.5", 3, 3.5, ()), ("MUSCL4", 4, 2.5, ()), ("MUSCL4 R3.5", 4, 3.5, ()), ("MUSCL5", 5, 3.5, ()))
    errs = Float64[]
    for n in (16, 32, 64)
        push!(errs, exact_err2(grid2d(n; range = R, jit = 0.2), eq, MUSCL(Float64, 2, 1, K, NoLimiter(), RusanovFlux(); kw...)))
    end
    println(rpad(name, 14), " errs=", round.(errs, sigdigits = 3), " EOC=", round.(log2.(errs[1:end-1] ./ errs[2:end]), digits = 2))
    flush(stdout)
end
