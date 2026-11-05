include(joinpath(@__DIR__, "header.jl"))
eq = LinearAdvection(((1.0,),), Float64)
bt = RK4_Classical_Tableau(Float64)
cases = [
  ("MUSCL1",  K -> MUSCL(Float64, 1, 1, 1, NoLimiter(), RusanovFlux()), 2.5),
  ("MUSCL2",  K -> MUSCL(Float64, 1, 1, 2, NoLimiter(), RusanovFlux()), 2.5),
  ("MUSCL3",  K -> MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux()), 3.5),
  ("MUSCL3 R2.5",  K -> MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux()), 2.5),
  ("MUSCL4",  K -> MUSCL(Float64, 1, 1, 4, NoLimiter(), RusanovFlux()), 3.5),
  ("MUSCL5",  K -> MUSCL(Float64, 1, 1, 5, NoLimiter(), RusanovFlux()), 5.5),
  ("Central2", K -> CentralDivergence(Float64, 1, 1, 2), 2.5),
  ("Central3", K -> CentralDivergence(Float64, 1, 1, 3), 3.5),
  ("Central4 R2.5", K -> CentralDivergence(Float64, 1, 1, 4), 2.5),
  ("Central4 R3.5", K -> CentralDivergence(Float64, 1, 1, 4), 3.5),
  ("WENO2",   K -> WENO(Float64, 1, 1, 2), 3.5),
  ("WENO3",   K -> WENO(Float64, 1, 1, 3), 4.5),
  ("UpwindClassic", K -> UpwindDivergence(Float64, 1, 1, 1, :Classic, RusanovFlux()), 2.5),
  ("UpwindTiwari", K -> UpwindDivergence(Float64, 1, 1, 1, :Tiwari, RusanovFlux()), 2.5),
]
for jit in (0.0, 0.2), (name, mk, R) in cases
    errs = Float64[]
    for N in (40, 80, 160)
        pg = grid1d(N; range = R, jit = jit)
        push!(errs, advect_err(pg, eq, mk(N), bt; cfl = 0.1))
    end
    eoc = log2.(errs[1:end-1] ./ errs[2:end])
    println(rpad(name, 16), " jit=", jit, " errs=", round.(errs, sigdigits = 3), " EOC=", round.(eoc, digits = 2))
end
