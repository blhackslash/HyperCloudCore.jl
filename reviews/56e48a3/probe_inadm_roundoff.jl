# What the inadmissible first-order candidates are at CFL 0.4 (box on zero background, U >= 0 admissible)
include(joinpath(@__DIR__, "common.jl"))
eq = Adv{2, Float64}(SVector(1.0, 0.5))
for (lbl, jit, mood, cfl) in (("jitter 0.2 U1", 0.2, MOOD(EPD1(), MOODu1(1e-6)), 0.4), ("jitter 0.2 PAD", 0.2, MOOD(EPD1(), MOODPAD()), 0.4),
                             ("lattice PAD", 0.0, MOOD(EPD1(), MOODPAD()), 0.4), ("jitter 0.2 PAD cfl 0.1", 0.2, MOOD(EPD1(), MOODPAD()), 0.1))
    pg = grid2d(100; jitter = jit)
    setic!(pg, x -> (abs(x[1] - 0.5) < 0.2 && abs(x[2] - 0.5) < 0.2) ? 1.0 : 0.0)
    m  = MUSCL(Float64, 2, 1, 2, VenkatakrishnanLimiter(:hard), RusanovFlux())
    ts = GeneralRKTimeStepper(eq, m, mood, RK3_SSP_Tableau(Float64))
    dt = cfl * get_time_step(pg, eq, m)
    counts = Int[]
    t = 0.0
    for k in 1:5; ts(eq, pg, t, dt); t += dt; push!(counts, mood_inadmissible(ts)); end
    u = [pg.rhos[i][1] for i in 1:pg.meta.N]
    println("$lbl: inadmissible per step $counts, min state after 5 steps $(minimum(u)), #negative $(count(<(0), u))")
end
