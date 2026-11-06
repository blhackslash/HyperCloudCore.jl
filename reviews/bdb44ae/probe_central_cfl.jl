# Uncached CFL path for non-MUSCL schemes: degree of the CFL fit vs the degree the Central safeguard uses.
include(joinpath(@__DIR__, "header.jl"))
eq = LinearAdvection(((1.0,),), Float64)
for R in (2.5, 3.5)
    pg = grid(1, Float64; n = 100, R)
    d3 = get_time_step(pg, eq, CentralDivergence(Float64, 1, 1, 3)); d4 = get_time_step(pg, eq, CentralDivergence(Float64, 1, 1, 4))
    @printf("1-D R=%.1f (%d neighbours): dt·n Central3 %.4f, Central4 %.4f (Central4 runs at degree %s here)\n", R, length(pg.neighbor.ranges[1]), d3 * 100, d4 * 100, R == 2.5 ? "3" : "4")
end
