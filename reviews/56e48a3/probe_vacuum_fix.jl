# Suggested fix for the Float32 vacuum hole: tolerance relative to |f_i| and the bound of the side the increment points to
include(joinpath(@__DIR__, "common.jl"))
@eval HCC @inline function _update_phi(strategy, phi::State{M, T}, delta_recon::State{M, T}, f_i::State{M, T},
                             u_max::State{M, T}, u_min::State{M, T}) where {M, T}
    return State{M, T}(ntuple(Val(M)) do m
        recon_m = delta_recon[m]
        bound   = recon_m > zero(T) ? u_max[m] : u_min[m]
        tol_m   = 16 * eps(T) * max(abs(f_i[m]), abs(bound))
        if abs(recon_m) > tol_m
            r = (bound - f_i[m]) / recon_m
            min(phi[m], limiter_phi(strategy, r))
        else
            phi[m]
        end
    end)
end
include(joinpath(@__DIR__, "probe_vacuum_f32.jl"))
# scale invariance of the fixed version (VK hard box, as the new test)
eq = Adv{1, Float64}(SVector(1.0))
function run_box(a; lim = VenkatakrishnanLimiter(:hard))
    pg = setic!(grid1d(60), x -> abs(x[1]) < 0.3 ? a : 0.0)
    m  = MUSCL(Float64, 1, 1, 3, lim, RusanovFlux())
    ts = GeneralRKTimeStepper(eq, m, MOOD(), RK2_SSP_Tableau(Float64))
    dt = 0.4 * 2 / 60
    for k in 0:9; ts(eq, pg, k * dt, dt); end
    [pg.rhos[i][1] for i in 1:pg.meta.N]
end
u1 = run_box(1.0); us = run_box(2.0^-50)
println("fixed tolerance: bitwise scaled = ", us == 2.0^-50 .* u1, ", extrema ", extrema(u1))
