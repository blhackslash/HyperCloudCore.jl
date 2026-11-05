# Probe: the IMEX tableaus in the stiff regime, u' = -u² + sin t - λ (u - cos t), λ = 1e2 … 1e6 (hand-applied
# stage recipe of GeneralRKTimeStepper, see probe_tableaus.jl). Reference: same tableau at dt/32.
try include(joinpath(@__DIR__, "probe_tableaus.jl")) catch end   # defines imex_run (its own testsets fail by design)
g = 1 - 1 / sqrt(2); δ = 1 - 1 / (2g)
ars222_3 = IMEXButcherTableau([0 0 0; 0 g 0; 0 1-g g], [0 0 0; g 0 0; δ 1-δ 0], [0, g, 1.0], [0, g, 1.0], [0, 1-g, g], [δ, 1-δ, 0])
@testset "stiff" begin
for λ in (1e2, 1e4, 1e6), (name, bt) in (("ARS222 (code)", IMEX_ARS222_Tableau(Float64)), ("ARS222 (3-stage)", ars222_3),
                                       ("PRSSP3 (code)", IMEX_PRSSP3_Tableau(Float64)), ("SSP2332", IMEX_SSP2332_Tableau(Float64)))
    ref = imex_run(bt, 1 / 2560; λ)
    e1, e2 = abs(imex_run(bt, 1 / 40; λ) - ref), abs(imex_run(bt, 1 / 80; λ) - ref)
    @printf("λ=%.0e %-18s err %.2e -> %.2e EOC %.2f\n", λ, name, e1, e2, log2(e1 / e2))
end
end
