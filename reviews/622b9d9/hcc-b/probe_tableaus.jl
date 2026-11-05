# Probe: order conditions of the IMEX tableaus (partitioned RK conditions up to order 3), node consistency,
# and the observed order of the IMEX schemes on a coupled scalar ODE, applied as GeneralRKTimeStepper does
# (Y_i = U + dt Σ_{j<i}(ã_ij KE_j + a_ij KI_j) + dt a_ii S(Y_i); U⁺ = U + dt Σ (b̃_j KE_j + b_j KI_j)).
using Test, HyperCloudCore, LinearAlgebra, Printf

function imex_conditions(bt)
    A, At = Matrix(bt.a), Matrix(bt.a_t)
    b, bt_ = Vector(bt.b), Vector(bt.b_t)
    c, ct = vec(sum(A; dims = 2)), vec(sum(At; dims = 2))
    Bs = (bt_, b); Cs = (ct, c); As = (At, A)
    o1 = [sum(bt_) - 1, sum(b) - 1]
    o2 = [x' * y - 1/2 for x in Bs for y in Cs]
    o3 = vcat([x' * (y .* z) - 1/3 for x in Bs for y in Cs for z in Cs],
              [x' * (Y * z) - 1/6 for x in Bs for Y in As for z in Cs])
    # separate (non-coupled) order 3 conditions of each part
    sep3 = [bt_' * (ct .^ 2) - 1/3, bt_' * (At * ct) - 1/6, b' * (c .^ 2) - 1/3, b' * (A * c) - 1/6]
    node_err = (maximum(abs, c - Vector(bt.c)), maximum(abs, ct - Vector(bt.c_t)))
    return (; o1 = maximum(abs, o1), o2 = maximum(abs, o2), o3 = maximum(abs, o3), sep3 = maximum(abs, sep3),
            sep2 = maximum(abs, [bt_' * ct - 1/2, b' * c - 1/2]), node_err)
end

# u' = F(u) + S(u), F(u) = -u^2 + sin(t) (explicit), S(u) = -λ (u - cos t) (implicit, solved exactly)
function imex_run(bt, dt; λ = 2.0, tend = 1.0)
    A, At = Matrix(bt.a), Matrix(bt.a_t); b, bt_ = Vector(bt.b), Vector(bt.b_t); ct = Vector(bt.c_t)
    s = length(b)
    F(t, u) = -u^2 + sin(t); S(t, u) = -λ * (u - cos(t))
    u = 1.0; t = 0.0
    nsteps = round(Int, tend / dt)
    for _ in 1:nsteps
        KE = zeros(s); KI = zeros(s)
        for i in 1:s
            ti = t + ct[i] * dt                       # the stepper's stage time for both parts
            Y = u + dt * sum(At[i, j] * KE[j] + A[i, j] * KI[j] for j in 1:i-1; init = 0.0)
            γ = dt * A[i, i]
            if abs(γ) > 0
                Y = (Y + γ * λ * cos(ti)) / (1 + γ * λ)
            end
            KI[i] = S(ti, Y); KE[i] = F(ti, Y)
        end
        u += dt * sum(bt_[j] * KE[j] + b[j] * KI[j] for j in 1:s); t += dt
    end
    return u
end

const IMEX = (IMEX_Euler_Tableau, IMEX_ARS233_Tableau, IMEX_ARS222_Tableau, IMEX_PRSSP3_Tableau, IMEX_SSP2332_Tableau)
const CLAIMED = Dict(IMEX_Euler_Tableau => 1, IMEX_ARS233_Tableau => 3, IMEX_ARS222_Tableau => 2,
                     IMEX_PRSSP3_Tableau => 3, IMEX_SSP2332_Tableau => 2)

@testset "probe_tableaus" begin
@testset "IMEX tableaus: order conditions and observed order" begin
    for make in IMEX
        bt = make(Float64)
        r = imex_conditions(bt)
        ref = imex_run(bt, 1 / 2560)
        e1 = abs(imex_run(bt, 1 / 40) - ref); e2 = abs(imex_run(bt, 1 / 80) - ref); e3 = abs(imex_run(bt, 1 / 160) - ref)
        eoc = (log2(e1 / e2), log2(e2 / e3))
        @printf("%-22s claimed %d | sep2 %.1e sep3 %.1e | coupled o2 %.1e o3 %.1e | nodes c %.1e c_t %.1e | EOC %.2f %.2f\n",
                nameof(make), CLAIMED[make], r.sep2, r.sep3, r.o2, r.o3, r.node_err..., eoc...)
        @test r.node_err[1] < 1e-12 && r.node_err[2] < 1e-12
        @test eoc[2] > CLAIMED[make] - 0.25
    end
end

@testset "Explicit tableaus: SSP coefficients" begin
    for make in (RK1_Euler_Tableau, RK2_SSP_Tableau, RK2_Ralston_Tableau, RK3_SSP_Tableau, RK4_Classical_Tableau, RK4_SSP_Tableau)
        @printf("%-22s C = %.12f\n", nameof(make), ssp_coefficient(make(Float64)))
    end
    # Float32 tableaus: order conditions of SSPRK(5,4) after rounding
    bt = RK4_SSP_Tableau(Float32)
    A, b, c = Float64.(Matrix(bt.a_t)), Float64.(Vector(bt.b_t)), Float64.(Vector(bt.c_t))
    @printf("RK4_SSP Float32: max |c - rowsum| = %.2e, b'c-1/2 = %.2e\n", maximum(abs, c - vec(sum(A; dims = 2))), b' * c - 0.5)
end

@testset "Proposed replacements reach their order" begin
    # ARS(2,2,2) in its published three-stage form (first stage explicit, Uⁿ); PR SSP3(3,3,2) (γ = 1 - 1/√2)
    g = 1 - 1 / sqrt(2); δ = 1 - 1 / (2g)
    ars222 = IMEXButcherTableau([0 0 0; 0 g 0; 0 1-g g], [0 0 0; g 0 0; δ 1-δ 0], [0, g, 1.0], [0, g, 1.0], [0, 1-g, g], [δ, 1-δ, 0])
    pr332  = IMEXButcherTableau([g 0 0; 1-2g g 0; 0.5-g 0 g], [0 0 0; 1.0 0 0; 0.25 0.25 0], [g, 1-g, 0.5], [0, 1.0, 0.5],
                                [1/6, 1/6, 2/3], [1/6, 1/6, 2/3])
    for (name, bt, p) in (("ARS222 (3-stage)", ars222, 2), ("PR SSP3(3,3,2)", pr332, 2))
        r = imex_conditions(bt)
        ref = imex_run(bt, 1 / 2560)
        e2 = abs(imex_run(bt, 1 / 80) - ref); e3 = abs(imex_run(bt, 1 / 160) - ref)
        @printf("%-22s coupled o2 %.1e | nodes %.1e %.1e | EOC %.2f | ssp_coefficient %.3f\n", name, r.o2, r.node_err..., log2(e2 / e3), ssp_coefficient(bt))
        @test r.o2 < 1e-14 && log2(e2 / e3) > p - 0.25
    end
end
end
