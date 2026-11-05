using Test, HyperCloudCore, LinearAlgebra
const HC = HyperCloudCore
explicit = (RK1_Euler_Tableau, RK2_SSP_Tableau, RK2_Ralston_Tableau, RK3_SSP_Tableau, RK4_Classical_Tableau, RK4_SSP_Tableau)
imex     = (IMEX_Euler_Tableau, IMEX_ARS233_Tableau, IMEX_ARS222_Tableau, IMEX_PRSSP3_Tableau, IMEX_SSP2332_Tableau)
println("== SSP coefficients Float64 vs Float32")
for mk in explicit
    c64 = ssp_coefficient(mk(Float64)); c32 = ssp_coefficient(mk(Float32))
    println(rpad(string(mk), 24), " C64 = ", c64, "  C32 = ", c32, "  rel diff = ", abs(c64 - c32) / max(c64, 1e-300))
end
println("== R(inf) of implicit parts, stiff accuracy")
for mk in imex
    bt = mk(Float64); A = Matrix(bt.a); b = Vector(bt.b); s = length(b)
    # R(z) = 1 + z b^T (I - z A)^{-1} e ; z -> -1e8
    R(z) = 1 + z * dot(b, (I - z * A) \ ones(s))
    println(rpad(string(mk), 24), " R(-1e8) = ", R(-1e8), "  stiffly accurate: ", b ≈ A[end, :], "  A diag: ", diag(A))
end
println("== explicit stability R_E(z) at imaginary axis (|R(iy)|) for small y (dissipation)")
for mk in imex
    bt = mk(Float64); At = Matrix(bt.a_t); bt_ = Vector(bt.b_t); s = length(bt_)
    R(z) = 1 + z * dot(bt_, (I - z * At) \ ones(s))
    println(rpad(string(mk), 24), " |R(0.5i)| = ", abs(R(0.5im)), " |R(1i)| = ", abs(R(1im)))
end
println("== order through scalar ODE: u' = -u*cos(t) [explicit] - lam*(u - sin(t)) [implicit], IMEX by hand with stage times from c_t")
function imex_ode(bt, n; lam = 2.0, tend = 1.0, implicit_time = :ct)
    A, At = Matrix(bt.a), Matrix(bt.a_t); b, bt_ = Vector(bt.b), Vector(bt.b_t); c, ct = Vector(bt.c), Vector(bt.c_t)
    s = length(b); dt = tend / n; u = 1.0; t = 0.0
    fE(t, u) = -u * cos(t); g(t) = sin(t)
    for _ in 1:n
        KE = zeros(s); KI = zeros(s)
        for i in 1:s
            Y = u + dt * sum(At[i, j] * KE[j] + A[i, j] * KI[j] for j in 1:i-1; init = 0.0)
            ti = t + (implicit_time == :ct ? ct[i] : c[i]) * dt
            γ = dt * A[i, i]
            Y = (Y + γ * lam * g(ti)) / (1 + γ * lam)          # solve Y = Yin - γ lam (Y - g)
            KI[i] = -lam * (Y - g(ti))
            KE[i] = fE(t + ct[i] * dt, Y)
        end
        u += dt * sum(bt_[j] * KE[j] + b[j] * KI[j] for j in 1:s); t += dt
    end
    return u
end
for mk in imex, it in (:ct, :c)
    bt = mk(Float64); ref = imex_ode(bt, 20480; implicit_time = it)
    e1 = abs(imex_ode(bt, 40; implicit_time = it) - ref); e2 = abs(imex_ode(bt, 80; implicit_time = it) - ref)
    println(rpad(string(mk), 24), " implicit times from ", it, ": EOC = ", round(log2(e1 / e2), digits = 2))
end
