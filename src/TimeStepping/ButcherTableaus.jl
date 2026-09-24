
export RK1_Euler_Tableau, RK2_Ralston_Tableau, RK3_SSP_Tableau, RK4_Classical_Tableau
export IMEX_Euler_Tableau, IMEX_ARS233_Tableau, IMEX_ARS222_Tableau, IMEX_PRSSP3_Tableau, IMEX_SSP2332_Tableau
# =========================================================================
# EXPLICIT RUNGE-KUTTA TABLEAUS
# =========================================================================
"""
    RK1_Euler_Tableau(::Type{T})
    RK2_Ralston_Tableau(::Type{T})
    RK3_SSP_Tableau(::Type{T})
    RK4_Classical_Tableau(::Type{T})

Constructs standard explicit Runge-Kutta Butcher tableaus of various temporal orders. 
- `RK1_Euler_Tableau`: 1st-order forward Euler scheme.
- `RK2_Ralston_Tableau`: 2nd-order Ralston method.
- `RK3_SSP_Tableau`: 3rd-order Strong Stability Preserving (SSP) scheme.
- `RK4_Classical_Tableau`: 4th-order classical RK scheme.
"""
function RK1_Euler_Tableau(::Type{T})::RKButcherTableau{T} where {T}
    return RKButcherTableau(zeros(T, 1, 1), T[1], T[1])
end

function RK2_Ralston_Tableau(::Type{T})::RKButcherTableau{T} where {T}
    A = T[0.0 0.0;
          2/3 0.0]
    return RKButcherTableau(A, T[1/4, 3/4], T[0.0, 2/3])
end

function RK3_SSP_Tableau(::Type{T})::RKButcherTableau{T} where {T}
    A = T[0.0  0.0  0.0;
          1.0  0.0  0.0;
          0.25 0.25 0.0]
    return RKButcherTableau(A, T[1/6, 1/6, 2/3], T[0.0, 1.0, 0.5])
end

function RK4_Classical_Tableau(::Type{T})::RKButcherTableau{T} where {T}
    A = T[0.0 0.0 0.0 0.0;
          0.5 0.0 0.0 0.0;
          0.0 0.5 0.0 0.0;
          0.0 0.0 1.0 0.0]
    return RKButcherTableau(A, T[1/6, 1/3, 1/3, 1/6], T[0.0, 0.5, 0.5, 1.0])
end

# =========================================================================
# IMEX RUNGE-KUTTA TABLEAUS
# =========================================================================
"""
    IMEX_Euler_Tableau(::Type{T})
    IMEX_ARS233_Tableau(::Type{T}, gamma_val)
    IMEX_ARS222_Tableau(::Type{T}, gamma_val)
    IMEX_PRSSP3_Tableau(::Type{T})
    IMEX_SSP2332_Tableau(::Type{T})

Constructs Implicit-Explicit (IMEX) Runge-Kutta Butcher tableaus designed to handle stiff source terms implicitly while treating advection explicitly.
- Contains basic explicit-implicit definitions like `IMEX_Euler`.
- Includes Ascher-Ruuth-Spiteri (ARS) schemes such as `ARS233` and `ARS222`, which allow optional parameterization of `gamma_val`.
- Includes Strong Stability Preserving (SSP) IMEX configurations like `PRSSP3` and `SSP2332`. 
"""
function IMEX_Euler_Tableau(::Type{T})::IMEXButcherTableau{T} where {T}
    At = T[0.0 0.0; 1.0 0.0]
    A  = T[0.0 0.0; 0.0 1.0]
    ct = T[0.0, 1.0]
    c  = T[0.0, 1.0]
    bt = T[1.0, 0.0]
    b  = T[0.0, 1.0]
    
    return IMEXButcherTableau(A, At, c, ct, b, bt)
end

function IMEX_ARS233_Tableau(::Type{T}, gamma_val::T = T((3.0 + sqrt(3.0))/6.0))::IMEXButcherTableau{T} where {T}
    A_impl = T[0.0 0.0             0.0;
               0.0 gamma_val       0.0;
               0.0 1.0-2*gamma_val gamma_val]

    At_expl = T[0.0           0.0                 0.0;
                gamma_val     0.0                 0.0;
                gamma_val-1.0 2.0*(1.0-gamma_val) 0.0]

    c_nodes   = T[0.0, gamma_val, 1.0-gamma_val] 
    b_weights = T[0.0, 0.5, 0.5]

    return IMEXButcherTableau(A_impl, At_expl, c_nodes, c_nodes, b_weights, b_weights)
end

function IMEX_ARS222_Tableau(::Type{T}, gamma_val::Union{T, Nothing}=nothing)::IMEXButcherTableau{T} where {T}
    g_coeff = isnothing(gamma_val) ? T(1.0 - 1.0 / sqrt(2.0)) : gamma_val
    delta   = T(1.0 - 1.0 / (2.0 * g_coeff))

    At_expl = T[0.0     0.0;
                g_coeff 0.0]
               
    A_impl  = T[g_coeff       0.0;
                1.0 - g_coeff g_coeff]
               
    ct_expl = T[0.0, g_coeff]
    c_impl  = T[g_coeff, 1.0]
    
    bt = T[delta, 1.0 - delta]
    b  = T[1.0 - g_coeff, g_coeff]

    return IMEXButcherTableau(A_impl, At_expl, c_impl, ct_expl, b, bt)
end

function IMEX_PRSSP3_Tableau(::Type{T})::IMEXButcherTableau{T} where {T}
    gamma0 = T(0.24169906235535784649) 

    At = T[0.0  0.0  0.0;
           1.0  0.0  0.0;
           0.25 0.25 0.0] 

    A_impl = zeros(T, 3, 3)
    A_impl[1,1] = gamma0
    A_impl[2,1] = T(1.0 - 2.0 * gamma0)
    A_impl[2,2] = gamma0
    A_impl[3,1] = T(((1.0 - gamma0) / (1.0 - 2.0 * gamma0) - 1.0 / (12.0 * gamma0)) / 2.0)
    A_impl[3,2] = T((1.0 / (12.0 * gamma0 * (1.0 - 2.0 * gamma0))) / 2.0)
    A_impl[3,3] = gamma0
    
    ct = T[0.0, 1.0, 0.5]
    c_impl = T[gamma0, 1.0 - gamma0, 0.5] 
    
    b_weights = T[1/6, 1/6, 2/3]

    return IMEXButcherTableau(A_impl, At, c_impl, ct, b_weights, b_weights)
end

function IMEX_SSP2332_Tableau(::Type{T})::IMEXButcherTableau{T} where {T}
    A  = T[0.25 0.0 0.0; 0.0  0.25 0.0; 1/3 1/3 1/3]
    At = T[0.0 0.0 0.0; 0.5 0.0 0.0; 0.5 0.5 0.0]
    c  = T[0.25, 0.25, 1.0]
    ct = T[0.0,  0.5,  1.0]
    b  = T[1/3, 1/3, 1/3]
    
    return IMEXButcherTableau(A, At, c, ct, b, b)
end