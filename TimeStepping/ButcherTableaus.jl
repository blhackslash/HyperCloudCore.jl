# =========================================================================
# EXPLICIT RUNGE-KUTTA TABLEAUS
# =========================================================================

function RK1_Euler_Tableau()::RKButcherTableau
    # c = 1.0 ensures the grid moves by the full dt BEFORE divergence is evaluated
    return RKButcherTableau(zeros(Float64, 1, 1), [1.0], [1.0])
end

function RK2_Ralston_Tableau()::RKButcherTableau
    A = [0.0 0.0;
         2/3 0.0]
    return RKButcherTableau(A, [1/4, 3/4], [0.0, 2/3])
end

function RK3_SSP_Tableau()::RKButcherTableau
    # Shu-Osher SSPRK3
    A = [0.0  0.0  0.0;
         1.0  0.0  0.0;
         0.25 0.25 0.0]
    return RKButcherTableau(A, [1/6, 1/6, 2/3], [0.0, 1.0, 0.5])
end

function RK4_Classical_Tableau()::RKButcherTableau
    # Standard Classical RK4
    A = [0.0 0.0 0.0 0.0;
         0.5 0.0 0.0 0.0;
         0.0 0.5 0.0 0.0;
         0.0 0.0 1.0 0.0]
    return RKButcherTableau(A, [1/6, 1/3, 1/3, 1/6], [0.0, 0.5, 0.5, 1.0])
end

# =========================================================================
# IMEX RUNGE-KUTTA TABLEAUS
# =========================================================================
# Note: The strict IMEXButcherTableau constructor with bounds checking 
# is housed in TimestepperTypes.jl

function IMEX_Euler_Tableau()::IMEXButcherTableau
    # 2-Stage formulation of IMEX Euler
    At = [0.0 0.0;
          1.0 0.0]
    A  = [0.0 0.0;
          0.0 1.0]
          
    ct = [0.0, 1.0]
    c  = [0.0, 1.0]
    
    bt = [1.0, 0.0]
    b  = [0.0, 1.0]
    
    return IMEXButcherTableau(A, At, c, ct, b, bt)
end

function IMEX_ARS233_Tableau(gamma_val::Float64 = (3.0 + sqrt(3.0))/6.0)::IMEXButcherTableau
    # ARS(2,3,3) scheme from Ascher, Ruuth, Spiteri (1997), 3rd order.
    A_impl = [0.0 0.0             0.0;
              0.0 gamma_val       0.0;
              0.0 1.0-2*gamma_val gamma_val]

    At_expl = [0.0           0.0                 0.0;
               gamma_val     0.0                 0.0;
               gamma_val-1.0 2.0*(1.0-gamma_val) 0.0]

    c_nodes   = [0.0, gamma_val, 1.0-gamma_val] 
    b_weights = [0.0, 0.5, 0.5]

    return IMEXButcherTableau(A_impl, At_expl, c_nodes, c_nodes, b_weights, b_weights)
end

function IMEX_ARS222_Tableau(gamma_val::Union{Float64, Nothing}=nothing)::IMEXButcherTableau
    # ARS(2,2,2) IMEX scheme from Ascher, Ruuth, Spiteri (1997), 2nd order, L-stable.
    g_coeff = isnothing(gamma_val) ? (1.0 - 1.0 / sqrt(2.0)) : gamma_val
    delta   = 1.0 - 1.0 / (2.0 * g_coeff)

    At_expl = [0.0     0.0;
               g_coeff 0.0]
               
    A_impl  = [g_coeff       0.0;
               1.0 - g_coeff g_coeff]
               
    ct_expl = [0.0, g_coeff]
    c_impl  = [g_coeff, 1.0]
    
    bt = [delta, 1.0 - delta]
    b  = [1.0 - g_coeff, g_coeff]

    return IMEXButcherTableau(A_impl, At_expl, c_impl, ct_expl, b, bt)
end

function IMEX_PRSSP3_Tableau()::IMEXButcherTableau
    # Pareschi & Russo (2005), Scheme (4.2)
    gamma0 = 0.24169906235535784649 

    At = [0.0  0.0  0.0;
          1.0  0.0  0.0;
          0.25 0.25 0.0] 

    A_impl = zeros(Float64, 3, 3)
    A_impl[1,1] = gamma0
    A_impl[2,1] = 1.0 - 2.0 * gamma0
    A_impl[2,2] = gamma0
    A_impl[3,1] = ((1.0 - gamma0) / (1.0 - 2.0 * gamma0) - 1.0 / (12.0 * gamma0)) / 2.0
    A_impl[3,2] = (1.0 / (12.0 * gamma0 * (1.0 - 2.0 * gamma0))) / 2.0
    A_impl[3,3] = gamma0
    
    ct = [0.0, 1.0, 0.5]
    c_impl = [gamma0, 1.0 - gamma0, 0.5] 
    
    b_weights = [1.0/6.0, 1.0/6.0, 2.0/3.0]

    return IMEXButcherTableau(A_impl, At, c_impl, ct, b_weights, b_weights)
end

function IMEX_SSP2332_Tableau()::IMEXButcherTableau
    # SSP2(3, 3, 2) Stiffly Accurate Scheme
    A = [0.25 0.0  0.0;
         0.0  0.25 0.0; 
         1/3  1/3  1/3]
         
    At = [0.0 0.0 0.0; 
          0.5 0.0 0.0;
          0.5 0.5 0.0]
          
    c  = [0.25, 0.25, 1.0]
    ct = [0.0,  0.5,  1.0]
    b  = [1/3, 1/3, 1/3]
    
    return IMEXButcherTableau(A, At, c, ct, b, b)
end