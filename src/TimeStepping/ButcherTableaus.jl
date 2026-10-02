
export RK1_Euler_Tableau, RK2_Ralston_Tableau, RK3_SSP_Tableau, RK4_Classical_Tableau
export IMEX_Euler_Tableau, IMEX_ARS233_Tableau, IMEX_ARS222_Tableau, IMEX_PRSSP3_Tableau, IMEX_SSP2332_Tableau


"""
    RKButcherTableau{T}

A structure storing the coefficients for explicit Runge-Kutta time integration.

# Fields
- `a::Matrix{T}`: The strictly lower-triangular matrix containing the explicit stage weights.
- `b::Vector{T}`: The vector containing the final combination weights.
- `c::Vector{T}`: The vector containing the fractional time steps for each stage.
"""
struct RKButcherTableau{T}
    a::Matrix{T}
    b::Vector{T}
    c::Vector{T}
end

"""
    IMEXButcherTableau{T}
    IMEXButcherTableau(a::Matrix{T}, a_t::Matrix{T}, c::Vector{T}, c_t::Vector{T}, b::Vector{T}, b_t::Vector{T})

A structure storing the paired coefficients for Implicit-Explicit (IMEX) Runge-Kutta time integration. The inner constructor enforces strict dimensional consistency across all components and verifies that the implicit matrix `a` is lower-triangular, while the explicit matrix `a_t` is strictly lower-triangular.

# Fields
- `a::Matrix{T}`: The implicit stage weights (lower triangular).
- `a_t::Matrix{T}`: The explicit stage weights (strictly lower triangular).
- `c::Vector{T}`: Fractional time steps for the implicit stages.
- `c_t::Vector{T}`: Fractional time steps for the explicit stages.
- `b::Vector{T}`: The final combination weights for the implicit evaluations.
- `b_t::Vector{T}`: The final combination weights for the explicit evaluations.
"""
struct IMEXButcherTableau{T} 
    a::Matrix{T}  
    a_t::Matrix{T} 
    c::Vector{T}   
    c_t::Vector{T} 
    b::Vector{T}   
    b_t::Vector{T}
    
    function IMEXButcherTableau(a::Matrix{T}, a_t::Matrix{T}, c::Vector{T}, c_t::Vector{T}, b::Vector{T}, b_t::Vector{T}) where {T}
        s = size(a, 1) 
        @assert (size(a, 2) == s && size(a_t, 1) == s && size(a_t, 2) == s &&
                 length(c) == s && length(c_t) == s && length(b) == s && length(b_t) == s) "All Butcher tableau components must match number of stages"    
        
        for i in 1:s, j in (i+1):s
            @assert a[i,j] == zero(T) "Implicit matrix A must be lower triangular."
        end
        for i in 1:s, j in i:s 
            @assert a_t[i,j] == zero(T) "Explicit matrix a_t must be strictly lower triangular."
        end
        new{T}(a, a_t, c, c_t, b, b_t)
    end
end

# =========================================================================
# EXPLICIT RUNGE-KUTTA TABLEAUS
# =========================================================================
"""
    RK1_Euler_Tableau(::Type{T}) -> RKButcherTableau{T}

Constructs the Butcher tableau for the standard 1st-order forward Euler scheme.
"""
function RK1_Euler_Tableau(::Type{T})::RKButcherTableau{T} where {T}
    return RKButcherTableau(zeros(T, 1, 1), T[1], T[1])
end

"""
    RK2_Ralston_Tableau(::Type{T}) -> RKButcherTableau{T}

Constructs the Butcher tableau for the 2nd-order explicit Ralston method.
"""
function RK2_Ralston_Tableau(::Type{T})::RKButcherTableau{T} where {T}
    A = T[0.0 0.0;
          2/3 0.0]
    return RKButcherTableau(A, T[1/4, 3/4], T[0.0, 2/3])
end

"""
    RK3_SSP_Tableau(::Type{T}) -> RKButcherTableau{T}

Constructs the Butcher tableau for a 3rd-order Strong Stability Preserving (SSP) Runge-Kutta scheme.
"""
function RK3_SSP_Tableau(::Type{T})::RKButcherTableau{T} where {T}
    A = T[0.0  0.0  0.0;
          1.0  0.0  0.0;
          0.25 0.25 0.0]
    return RKButcherTableau(A, T[1/6, 1/6, 2/3], T[0.0, 1.0, 0.5])
end

"""
    RK4_Classical_Tableau(::Type{T}) -> RKButcherTableau{T}

Constructs the Butcher tableau for the standard 4th-order classical Runge-Kutta scheme.
"""
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
    IMEX_Euler_Tableau(::Type{T}) -> IMEXButcherTableau{T}

Constructs a basic 1st-order IMEX Euler tableau pairing forward and backward Euler steps.
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

"""
    IMEX_ARS233_Tableau(::Type{T}, gamma_val::T = (3+√3)/6) -> IMEXButcherTableau{T}

Constructs the Ascher-Ruuth-Spiteri ARS233 IMEX tableau, a 3-stage, 2nd-order scheme with a parameterized `gamma_val` diagonal.
"""
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

"""
    IMEX_ARS222_Tableau(::Type{T}, gamma_val::Union{T, Nothing}=nothing) -> IMEXButcherTableau{T}

Constructs the Ascher-Ruuth-Spiteri ARS222 IMEX tableau, a 2-stage, 2nd-order scheme. If `gamma_val` is omitted, it defaults to `1 - 1/√2`.
"""
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

"""
    IMEX_PRSSP3_Tableau(::Type{T}) -> IMEXButcherTableau{T}

Constructs the 3rd-order Strong Stability Preserving (SSP) IMEX PRSSP3 tableau designed for robust stability with stiff sources.
"""
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

"""
    IMEX_SSP2332_Tableau(::Type{T}) -> IMEXButcherTableau{T}

Constructs the SSP2332 IMEX tableau, offering Strong Stability Preserving properties within a parameterized implicit-explicit setup.
"""
function IMEX_SSP2332_Tableau(::Type{T})::IMEXButcherTableau{T} where {T}
    A  = T[0.25 0.0 0.0; 0.0  0.25 0.0; 1/3 1/3 1/3]
    At = T[0.0 0.0 0.0; 0.5 0.0 0.0; 0.5 0.5 0.0]
    c  = T[0.25, 0.25, 1.0]
    ct = T[0.0,  0.5,  1.0]
    b  = T[1/3, 1/3, 1/3]
    
    return IMEXButcherTableau(A, At, c, ct, b, b)
end
