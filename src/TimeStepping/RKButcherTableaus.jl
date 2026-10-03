export RKButcherTableau
export RK1_Euler_Tableau, RK2_Ralston_Tableau, RK3_SSP_Tableau, RK4_Classical_Tableau
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