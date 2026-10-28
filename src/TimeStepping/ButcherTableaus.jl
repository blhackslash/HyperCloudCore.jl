export IMEXButcherTableau, RKButcherTableau, num_stages, ssp_coefficient
export RK1_Euler_Tableau, RK2_SSP_Tableau, RK2_Ralston_Tableau, RK3_SSP_Tableau, RK4_Classical_Tableau, RK4_SSP_Tableau
export IMEX_Euler_Tableau, IMEX_ARS233_Tableau, IMEX_ARS222_Tableau, IMEX_PRSSP3_Tableau, IMEX_SSP2332_Tableau

"""
    IMEXButcherTableau{T}
    IMEXButcherTableau(a, a_t, c, c_t, b, b_t)

The single tableau type used by `GeneralRKTimeStepper`. An explicit Runge-Kutta scheme is the
special case `a == 0`, `b == 0` (see `RKButcherTableau`).

# Fields
- `a`, `c`, `b`     : implicit weights (lower triangular), nodes and final weights.
- `a_t`, `c_t`, `b_t`: explicit weights (strictly lower triangular), nodes and final weights.

# MOOD note
For an explicit tableau with a positive SSP coefficient ([`ssp_coefficient`](@ref): `RK1_Euler`, `RK2_SSP`,
`RK2_Ralston` (`C = 1/2`), `RK3_SSP`, `RK4_SSP`) the time stepper checks the forward Euler substeps of the
canonical Shu–Osher form, of which every stage is a convex combination; that is rigorous for any SSP method.
For the others (`RK4_Classical`, the IMEX tableaus) it checks the stage candidates, which has no theoretical
backing.
"""
struct IMEXButcherTableau{T, MA <: AbstractMatrix{T}, VA <: AbstractVector{T}}
    a::MA
    a_t::MA
    c::VA
    c_t::VA
    b::VA
    b_t::VA

    # Host tableaus use Matrix/Vector; inside kernels they arrive as SMatrix/SVector (see DeviceAdapt.jl).
    function IMEXButcherTableau(a::MA, a_t::MA, c::VA, c_t::VA, b::VA, b_t::VA) where {T, MA <: AbstractMatrix{T}, VA <: AbstractVector{T}}
        s = size(a, 1)
        @assert (size(a, 2) == s && size(a_t) == (s, s) &&
                 length(c) == s && length(c_t) == s && length(b) == s && length(b_t) == s) "All Butcher tableau components must match number of stages"
        for i in 1:s, j in (i+1):s
            @assert a[i, j] == zero(T) "Implicit matrix a must be lower triangular."
        end
        for i in 1:s, j in i:s
            @assert a_t[i, j] == zero(T) "Explicit matrix a_t must be strictly lower triangular."
        end
        new{T, MA, VA}(a, a_t, c, c_t, b, b_t)
    end
end

"""
    num_stages(bt::IMEXButcherTableau) -> Int

Number of stages `s` of the tableau.
"""
@inline num_stages(bt::IMEXButcherTableau) = length(bt.b)

"""
    static_tableau(bt::IMEXButcherTableau, ::Val{S}) -> IMEXButcherTableau

`bt` with `SMatrix` / `SVector` components, `S = num_stages(bt)`: a bits object with a concrete type
that kernel bodies can take as an argument. `GeneralRKTimeStepper` stores its tableau in this form.
"""
static_tableau(bt::IMEXButcherTableau{T}, ::Val{S}) where {T, S} = IMEXButcherTableau(
    SMatrix{S, S, T}(bt.a), SMatrix{S, S, T}(bt.a_t), SVector{S, T}(bt.c), SVector{S, T}(bt.c_t),
    SVector{S, T}(bt.b), SVector{S, T}(bt.b_t))

"""
    RKButcherTableau(a, b, c) -> IMEXButcherTableau

Wraps an explicit tableau `(a, b, c)` into an `IMEXButcherTableau` with a zero implicit part.
"""
function RKButcherTableau(a::Matrix{T}, b::Vector{T}, c::Vector{T}) where {T}
    s = length(b)
    return IMEXButcherTableau(zeros(T, s, s), a, c, c, zeros(T, s), b)
end

"""
    ssp_coefficient(bt::IMEXButcherTableau) -> Float64

The SSP coefficient `C` of an explicit tableau: the largest `r` for which the canonical Shu–Osher form
(Ketcheson 2008) `α_r = r K (I + r K)⁻¹`, `v_r = (I − α_r) e` (`K` the Butcher matrix with the weights `b` as
last row) has no negative coefficient. Then every stage and the final update are
`Y_i = v_i Uⁿ + Σ_j α_ij (Y_j + (Δt/C) F(Y_j))`, convex combinations of `Uⁿ` and the forward Euler substeps
`Y_j + (Δt/C) F(Y_j)`. `0` for tableaus that are not SSP (classical RK4) and for IMEX tableaus (non-zero
implicit part). Computed by bisection on the host (accurate to about 1e-10).
"""
function ssp_coefficient(bt::IMEXButcherTableau)
    any(!iszero, bt.a) && return 0.0
    s = num_stages(bt)
    K = zeros(s + 1, s + 1)
    K[1:s, 1:s] .= Float64.(Matrix(bt.a_t)); K[s + 1, 1:s] .= Float64.(Vector(bt.b_t))
    ok(r) = (α = r * K / (I + r * K); minimum(α) >= -1e-13 && minimum((I - α) * ones(s + 1)) >= -1e-13)
    r_lo, r_hi = 1e-3, 1e-3
    ok(r_lo) || return 0.0                      # below 1e-3 the method is not usefully SSP (classical RK4: C = 0)
    while ok(r_hi) && r_hi < 1e3
        r_hi *= 2
    end
    r_lo = r_hi / 2
    for _ in 1:60
        mid = (r_lo + r_hi) / 2
        ok(mid) ? (r_lo = mid) : (r_hi = mid)
    end
    return r_lo
end

# =========================================================================
# EXPLICIT TABLEAUS
# =========================================================================

"Forward Euler (SSP)."
RK1_Euler_Tableau(::Type{T}) where {T} = RKButcherTableau(zeros(T, 1, 1), T[1], T[0])

"2-stage, 2nd-order SSP (Heun). Preferred over Ralston when MOOD is active."
function RK2_SSP_Tableau(::Type{T}) where {T}
    A = T[0.0 0.0;
          1.0 0.0]
    return RKButcherTableau(A, T[1/2, 1/2], T[0.0, 1.0])
end

"2-stage, 2nd-order Ralston (not SSP)."
function RK2_Ralston_Tableau(::Type{T}) where {T}
    A = T[0.0 0.0;
          2/3 0.0]
    return RKButcherTableau(A, T[1/4, 3/4], T[0.0, 2/3])
end

"3-stage, 3rd-order SSP (Shu-Osher)."
function RK3_SSP_Tableau(::Type{T}) where {T}
    A = T[0.0  0.0  0.0;
          1.0  0.0  0.0;
          0.25 0.25 0.0]
    return RKButcherTableau(A, T[1/6, 1/6, 2/3], T[0.0, 1.0, 0.5])
end

"Classical RK4 (not SSP)."
function RK4_Classical_Tableau(::Type{T}) where {T}
    A = T[0.0 0.0 0.0 0.0;
          0.5 0.0 0.0 0.0;
          0.0 0.5 0.0 0.0;
          0.0 0.0 1.0 0.0]
    return RKButcherTableau(A, T[1/6, 1/3, 1/3, 1/6], T[0.0, 0.5, 0.5, 1.0])
end

"""
    RK4_SSP_Tableau(T)

SSPRK(5,4) of Spiteri and Ruuth (SIAM J. Numer. Anal. 40, 2002): five stages, order 4, the optimal
four-stage-free SSP coefficient `C ≈ 1.508` (SSP-RK3 has `C = 1`, so it allows a 1.5 times larger step per
stage, i.e. about the same cost per unit time as RK4 with the SSP guarantee). The Butcher form of the
published Shu–Osher coefficients; the order conditions up to 4 hold to round-off.
"""
function RK4_SSP_Tableau(::Type{T}) where {T}
    A = T[0.0                  0.0                  0.0                  0.0                  0.0;
          0.39175222657189002  0.0                  0.0                  0.0                  0.0;
          0.21766909626116876  0.36841059305037099  0.0                  0.0                  0.0;
          0.082692086657810582 0.13995850219189535  0.25189177427169401  0.0                  0.0;
          0.067966283637114752 0.11503469850463156  0.20703489859738566  0.54497475022852104  0.0]
    b = T[0.14681187608478657, 0.24848290944497617, 0.10425883033198098, 0.2744389009013507, 0.226007483236906]
    c = T[0.0, 0.39175222657189002, 0.58607968931153975, 0.47454236312139997, 0.93501063096765302]
    return RKButcherTableau(A, b, c)
end

# =========================================================================
# IMEX TABLEAUS
# =========================================================================

"1st-order IMEX Euler (forward/backward)."
function IMEX_Euler_Tableau(::Type{T}) where {T}
    At = T[0.0 0.0; 1.0 0.0]
    A  = T[0.0 0.0; 0.0 1.0]
    c  = T[0.0, 1.0]
    return IMEXButcherTableau(A, At, c, c, T[0.0, 1.0], T[1.0, 0.0])
end

"Ascher-Ruuth-Spiteri ARS(2,3,3): 3 stages, 3rd order (not SSP)."
function IMEX_ARS233_Tableau(::Type{T}, γ::T = T((3.0 + sqrt(3.0)) / 6.0)) where {T}
    A  = T[0.0 0.0     0.0;
           0.0 γ       0.0;
           0.0 1.0-2γ  γ]
    At = T[0.0   0.0      0.0;
           γ     0.0      0.0;
           γ-1.0 2.0(1-γ) 0.0]
    c  = T[0.0, γ, 1.0-γ]
    b  = T[0.0, 0.5, 0.5]
    return IMEXButcherTableau(A, At, c, c, b, b)
end

"Ascher-Ruuth-Spiteri ARS(2,2,2): 2 stages, 2nd order (not SSP)."
function IMEX_ARS222_Tableau(::Type{T}, γ::Union{T, Nothing} = nothing) where {T}
    g = isnothing(γ) ? T(1.0 - 1.0 / sqrt(2.0)) : γ
    δ = T(1.0 - 1.0 / (2.0 * g))
    At = T[0.0 0.0; g 0.0]
    A  = T[g 0.0; 1.0-g g]
    return IMEXButcherTableau(A, At, T[g, 1.0], T[0.0, g], T[1.0-g, g], T[δ, 1.0-δ])
end

"Pareschi-Russo SSP3(3,3,3): 3rd-order, SSP explicit part."
function IMEX_PRSSP3_Tableau(::Type{T}) where {T}
    γ = T(0.24169906235535784649)
    At = T[0.0  0.0  0.0;
           1.0  0.0  0.0;
           0.25 0.25 0.0]
    A = zeros(T, 3, 3)
    A[1, 1] = γ
    A[2, 1] = T(1.0 - 2.0γ);  A[2, 2] = γ
    A[3, 1] = T(((1.0 - γ) / (1.0 - 2.0γ) - 1.0 / (12.0γ)) / 2.0)
    A[3, 2] = T((1.0 / (12.0γ * (1.0 - 2.0γ))) / 2.0)
    A[3, 3] = γ
    b = T[1/6, 1/6, 2/3]
    return IMEXButcherTableau(A, At, T[γ, 1.0-γ, 0.5], T[0.0, 1.0, 0.5], b, b)
end

"SSP2(3,3,2): 2nd-order, SSP explicit part, L-stable implicit part."
function IMEX_SSP2332_Tableau(::Type{T}) where {T}
    A  = T[0.25 0.0 0.0; 0.0 0.25 0.0; 1/3 1/3 1/3]
    At = T[0.0 0.0 0.0; 0.5 0.0 0.0; 0.5 0.5 0.0]
    b  = T[1/3, 1/3, 1/3]
    return IMEXButcherTableau(A, At, T[0.25, 0.25, 1.0], T[0.0, 0.5, 1.0], b, b)
end
