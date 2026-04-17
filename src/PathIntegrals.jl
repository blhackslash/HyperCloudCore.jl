abstract type AbstractPath end
abstract type DifferentialOrder end
struct Order0 <: DifferentialOrder end
struct Order1 <: DifferentialOrder end

const DO0 = Order0() 
const DO1 = Order1()
# ---------------------------------------------------------
# 1. Standard Straight Line Path
# ---------------------------------------------------------
struct LinePath <: AbstractPath end

struct NaiveAveragePath <: AbstractPath end

# DO0 (The State): Returns a constant average regardless of 's'
@inline function (::NaiveAveragePath)(eq::HyperbolicPDE, s, uL::State{M}, uR::State{M}, ::Order0) where {M}
    return 0.5 * (uL + uR)
end

# DO1 (The Derivative): Fakes the derivative to be the standard jump
@inline function (::NaiveAveragePath)(eq::HyperbolicPDE, s, uL::State{M}, uR::State{M}, ::Order1) where {M}
    return uR - uL
end

# Pure SVector math, completely unrolled by the compiler!
@inline (::LinePath)(eq::HyperbolicPDE, s, uL::State{M}, uR::State{M}, ::Order0) where {M} = uL + s * (uR - uL)
@inline (::LinePath)(eq::HyperbolicPDE, s, uL::State{M}, uR::State{M}, ::Order1) where {M} = uR - uL


# ---------------------------------------------------------
# 2. The Mapped Path Decorator
# ---------------------------------------------------------
struct MappedPath{P <: AbstractPath} <: AbstractPath
    base_path::P
end

# Default constructor wraps a standard LinePath
MappedPath() = MappedPath(LinePath())

# 0th Derivative: Map to Cons -> Evaluate Base Path -> Map to Prim
@inline function (mp::MappedPath)(eq::HyperbolicPDE, s, vL::State{M}, vR::State{M}, ::Order0) where {M}
    wL = prim2cons(eq, vL)
    wR = prim2cons(eq, vR)
    
    w_s = mp.base_path(eq, s, wL, wR, DO0)
    
    return cons2prim(eq, w_s)
end

# 1st Derivative: The Chain Rule (∂V/∂W * dW/ds)
@inline function (mp::MappedPath)(eq::HyperbolicPDE, s, vL::State{M}, vR::State{M}, ::Order1) where {M}
    wL = prim2cons(eq, vL)
    wR = prim2cons(eq, vR)
    
    w_s  = mp.base_path(eq, s, wL, wR, DO0)
    dw_s = mp.base_path(eq, s, wL, wR, DO1)
    
    # We use SVector Finite Differences here as a highly-efficient, generalized chain rule.
    # (If you ever want to use an exact analytical Jacobian, you can just create a new 
    # `AnalyticalMappedPath <: AbstractPath` and plug the matrix in here!)
    eps_fd = 1e-6
    V_s      = cons2prim(eq, w_s)
    V_s_plus = cons2prim(eq, w_s + eps_fd * dw_s)
    
    return (V_s_plus - V_s) / eps_fd
end

# Abstract definition
function path_integral(eq::HyperbolicPDE, u_left::Tuple, u_right::Tuple)
    error("path_integral not implemented for $(typeof(eq))")
end

# --- Helper: Matrix-Vector Product for Non-Conservative Systems ---
# To avoid heap-allocated matrices, we define A(U)*v directly as a Tuple.
function A_matrix_times_vector(eq::HyperbolicPDE, U::Tuple, v::Tuple)
    error("A_matrix_times_vector not implemented for $(typeof(eq))")
end

# =========================================================================
# NON-CONSERVATIVE MATVECS (1D)
# =========================================================================

# 1. LAGRANGIAN (Material Derivative Frame)
# A(V) = [0, ρ, 0; 0, 0, 1/ρ; 0, γp, 0]
@inline function A_matrix_times_vector(::EulerEquation{1, 3, Lagrangian}, V::State{3}, dV::State{3})
    rho, u, p = V[1], V[2], V[3]
    drho, du, dp = dV[1], dV[2], dV[3]
    
    return State{3}(rho * du, dp / rho, GAS_GAMMA_EULER * p * du)
end

# 2. PRIMITIVE (Eulerian Frame)
# A(V) = [u, ρ, 0; 0, u, 1/ρ; 0, γp, u]
@inline function A_matrix_times_vector(::EulerEquation{1, 3, Primitive}, V::State{3}, dV::State{3})
    rho, u, p = V[1], V[2], V[3]
    drho, du, dp = dV[1], dV[2], dV[3]
    
    return State{3}(
        u * drho + rho * du,
        u * du + dp / rho,
        GAS_GAMMA_EULER * p * du + u * dp
    )
end

@inline function gauss_lobatto_5()
    # Standard 5-point Lobatto nodes on [-1, 1] are: -1, -sqrt(3/7), 0, sqrt(3/7), 1
    # Transformed to [0, 1] using s = (x + 1) / 2
    s2_offset = 0.5 * sqrt(3/7)
    nodes = (
        0.0, 
        0.5 - s2_offset, 
        0.5, 
        0.5 + s2_offset, 
        1.0
    )
    
    # Standard 5-point Lobatto weights on [-1, 1] are: 1/10, 49/90, 32/45, 49/90, 1/10
    # Transformed to [0, 1] using W = w / 2
    weights = (
        1/20,      # 0.05
        49/180,    # ~0.2722
        16/45,     # ~0.3555
        49/180, 
        1/20
    )
    return nodes, weights
end

@inline function simpson_3_point()
    # Nodes on [0, 1]
    nodes = (0.0, 0.5, 1.0)
    # Weights (must sum to 1.0)
    weights = (1/6, 4/6, 1/6)
    return nodes, weights
end

@inline function path_integral(eq::HyperbolicPDE{D, M}, path::AbstractPath, uL::State{M}, uR::State{M}) where {D, M}
    # Gauss-Lobatto 5 is strongly recommended for non-conservative integrals
    nodes, weights = gauss_lobatto_5() 
    
    integral = zero(State{M})

    for i in eachindex(nodes)
        s = nodes[i]
        w = weights[i]
        
        # The Path object handles all coordinate transformations internally!
        U_s  = path(eq, s, uL, uR, DO0)
        dU_s = path(eq, s, uL, uR, DO1)
        
        term = A_matrix_times_vector(eq, U_s, dU_s)
        integral += w * term
    end
    
    if maximum(abs.(integral)) > 1000; error("Integral too large!"); end
    return integral
end

# 1. Conservative Fallback: Returns exactly 0 at compile time!
@inline function evaluate_nc_jump(
    eq::HyperbolicPDE{D, M, Conservative}, f_L::State{M}, f_R::State{M}, dist_k::Space{D}
) where {D, M}
    return zero(Flux{D, M})
end

# 2. Non-Conservative Evaluation: Triggers the Path Integral!
@inline function evaluate_nc_jump(
    eq::HyperbolicPDE{D, M, <:NCRepresentation}, f_L::State{M}, f_R::State{M}, dist_k::Space{D}
) where {D, M}
    
    # We now access the path via the representation!
    jump = path_integral(eq, eq.rep.path, f_L, f_R)
    
    # Project the jump into the D-dimensional Flux tensor
    if D == 1
        return Flux{1, M}(( 0.5 * jump, ))
    else
        dist_mag = sqrt(sum(abs2, dist_k))
        n = dist_k ./ dist_mag
        return Flux{D, M}(ntuple(d -> 0.5 * jump * n[d], Val(D)))
    end
end

