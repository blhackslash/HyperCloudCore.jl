export Primitive, Conservative, Lagrangian, DifferentialOrder, Order0, Order1
export LinePath, NaiveAveragePath, MappedPath

"""
    Order0
    Order1

Dispatch singletons representing the differential order requested during path integral evaluation.
- `Order0`: Requests the interpolated state along the path.
- `Order1`: Requests the derivative of the state with respect to the path parameter.
"""
abstract type DifferentialOrder end
struct Order0 <: DifferentialOrder end
struct Order1 <: DifferentialOrder end

const DO0 = Order0() 
const DO1 = Order1()

"""
    GaussLobatto5
    Simpson3

Numerical quadrature rules utilized by the path integrator.
- Provide `get_nodes_weights` functions to retrieve the specific interpolation nodes `s` and quadrature weights `w` tailored to the requested floating-point precision.
"""
struct GaussLobatto5 <: PathIntegrator end
struct Simpson3 <: PathIntegrator end

"""
    LinePath <: AbstractPath
    MappedPath{P} <: AbstractPath
    NaiveAveragePath <: AbstractPath

Path definitions used to evaluate non-conservative products.
- `LinePath`: Performs a direct, straight-line interpolation in the active state space, fully unrolled by the compiler.
- `MappedPath`: Evaluates paths across variable representations. It transforms states into conservative variables, interpolates via the `base_path`, and transforms the result back into primitive variables. Evaluates the `Order1` derivative using a finite-difference approximation of the chain rule.
- `NaiveAveragePath`: A simplified path returning a constant arithmetic average for the state and a standard jump for the derivative, independent of the path parameter `s`.
"""
struct LinePath <: AbstractPath end

struct MappedPath{P <: AbstractPath} <: AbstractPath
    base_path::P
end
MappedPath() = MappedPath(LinePath())

struct NaiveAveragePath <: AbstractPath end

struct Primitive{PI <: PathIntegral} <: NCRepresentation{PI}
    integral::PI
end
Primitive() = Primitive(PathIntegral(MappedPath(), GaussLobatto5()))

struct Lagrangian{PI <: PathIntegral} <: NCRepresentation{PI}
    integral::PI
end
Lagrangian() = Lagrangian(PathIntegral(MappedPath(), GaussLobatto5()))

# DO0 (The State): Returns a constant average regardless of 's'
@inline function (::NaiveAveragePath)(eq::HyperbolicPDE, s::T, u_L::State{M, T}, u_R::State{M, T}, ::Order0) where {M, T}
    return T(0.5) * (u_L + u_R)
end

# DO1 (The Derivative): Fakes the derivative to be the standard jump
@inline function (::NaiveAveragePath)(eq::HyperbolicPDE, s::T, u_L::State{M, T}, u_R::State{M, T}, ::Order1) where {M, T}
    return u_R - u_L
end

# Pure SVector math, completely unrolled by the compiler!
@inline (::LinePath)(eq::HyperbolicPDE, s::T, u_L::State{M, T}, u_R::State{M, T}, ::Order0) where {M, T} = u_L + s * (u_R - u_L)
@inline (::LinePath)(eq::HyperbolicPDE, s::T, u_L::State{M, T}, u_R::State{M, T}, ::Order1) where {M, T} = u_R - u_L

# 0th Derivative: Map to Cons -> Evaluate Base Path -> Map to Prim
@inline function (mp::MappedPath)(eq::HyperbolicPDE, s::T, v_L::State{M, T}, v_R::State{M, T}, ::Order0) where {M, T}
    w_L = prim2cons(eq, v_L)
    w_R = prim2cons(eq, v_R)
    
    w_s = mp.base_path(eq, s, w_L, w_R, DO0)
    return cons2prim(eq, w_s)
end

# 1st Derivative: The Chain Rule (∂V/∂W * dW/ds)
@inline function (mp::MappedPath)(eq::HyperbolicPDE, s::T, v_L::State{M, T}, v_R::State{M, T}, ::Order1) where {M, T}
    w_L = prim2cons(eq, v_L)
    w_R = prim2cons(eq, v_R)
    
    w_s  = mp.base_path(eq, s, w_L, w_R, DO0)
    dw_s = mp.base_path(eq, s, w_L, w_R, DO1)
    
    eps_fd = T(1e-6)
    V_s      = cons2prim(eq, w_s)
    V_s_plus = cons2prim(eq, w_s + eps_fd * dw_s)
    
    return (V_s_plus - V_s) / eps_fd
end

@inline function get_nodes_weights(::GaussLobatto5, ::Type{T}) where {T}
    s2_offset = T(0.5) * sqrt(T(3)/T(7))
    nodes = (zero(T), T(0.5) - s2_offset, T(0.5), T(0.5) + s2_offset, one(T))
    weights = (T(1/20), T(49/180), T(16/45), T(49/180), T(1/20))
    return nodes, weights
end

@inline function get_nodes_weights(::Simpson3, ::Type{T}) where {T}
    nodes = (zero(T), T(0.5), one(T))
    weights = (T(1/6), T(4/6), T(1/6))
    return nodes, weights
end

"""
    (pi::PathIntegral)(eq::HyperbolicPDE, u_L::State, u_R::State, d::Int)

Evaluates the non-conservative path integral across an interface for a specific spatial dimension `d`.

# Details
- Retrieves the quadrature nodes and weights defined by `pi.integrator`.
- Evaluates the path state (`U_s`) and path derivative (`dU_s`) at each node.
- Computes the advective velocity (Jacobian) at `U_s` and numerically integrates the non-conservative product.
- Throws a hard error if the resulting integral exceeds 1000, indicating severe numerical instability.
"""
@inline function (pi::PathIntegral)(eq::HyperbolicPDE{D, M, T}, u_L::State{M, T}, u_R::State{M, T}, d::Int) where {D, M, T}
    # Dispatch to get the correct rule based on the integrator
    nodes, weights = get_nodes_weights(pi.integrator, T) 
    
    integral = zero(State{M, T})

    for i in eachindex(nodes)
        s = nodes[i]
        w = weights[i]
        
        U_s  = pi.path(eq, s, u_L, u_R, DO0)
        dU_s = pi.path(eq, s, u_L, u_R, DO1)
        
        term = velocity(eq, U_s, d) * dU_s
        integral += w * term
    end
    
    if maximum(abs.(integral)) > T(1000)
        error("Integral too large!")
    end
    return integral
end

# 1. Conservative Fallback
"""
    evaluate_nc_jump(eq::HyperbolicPDE, f_L::Flux, f_R::Flux, dist_k::Space)

Computes the non-conservative jump across a reconstructed particle interface.

# Details
- **Conservative Fallback:** For systems strictly utilizing a `Conservative` representation, this safely evaluates as a zero-cost `zero(Flux)`.
- **Non-Conservative Execution:** For `NCRepresentation` systems, it invokes the path integral functor for each dimension and properly scales the jump by `0.5` while applying the upwind sign derived from `dist_k`.
"""
@inline function evaluate_nc_jump(
    eq::HyperbolicPDE{D, M, T, Conservative}, f_L::Flux{D, M, T}, f_R::Flux{D, M, T}, dist_k::Space{D, T}
) where {D, M, T}
    return zero(Flux{D, M, T})
end

# 2. Non-Conservative Evaluation
@inline function evaluate_nc_jump(
    eq::HyperbolicPDE{D, M, T, <:NCRepresentation}, f_L::Flux{D, M, T}, f_R::Flux{D, M, T}, dist_k::Space{D, T}
) where {D, M, T}    
    return Flux{D, M, T}(ntuple(Val(D)) do d
        # Invoke the functor!
        jump_d = eq.rep.integral(eq, f_L[d], f_R[d], d)
        sign_i = dist_k[d] >= zero(T) ? one(T) : -one(T)
        
        T(0.5) * jump_d * sign_i
    end)
end