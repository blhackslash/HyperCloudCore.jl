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

function path_integral(eq::HyperbolicPDE, u_left::Tuple, u_right::Tuple)
    error("path_integral not implemented for $(typeof(eq))")
end

function A_matrix_times_vector(eq::HyperbolicPDE, U::Tuple, v::Tuple)
    error("A_matrix_times_vector not implemented for $(typeof(eq))")
end

# =========================================================================
# NON-CONSERVATIVE MATVECS (1D)
# =========================================================================

# 1. LAGRANGIAN (Material Derivative Frame)
@inline function A_matrix_times_vector(eq::EulerEquation{1, 3, T, LR}, V::State{3, T}, dV::State{3, T}) where {T, LR <: Lagrangian}
    rho, u, p = V[1], V[2], V[3]
    drho, du, dp = dV[1], dV[2], dV[3]
    
    return State{3, T}(rho * du, dp / rho, eq.gamma * p * du)
end

# 2. PRIMITIVE (Eulerian Frame)
@inline function A_matrix_times_vector(eq::EulerEquation{1, 3, T, PR}, V::State{3, T}, dV::State{3, T}) where {T, PR <: Primitive}
    rho, u, p = V[1], V[2], V[3]
    drho, du, dp = dV[1], dV[2], dV[3]
    
    return State{3, T}(
        u * drho + rho * du,
        u * du + dp / rho,
        eq.gamma * p * du + u * dp
    )
end

@inline function gauss_lobatto_5(::Type{T}) where {T}
    s2_offset = T(0.5) * sqrt(T(3)/T(7))
    nodes = (
        zero(T), 
        T(0.5) - s2_offset, 
        T(0.5), 
        T(0.5) + s2_offset, 
        one(T)
    )
    
    weights = (
        T(1/20),
        T(49/180),
        T(16/45),
        T(49/180), 
        T(1/20)
    )
    return nodes, weights
end

@inline function simpson_3_point(::Type{T}) where {T}
    nodes = (zero(T), T(0.5), one(T))
    weights = (T(1/6), T(4/6), T(1/6))
    return nodes, weights
end

@inline function path_integral(eq::HyperbolicPDE{D, M, T}, path::AbstractPath, u_L::State{M, T}, u_R::State{M, T}) where {D, M, T}
    nodes, weights = gauss_lobatto_5(T) 
    
    integral = zero(State{M, T})

    for i in eachindex(nodes)
        s = nodes[i]
        w = weights[i]
        
        U_s  = path(eq, s, u_L, u_R, DO0)
        dU_s = path(eq, s, u_L, u_R, DO1)
        
        term = A_matrix_times_vector(eq, U_s, dU_s)
        integral += w * term
    end
    
    if maximum(abs.(integral)) > T(1000)
        error("Integral too large!")
    end
    return integral
end

# 1. Conservative Fallback
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
        jump_d = path_integral(eq, eq.rep.path, f_L[d], f_R[d])
        sign_i = dist_k[d] >= zero(T) ? one(T) : -one(T)
        
        T(0.5) * jump_d * sign_i
    end)
end