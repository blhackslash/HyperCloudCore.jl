# --- Linear Advection ---
@inline velocity(eq::LinearAdvection{D}, u::SVector{1, Float64}) where {D} = Space{D}(eq.vel...)
# --- Multi-D Flux ---
# Returns: Flux{D,M}
@inline function flux(eq::LinearAdvection{D, M}, U::State{M}) where {D, M}
    # Builds the flux vector for each dimension 'd' using a generated tuple
    return Flux{D,M}(ntuple(d -> eq.vel[d] .* U, Val(D)))
end

# --- Burgers Equation 1D ---
BurgersEquation(a::Float64) = BurgersEquation{a}()
BurgersEquation() = BurgersEquation{0.0}()

@inline function velocity(::BurgersEquation{a}, u::SVector{1, Float64}) where {a}
    return SVector{1, Float64}((1.0 - a) * u[1])
end

@inline function flux(::BurgersEquation{a}, u::SVector{1, Float64}) where {a}
    return SVector{1, Float64}(0.5 * (1.0 - a) * u[1]^2)
end

@inline velocity(eq::BurgersEquation2D, u::SVector{1, Float64}) = SVector{2, Float64}(u[1], u[1])
@inline flux(eq::BurgersEquation2D, u::SVector{1, Float64}) = SVector{2, Float64}(0.5 * u[1]^2, 0.5 * u[1]^2)

# --- TestU3 Equation ---
TestU3Equation(a::Float64) = TestU3Equation{a}()

@inline function velocity(::TestU3Equation{a}, u::SVector{1, Float64}) where {a}
    return SVector{1, Float64}((1.0 - a) * u[1]^2)
end

@inline function flux(::TestU3Equation{a}, u::SVector{1, Float64}) where {a}
    return SVector{1, Float64}(0.33333 * (1.0 - a) * u[1]^3)
end

#----------------------------------#
# --- System Equation Examples --- #
#----------------------------------#

# --- 1D Euler Equations ---

function pressure_from_euler_conserved(rho::Float64, m::Float64, E::Float64)::Float64
    if rho < 1e-9; return 1e-9; end
    pressure = (GAS_GAMMA_EULER - 1.0) * (E - 0.5 * m^2 / rho)
    return max(pressure, 1e-9)
end

function flux(eq::Euler1D, U::SVector{3, Float64})::SVector{3, Float64}
    rho, m, E = U[1], U[2], U[3]
    if rho < 1e-9
        return SVector{3, Float64}(0.0, pressure_from_euler_conserved(1e-9, 0.0, 0.0), 0.0)
    end
    ux = m / rho
    p = pressure_from_euler_conserved(rho, m, E)
    return SVector{3, Float64}(m, m * ux + p, (E + p) * ux)
end

# --- 2D Euler Equations ---
struct Euler2D <: HyperbolicPDESystem{2, 4} end

function pressure_from_euler_conserved(U::SVector{4, Float64})::Float64
    rho, mx, my, E = U[1], U[2], U[3], U[4]
    if rho < 1e-9; return 1e-9; end
    pressure = (GAS_GAMMA_EULER - 1.0) * (E - 0.5 * (mx^2 + my^2) / rho)
    return max(pressure, 1e-9)
end

function flux(eq::Euler2D, U::SVector{4, Float64})::SVector{2, SVector{4, Float64}}
    rho, mx, my, E = U[1], U[2], U[3], U[4]
    if rho < 1e-9
        p_fallback = pressure_from_euler_conserved(SVector{4, Float64}(1e-9, 0.0, 0.0, 0.0))
        return SVector{2, SVector{4, Float64}}(
            SVector{4, Float64}(0.0, p_fallback, 0.0, 0.0),
            SVector{4, Float64}(0.0, 0.0, p_fallback, 0.0)
        )
    end
    p = pressure_from_euler_conserved(U)
    ux = mx / rho
    uy = my / rho
    
    F = SVector{4, Float64}(rho * ux, rho * ux^2 + p, rho * ux * uy, (E + p) * ux)
    G = SVector{4, Float64}(rho * uy, rho * ux * uy, rho * uy^2 + p, (E + p) * uy)
    
    # Returning an SVector of SVectors allows flux(eq, U)[d] to magically work!
    return SVector{2, SVector{4, Float64}}(F, G)
end
"""
Lagrangian Euler implementation using primitive variables, i.e. 
U = (ρ,u,p) and A(U) matrix: [[0,ρ,0],[0,0,1/ρ],[0,γp,0]]
"""
struct LEuler1D{P} <: HyperbolicPDESystem{1, 3}
    path::P
    function LEuler1D(;path::P = LinePath{3}()) where P <: AbstractPath{3}
        new{P}(path)
    end
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

# Implementation for LEuler1D (Primitive Euler)
# A(U) = [[0, rho, 0], [0, 0, 1/rho], [0, gamma*p, 0]]
@inline function A_matrix_times_vector(::LEuler1D, U::Tuple, v::Tuple)
    rho, u, p = U
    v1, v2, v3 = v
    # Result = [rho*v2, (1/rho)*v3, (gamma*p)*v2]
    return (rho * v2, (1.0 / rho) * v3, GAS_GAMMA_EULER * p * v2)
end
# In HyperbolicPDEs.jl or your test script

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

"""
Calculates the path integral ∫ A(Φ(s)) ∂sΦ ds numerically. [cite: 89, 249]
This version is specialized for N-component systems to ensure zero allocation.
"""
@inline function path_integral(eq::HyperbolicPDE{D, N}, uL::NTuple{N, Float64}, uR::NTuple{N, Float64}) where {D, N}
    nodes, weights = simpson_3_point()
    path = eq.path # Assumes path is stored in the PDE struct

    # Initialize the integral tuple with zeros
    integral = ntuple(_ -> 0.0, Val(N))

    # Loop over 5 quadrature points
    for i in eachindex(nodes)
        s = nodes[i]
        w = weights[i]
        
        # Phi(s) and dPhi(s) [cite: 79]
        U_s = path(s, uL, uR, DO0)
        dU_s = path(s, uL, uR, DO1)
        # Compute A(U_s) * dU_s
        term = A_matrix_times_vector(eq, U_s, dU_s)
        # Accumulate: integral += w * term
        integral = ntuple(k -> integral[k] + w * term[k], Val(N))
        @debug "Integral Calculation" group=:quad node=i integrand=term integral=integral Φ=U_s Dϕ=dU_s
    end
    if maximum(abs.(integral)) > 1000; error("Integral too large!") end
    return integral
end

@inline function prim2cons(eq, U)
    return U
end

@inline function cons2prim(eq, U)
    return U
end

# Add these helpers to convert between states
@inline function prim2cons(::LEuler1D{P}, U::State{M}) where {P,M}
    rho, u, p = U
    E = p / (GAS_GAMMA_EULER - 1.0) + 0.5 * rho * u^2
    return State{M}(rho, rho * u, E)
end

@inline function cons2prim(::LEuler1D{P}, W::State{M}) where {P,M}
    rho, m, E = W
    safe_rho = max(rho, 1e-7)
    u = m / safe_rho
    p = (GAS_GAMMA_EULER - 1.0) * (E - 0.5 * m^2 / safe_rho)
    return State{M}(safe_rho, u, max(p, 1e-7))
end

# Modified Path Integral
@inline function path_integral(eq::LEuler1D{P}, uL::NTuple{3, Float64}, uR::NTuple{3, Float64}) where {P<:AbstractPath}
    nodes, weights = gauss_lobatto_5() 
    
    # 1. Convert endpoints to Conservative variables
    wL = prim2cons(eq, uL)
    wR = prim2cons(eq, uR)

    integral = (0.0, 0.0, 0.0)

    for i in eachindex(nodes)
        s = nodes[i]
        w = weights[i]
        
        # 2. Linearly interpolate in CONSERVATIVE space
        w_s = ntuple(k -> wL[k] + s * (wR[k] - wL[k]), Val(3))
        
        # Derivative of conservative path with respect to s
        dw_s = ntuple(k -> wR[k] - wL[k], Val(3))
        
        # 3. Map the state and the derivative BACK to primitive space
        # (Using finite differences for the mapped derivative is safest and easiest)
        eps_fd = 1e-6
        w_s_plus = ntuple(k -> w_s[k] + eps_fd * dw_s[k], Val(3))
        
        U_s = cons2prim(eq,w_s)
        U_s_plus = cons2prim(eq,w_s_plus)
        dU_s = ntuple(k -> (U_s_plus[k] - U_s[k]) / eps_fd, Val(3))
        
        # 4. Calculate the non-conservative product using the mapped path
        term = A_matrix_times_vector(eq, U_s, dU_s)
        integral = ntuple(k -> integral[k] + w * term[k], Val(3))
    end
    
    return integral
end