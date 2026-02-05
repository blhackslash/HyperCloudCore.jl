module HyperbolicPDEs

export ScalarHyperbolicPDE, LinearAdvection, BurgersEquation, BurgersEquation2D, TestU3Equation,
       velocity, flux, HyperbolicPDESystem, Euler1D, Euler2D, pressure_from_euler_conserved,
       HyperbolicPDE, n_dimensions, DiagonalHyperbolicSystem, LagrangianEuler1D

# A PDE in D dimensions with N variables.
abstract type HyperbolicPDE{D, N} end

# A helper for scalar PDEs (where N is always 1)
abstract type ScalarHyperbolicPDE{D} <: HyperbolicPDE{D, 1} end

# A helper for systems of PDEs
abstract type HyperbolicPDESystem{D, N} <: HyperbolicPDE{D, N} end

const DiagonalHyperbolicSystem{N, D} = NTuple{N, <:ScalarHyperbolicPDE{D}}

#----------------------------------#
# --- Scalar Equation Examples --- #
#----------------------------------#

struct LinearAdvection{D} <: ScalarHyperbolicPDE{D} 
    vel::NTuple{D, Float64} # Store velocity as a tuple of length D
end

# Constructors for convenience
LinearAdvection(vel::Real) = LinearAdvection{1}((Float64(vel),))
LinearAdvection(vel::Tuple{<:Real, <:Real}) = LinearAdvection{2}(Float64.(vel))

# --- REFINEMENT 1: Unify `velocity` and `flux` for LinearAdvection ---

# For 1D, return the scalar velocity, not a 1-tuple
@inline velocity(eq::LinearAdvection{1}, u::Float64) = eq.vel[1]
# For 2D, return the tuple
@inline velocity(eq::LinearAdvection{2}, u::Float64) = eq.vel

# Use broadcasting (`.*`) to create one `flux` method for any dimension D
@inline flux(eq::LinearAdvection{2}, u::Float64) = (eq.vel[1] * u, eq.vel[2] * u)
@inline flux(eq::LinearAdvection{1}, u::Float64) = eq.vel[1] * u


struct BurgersEquation2D <: ScalarHyperbolicPDE{2} end
@inline velocity(eq::BurgersEquation2D, u::Float64) = (u, u)
@inline flux(eq::BurgersEquation2D, u::Float64) = (0.5 * u^2, 0.5 * u^2)

# 1. Define the Parametric Struct
# The 'A' parameter is part of the type definition.
struct BurgersEquation{a} <: ScalarHyperbolicPDE{1} end

# 2. Define Outer Constructors
# This allows you to call BurgersEquation(0.5)
BurgersEquation(a::Float64) = BurgersEquation{a}()

# This allows you to call BurgersEquation() and get the classic behavior (A=0.0)
BurgersEquation() = BurgersEquation{0.0}()



# 3. Define the Physics using the Type Parameter
# We extract 'A' from the type using the 'where {A}' syntax.

@inline function velocity(::BurgersEquation{a}, u::Float64) where {a}
    # Classic case (A=0): returns u
    # Generalized case: returns (1-A) * u
    return (1.0 - a) * u
end

@inline function flux(::BurgersEquation{a}, u::Float64) where {a}
    # Classic case (A=0): returns 0.5 * u^2
    # Generalized case: returns 0.5 * (1-A) * u^2
    return .5 * (1.0 - a) * u^2
end
struct TestU3Equation{a} <: ScalarHyperbolicPDE{1} end

TestU3Equation(a::Float64) = TestU3Equation{a}()

@inline function velocity(::TestU3Equation{a}, u::Float64) where {a}
    # Classic case (A=0): returns u
    # Generalized case: returns (1-A) * u
    return (1.0 - a) * u^2
end

@inline function flux(::TestU3Equation{a}, u::Float64) where {a}
    # Classic case (A=0): returns 0.5 * u^2
    # Generalized case: returns 0.5 * (1-A) * u^2
    return 0.33333 * (1.0 - a) * u^3
end

#--------------------------------#
# --- System Equation Examples --- #
#--------------------------------#

const GAS_GAMMA_EULER = 1.4 # --- REFINEMENT 2: Use a single constant ---

# --- 1D Euler Equations ---
struct Euler1D <: HyperbolicPDESystem{1, 3} end

function pressure_from_euler_conserved(rho::Float64, m::Float64, E::Float64)::Float64
    if rho < 1e-9; return 1e-9; end
    pressure = (GAS_GAMMA_EULER - 1.0) * (E - 0.5 * m^2 / rho)
    return max(pressure, 1e-9)
end

function flux(eq::Euler1D, U)::NTuple{3, Float64}
    rho, m, E = U
    if rho < 1e-9; return (0.0, pressure_from_euler_conserved(1e-9, 0.0, 0.0), 0.0); end
    ux = m / rho
    p = pressure_from_euler_conserved(rho, m, E)
    return (m, m * ux + p, (E + p) * ux)
end

# --- 1D Lagrangian Euler Equations ---
struct LagrangianEuler1D <: HyperbolicPDESystem{1, 3} end

# Helper to get pressure from Lagrangian state
function pressure_from_lagrangian(V::Float64, u::Float64, e::Float64)::Float64
    # e is total specific energy: internal energy + kinetic energy
    # e = i + 0.5 * u^2  => i = e - 0.5 * u^2
    internal_energy = e - 0.5 * u^2
    
    if V < 1e-9; V = 1e-9; end
    
    # P = (gamma - 1) * rho * internal_energy = (gamma - 1) * i / V
    pressure = (GAS_GAMMA_EULER - 1.0) * internal_energy / V
    return max(pressure, 1e-9)
end

function flux(eq::LagrangianEuler1D, W)::NTuple{3, Float64}
    # W is the state vector: (Specific Volume, Velocity, Total Specific Energy)
    V, u, e = W
    
    p = pressure_from_lagrangian(V, u, e)
    
    # The flux vector in Lagrangian coordinates (d/dt W + d/dm F = 0):
    # 1. dV/dt - du/dm = 0      => Flux is -u
    # 2. du/dt + dp/dm = 0      => Flux is p
    # 3. de/dt + d(p*u)/dm = 0  => Flux is p*u
    
    return (-u, p, p * u)
end

# --- 2D Euler Equations ---
struct Euler2D <: HyperbolicPDESystem{2, 4} end

function pressure_from_euler_conserved(U)::Float64
    rho, mx, my, E = U
    if rho < 1e-9; return 1e-9; end
    pressure = (GAS_GAMMA_EULER - 1.0) * (E - 0.5 * (mx^2 + my^2) / rho)
    return max(pressure, 1e-9)
end

function flux(eq::Euler2D, U)::NTuple{2, NTuple{4, Float64}}
    rho, mx, my, E = U
    if rho < 1e-9
        # --- REFINEMENT 3: Clean up redundant calls ---
        p_fallback = pressure_from_euler_conserved((1e-9, 0.0, 0.0, 0.0))
        return ((0.0, p_fallback, 0.0, 0.0), (0.0, 0.0, p_fallback, 0.0))
    end
    p = pressure_from_euler_conserved(U)
    ux = mx / rho
    uy = my / rho
    F = (rho * ux, rho * ux^2 + p, rho * ux * uy, (E + p) * ux)
    G = (rho * uy, rho * ux * uy, rho * uy^2 + p, (E + p) * uy)
    return (F, G)
end

"""
    sound_speed(rho, p)
Calculates the local speed of sound.
"""
@inline function sound_speed(rho::Real, p::Real)
    return sqrt(GAS_GAMMA_EULER * p / rho)
end

"""
    velocity(eq::Euler1D, U)
Returns the characteristic speeds (eigenvalues) for the 1D Euler system.
These are used for wave speeds and numerical flux dissipation (e.g., Rusanov).
"""
function velocity(eq::Euler1D, U)::NTuple{3, Float64}
    rho, m, E = U
    if rho < 1e-9
        return (0.0, 0.0, 0.0)
    end
    
    p = pressure_from_euler_conserved(rho, m, E)
    u = m / rho
    c = sound_speed(rho, p)
    
    return (u - c, u, u + c)
end

"""
    fluid_velocity(eq::Euler1D, U)
Returns the macroscopic fluid velocity (u). 
Useful for Lagrangian grid movement (v_grid = u).
"""
@inline function fluid_velocity(eq::Euler1D, U)
    return U[2] / U[1] # m / rho
end

"""
    velocity(eq::Euler2D, U)
Returns the eigenvalues in the x and y coordinate directions.
Format: ((λx1, λx2, λx3, λx4), (λy1, λy2, λy3, λy4))
"""
function velocity(eq::Euler2D, U)::NTuple{2, NTuple{4, Float64}}
    rho, mx, my, E = U
    if rho < 1e-9
        zero_vec = (0.0, 0.0, 0.0, 0.0)
        return (zero_vec, zero_vec)
    end
    
    p = pressure_from_euler_conserved(U)
    ux = mx / rho
    uy = my / rho
    c = sound_speed(rho, p)
    
    # Eigenvalues for the x-direction (F flux)
    vals_x = (ux - c, ux, ux, ux + c)
    
    # Eigenvalues for the y-direction (G flux)
    vals_y = (uy - c, uy, uy, uy + c)
    
    return (vals_x, vals_y)
end

"""
    fluid_velocity(eq::Euler2D, U)
Returns the macroscopic fluid velocity vector (ux, uy).
"""
@inline function fluid_velocity(eq::Euler2D, U)
    return (U[2] / U[1], U[3] / U[1]) # (mx/rho, my/rho)
end

end # Module