# =========================================================================
# CONSTANTS & DIFFERENTIAL ORDERS
# =========================================================================

abstract type DifferentialOrder end
struct Order0 <: DifferentialOrder end
struct Order1 <: DifferentialOrder end

const DO0 = Order0() 
const DO1 = Order1()

# =========================================================================
# PATHS
# =========================================================================
abstract type AbstractPath end
struct LinePath <: AbstractPath end

struct MappedPath{P <: AbstractPath} <: AbstractPath
    base_path::P
end
MappedPath() = MappedPath(LinePath())

struct NaiveAveragePath <: AbstractPath end

# =========================================================================
# EQUATION REPRESENTATIONS
# =========================================================================
abstract type EquationRepresentation end

# Conservative form (No path needed)
struct Conservative <: EquationRepresentation end

# Non-Conservative forms (Hold the path logic)
abstract type NCRepresentation{P <: AbstractPath} <: EquationRepresentation end

struct Primitive{P} <: NCRepresentation{P}
    path::P
end
Primitive() = Primitive(MappedPath())

struct Lagrangian{P} <: NCRepresentation{P}
    path::P
end
Lagrangian() = Lagrangian(MappedPath())

# =========================================================================
# HYPERBOLIC PDE ABSTRACT TYPES
# =========================================================================
# A PDE in D dimensions with M variables, Real type T, in Representation R
abstract type HyperbolicPDE{D, M, T, R <: EquationRepresentation} end

# =========================================================================
# CONCRETE PDEs & SMART CONSTRUCTORS
# =========================================================================

# --- Linear Advection ---
struct LinearAdvection{D, M, T, R} <: HyperbolicPDE{D, M, T, R}
    vel::Flux{D, M, T}
    rep::R
end

function LinearAdvection(velocities; rep::R = Conservative()) where {R <: EquationRepresentation}
    svec_vel = param2fvec(velocities)
    D = length(svec_vel)
    M = length(svec_vel[1])
    T = eltype(svec_vel[1]) # Extract T dynamically from the provided velocities
    return LinearAdvection{D, M, T, R}(Flux{D, M, T}(svec_vel), rep)
end

# --- Burgers Equation (Multi-Dimensional) ---
struct BurgersEquation{D, T, R} <: HyperbolicPDE{D, 1, T, R} 
    rep::R
end

# Removed the parameterless BurgersEquation{D}() fallback
BurgersEquation(::Val{D}, ::Type{T}, rep::R = Conservative()) where {D, T, R <: EquationRepresentation} = BurgersEquation{D, T, R}(rep)


# --- Euler Equation ---
struct EulerEquation{D, M, T, R} <: HyperbolicPDE{D, M, T, R}
    gamma::T
    rep::R
end

# Added gamma parameter and removed default Float64
function EulerEquation(::Val{D}, ::Type{T}, gamma::T = T(1.4), rep::R = Conservative()) where {D, T, R <: EquationRepresentation}
    return EulerEquation{D, D + 2, T, R}(gamma, rep)
end


# --- Test U3 Equation ---
struct TestU3Equation{a, T, R} <: HyperbolicPDE{1, 1, T, R} 
    rep::R
end

# Removed default Float64
TestU3Equation(a::Real, ::Type{T}, rep::R=Conservative()) where {T, R} = TestU3Equation{a, T, R}(rep)