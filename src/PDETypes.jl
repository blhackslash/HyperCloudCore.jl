# =========================================================================
# CONSTANTS & DIFFERENTIAL ORDERS
# =========================================================================
const GAS_GAMMA_EULER = 1.4

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
# A PDE in D dimensions with M variables, in Representation R
abstract type HyperbolicPDE{D, M, R <: EquationRepresentation} end

# =========================================================================
# CONCRETE PDEs & SMART CONSTRUCTORS
# =========================================================================

# --- Linear Advection ---
struct LinearAdvection{D, M, R} <: HyperbolicPDE{D, M, R}
    vel::Flux{D, M}
    rep::R
end

function LinearAdvection(velocities; rep::R = Conservative()) where {R <: EquationRepresentation}
    svec_vel = param2fvec(velocities)
    D = length(svec_vel)
    M = length(svec_vel[1])
    return LinearAdvection{D, M, R}(Flux{D, M}(svec_vel), rep)
end

# --- Burgers Equation (Multi-Dimensional) ---
struct BurgersEquation{D, R} <: HyperbolicPDE{D, 1, R} 
    rep::R
end

BurgersEquation{D}() where {D} = BurgersEquation{D, Conservative}(Conservative())
BurgersEquation(::Val{D}, rep::R = Conservative()) where {D, R <: EquationRepresentation} = BurgersEquation{D, R}(rep)

# --- Euler Equation ---
struct EulerEquation{D, M, R} <: HyperbolicPDE{D, M, R}
    rep::R
end

# Unified Smart Constructor: Pass the Dimension and the Representation instance
function EulerEquation(::Val{D}, rep::R = Conservative()) where {D, R <: EquationRepresentation}
    return EulerEquation{D, D + 2, R}(rep)
end

# --- Test U3 Equation ---
struct TestU3Equation{a} end
TestU3Equation(a::Float64) = TestU3Equation{a}()