const GAS_GAMMA_EULER = 1.4

abstract type EquationRepresentation end

# Conservative form (No path needed)
struct Conservative <: EquationRepresentation end

# Non-Conservative forms (Hold the path logic)
abstract type NCRepresentation{P <: AbstractPath} <: EquationRepresentation end

struct Primitive{P} <: NCRepresentation{P}
    path::P
end
# Smart constructors default to the MappedPath for Primitive/Lagrangian
Primitive() = Primitive(MappedPath())

struct Lagrangian{P} <: NCRepresentation{P}
    path::P
end
Lagrangian() = Lagrangian(MappedPath())
# =========================================================================
# HYPERBOLIC PDE DEFINITIONS
# =========================================================================
# A PDE in D dimensions with M variables, in Representation R
abstract type HyperbolicPDE{D, M, R <: EquationRepresentation} end

# Helper for scalar PDEs (M is always 1)
abstract type ScalarHyperbolicPDE{D, R} <: HyperbolicPDE{D, 1, R} end

# Helper for systems of PDEs
abstract type HyperbolicPDESystem{D, M, R} <: HyperbolicPDE{D, M, R} end

struct LinearAdvection{D, M, R} <: HyperbolicPDE{D, M, R}
    vel::Flux{D, M}
    rep::R
end
# Constructor forces Conservative
function LinearAdvection(velocities)
    svec_vel = param2fvec(velocities)
    D = length(svec_vel)
    M = length(svec_vel[1])
    return LinearAdvection{D, M, Conservative}(Flux{D, M}(svec_vel), Conservative())
end

struct BurgersEquationMD{D, R} <: ScalarHyperbolicPDE{D, R} 
    rep::R
end
BurgersEquationMD{D}() where {D} = BurgersEquationMD{D, Conservative}(Conservative())

struct EulerEquation{D, M, R} <: HyperbolicPDESystem{D, M, R}
    rep::R
end

# Smart Constructor: You just pass the Dimension and the Representation instance!
function EulerEquation(::Val{D}, rep::R = Conservative()) where {D, R <: EquationRepresentation}
    return EulerEquation{D, D + 2, R}(rep)
end

function LinearAdvection(velocities)
    svec_vel = param2fvec(velocities)
    D = length(svec_vel)
    M = length(svec_vel[1])
    # It is already an SVector of SVectors, just cast it!
    return LinearAdvection{D, M}(Flux{D, M}(svec_vel))
end
# 1. Smart Constructor
BurgersEquationMD(::Val{D}, rep::R = Conservative()) where {D, R <: EquationRepresentation} = BurgersEquationMD{D, R}(rep)

# 2. Identity Conversions (Primitive 'u' == Conservative 'u')
@inline prim2cons(::BurgersEquationMD, u::State{1}) = u
@inline cons2prim(::BurgersEquationMD, w::State{1}) = w

# 3. Conservative Flux for Primitive Representation is EXACTLY ZERO
@inline flux(::BurgersEquationMD{D, <:Primitive}, u::State{1}) where {D} = zero(Flux{D, 1})

# 4. Non-Conservative Matrix-Vector Product: A(u) * du = u * du
@inline function A_matrix_times_vector(::BurgersEquationMD{D, <:Primitive}, u::State{1}, du::State{1}) where {D}
    return State{1}(u[1] * du[1])
end

@inline function flux(eq::LinearAdvection{D, M}, U::State{M}) where {M, D}
    # Direct component-wise multiplication per spatial column!
    return Flux{D, M}(ntuple(d -> eq.vel[d] .* U, Val(D)))
end
@inline function velocity(eq::LinearAdvection{D, M}, U::State{M}) where {M, D}
    # Direct component-wise multiplication per spatial column!
    return eq.vel
end
@inline function velocity(eq::LinearAdvection{D, 1}, U::State{1}) where {D}
    # Direct component-wise multiplication per spatial column!
    return Space{D}(ntuple(i->eq.vel[i][1],Val(D)))
end

@inline function sort_flux(f_i::State{M}, f_j::State{M}, F_i::Flux{D, M}, F_j::Flux{D, M}, dist_k::Space{D}) where {D, M}
    # Builds the arrays natively column-by-column
    f_L = Flux{D, M}(ntuple(d -> dist_k[d] > 0 ? f_i : f_j, Val(D)))
    f_R = Flux{D, M}(ntuple(d -> dist_k[d] > 0 ? f_j : f_i, Val(D)))
    
    F_L = Flux{D, M}(ntuple(d -> dist_k[d] > 0 ? F_i[d] : F_j[d], Val(D)))
    F_R = Flux{D, M}(ntuple(d -> dist_k[d] > 0 ? F_j[d] : F_i[d], Val(D)))
    
    return f_L, f_R, F_L, F_R
end

@inline function flux(eq::BurgersEquationMD{D}, u::State{1}) where {D}
    # F(u) = 1/2 u^2 in all D spatial directions
    # Builds an SVector of length D, where each element is a State{1}
    return Flux{D, 1}(ntuple(_ -> 0.5 * u.^2, Val(D)))
end

@inline function velocity(eq::BurgersEquationMD{D}, u::State{1}) where {D}
    # f'(u) = u in all D spatial directions
    return Space{D}(ntuple(_ -> u[1], Val(D)))
end

# --- Burgers Equation 1D ---
BurgersEquation(a::Float64) = BurgersEquation{a}()
BurgersEquation() = BurgersEquation{0.0}()

@inline function velocity(::BurgersEquation{a}, u::SVector{1, Float64}) where {a}
    return (1.0 - a) * u
end

@inline function flux(::BurgersEquation{a}, u::State{1}) where {a}
    return SVector{1,State{1}}((0.5 * (1.0 - a) * u.*u,))
end
# --- TestU3 Equation ---
TestU3Equation(a::Float64) = TestU3Equation{a}()

@inline function velocity(::TestU3Equation{a}, u::SVector{1, Float64}) where {a}
    return SVector{1, Float64}((1.0 - a) * u[1]^2)
end

@inline function flux(::TestU3Equation{a}, u::SVector{1, Float64}) where {a}
    return SVector{1, Float64}(0.33333 * (1.0 - a) * u[1]^3)
end

# 1. Conservative Setup
# Default path is a standard straight LinePath
function EulerEquation(::Val{D}, ::Type{Conservative}; path::AbstractPath = LinePath()) where {D}
    return EulerEquation{D, D+2, Conservative, typeof(path)}(path)
end

# 2. Primitive & Lagrangian Setup
# Default path is a LinePath wrapped in our new MappedPath decorator
function EulerEquation(::Val{D}, ::Type{Rep}; path::AbstractPath = MappedPath()) where {D, Rep <: Union{Primitive, Lagrangian}}
    return EulerEquation{D, D+2, Rep, typeof(path)}(path)
end

# =========================================================================
# UNIVERSAL D-DIMENSIONAL STATE CONVERSIONS
# =========================================================================
@inline function prim2cons(eq::EulerEquation{D, M}, V::State{M}) where {D, M}
    rho = V[1]
    # Dynamically extracts D velocity components!
    u = SVector{D, Float64}(ntuple(d -> V[1+d], Val(D))) 
    p = V[M]
    
    m = rho .* u
    E = p / (GAS_GAMMA_EULER - 1.0) + 0.5 * rho * sum(abs2, u)
    
    return State{M}(rho, m..., E)
end

@inline function cons2prim(eq::EulerEquation{D, M}, U::State{M}) where {D, M}
    rho = max(U[1], 1e-7)
    m = SVector{D, Float64}(ntuple(d -> U[1+d], Val(D)))
    E = U[M]
    
    u = m ./ rho
    p = (GAS_GAMMA_EULER - 1.0) * (E - 0.5 * sum(abs2, m) / rho)
    
    return State{M}(rho, u..., max(p, 1e-7))
end

# =========================================================================
# UNIVERSAL D-DIMENSIONAL CONSERVATIVE FLUX
# =========================================================================
@inline function flux(eq::EulerEquation{D, M, Conservative}, U::State{M}) where {D, M}
    V = cons2prim(eq, U)
    rho = V[1]
    u = SVector{D, Float64}(ntuple(d -> V[1+d], Val(D)))
    p = V[M]
    E = U[M]
    
    # Builds an SVector of D columns, seamlessly handling 1D, 2D, or 3D
    return Flux{D, M}(ntuple(Val(D)) do d
        ud = u[d]
        mass_flux = rho * ud
        mom_flux = ntuple(i -> rho * u[i] * ud + (i == d ? p : 0.0), Val(D))
        energy_flux = ud * (E + p)
        
        State{M}(mass_flux, mom_flux..., energy_flux)
    end)
end