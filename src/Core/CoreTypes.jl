## ------------------------------- HyperbolicPDEs -------------------------------
abstract type DifferentialOrder end
struct Order0 <: DifferentialOrder end
struct Order1 <: DifferentialOrder end

const DO0 = Order0() 
const DO1 = Order1()

const GAS_GAMMA_EULER = 1.4

abstract type AbstractPath{N} end
struct LinePath{N} <: AbstractPath{N} end

# A PDE in D dimensions with N variables.
abstract type HyperbolicPDE{D, N} end

# A helper for scalar PDEs (where N is always 1)
abstract type ScalarHyperbolicPDE{D} <: HyperbolicPDE{D, 1} end

# A helper for systems of PDEs
abstract type HyperbolicPDESystem{D, N} <: HyperbolicPDE{D, N} end
abstract type NCHyperbolicPDESystem{D, N} <: HyperbolicPDESystem{D, N} end

const DiagonalHyperbolicSystem{N, D} = NTuple{N, <: ScalarHyperbolicPDE{D}}

struct LinearAdvection{D, NM} <: HyperbolicPDESystem{D, NM}
    vel::SVector{D, SVector{NM, Float64}}
end

# Single constructor handles Floats, Tuples, and Tuples-of-Tuples!
function LinearAdvection(velocities)
    svec_vel = param2svec(velocities)
    
    # Extract D and NM directly from the generated SVector's type!
    D = length(svec_vel)
    NM = length(svec_vel[1])
    
    return LinearAdvection{D, NM}(svec_vel)
end

struct BurgersEquation{a} <: ScalarHyperbolicPDE{1} end
struct TestU3Equation{a} <: ScalarHyperbolicPDE{1} end
struct Euler1D <: HyperbolicPDESystem{1, 3} end
# --- Burgers Equation 2D ---
struct BurgersEquation2D <: ScalarHyperbolicPDE{2} end
## ------------------------------- Initial Conditions -------------------------------
abstract type InitialCondition end
abstract type SmoothInitialCondition <: InitialCondition end

struct Gauss{D, NM} <: SmoothInitialCondition
    a::SVector{NM, Float64}
    b::SVector{D, Float64}
    width::Float64
end

struct Box{D, NM} <: InitialCondition
    u_bg::SVector{NM, Float64}
    u_box::SVector{NM, Float64}
    mins::SVector{D, Float64}
    maxs::SVector{D, Float64}
end

struct Sine{D, NM} <: SmoothInitialCondition
    a::SVector{NM, Float64}
    period::SVector{D, Float64}
    c_offset::SVector{NM, Float64}
end

struct Riemann{D, NM} <: InitialCondition
    uL::SVector{NM, Float64}
    uR::SVector{NM, Float64}
    p0::SVector{D, Float64}
    n::SVector{D, Float64}
end

struct SRiemann{D, NM} <: SmoothInitialCondition
    uL::SVector{NM, Float64}
    uR::SVector{NM, Float64}
    p0::SVector{D, Float64}
    n::SVector{D, Float64}
    width::Float64
end

struct QuadrantRiemann{D, NM, N_states} <: InitialCondition
    u_states::NTuple{N_states, SVector{NM, Float64}} 
    p0::SVector{D, Float64}
end

const EulerShockTube = Riemann{Float64,NTuple{3,Float64}}

struct SimSetting
    tmax::Float64
    dt::Float64
    interpRange::Float64
    interpAlpha::Float64
    saveFreq::Int64
end

# Order important!
include("GridTypes.jl")
include("InterpolationTypes.jl")
include("TimestepperTypes.jl")

include("HyperbolicPDEs.jl")
include("CoreUtils.jl")