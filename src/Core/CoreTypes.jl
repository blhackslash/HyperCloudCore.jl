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

struct LinearAdvection{D} <: ScalarHyperbolicPDE{D} 
    vel::NTuple{D, Float64} # Store velocity as a tuple of length D
end

struct BurgersEquation{a} <: ScalarHyperbolicPDE{1} end
struct TestU3Equation{a} <: ScalarHyperbolicPDE{1} end
struct Euler1D <: HyperbolicPDESystem{1, 3} end

## ------------------------------- Initial Conditions -------------------------------
abstract type InitialCondition end
abstract type SmoothInitialCondition <: InitialCondition end
abstract type ShockInitialCondition <: InitialCondition end

"Gaussian distribution for scalar or system states."
struct Gauss{T, S} <: SmoothInitialCondition
    a::S      # Amplitude (can be a scalar or a vector/tuple)
    b::T      # Center (Float64 for 1D, NTuple for 2D)
    width::Float64
end

"Box distribution for scalar or system states."
struct Box{S} <: ShockInitialCondition
    u_background::S
    u_box::S
    x_start::Float64
    x_end::Float64
    y_start::Union{Float64, Nothing}
    y_end::Union{Float64, Nothing}
end

"Sine wave for scalar states (systems would require more specific definition)."
struct Sine <: SmoothInitialCondition
    a::Float64
    b_period::Float64
    c_offset::Float64
end

"Riemann problem (shock/rarefaction) for scalar or system states in 1D or 2D."
struct Riemann{T, S} <: ShockInitialCondition
    uL::S
    uR::S
    p0::T  # 1D: x0 position. 2D: point on line.
    n::T   # 1D: defaults to 1.0. 2D: normal vector.
end

"Smoothed Riemann problem (arctan) for scalar or system states."
struct SRiemann{T, S} <: SmoothInitialCondition
    uL::S
    uR::S
    x0::T      # Center of the transition
    width::T   # Smoothing width (steepness)
end

struct QuadrantRiemann{D, M, T} <: ShockInitialCondition
    u_states::NTuple{D,NTuple{M,Float64}} # Vector of states for each quadrant
    p0::T               # Center point of the quadrants
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