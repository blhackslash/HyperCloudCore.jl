## ------------------------------- SVector Types -------------------------------
const Space{D} = SVector{D, Float64}
const State{M} = SVector{M, Float64}
const Flux{D, M} = SVector{D, State{M}}
const Kinetic{K} = SVector{K, Float64}

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
abstract type HyperbolicPDE{D, M} end

# A helper for scalar PDEs (where N is always 1)
abstract type ScalarHyperbolicPDE{D} <: HyperbolicPDE{D, 1} end

# A helper for systems of PDEs
abstract type HyperbolicPDESystem{D, M} <: HyperbolicPDE{D, M} end
abstract type NCHyperbolicPDESystem{D, M} <: HyperbolicPDESystem{D, M} end

const DiagonalHyperbolicSystem{M, D} = NTuple{M, <: ScalarHyperbolicPDE{D}}

struct LinearAdvection{D, M} <: HyperbolicPDE{D, M}
    vel::Flux{D, M}
end

struct BurgersEquation{a} <: ScalarHyperbolicPDE{1} end
struct TestU3Equation{a} <: ScalarHyperbolicPDE{1} end
struct Euler1D <: HyperbolicPDESystem{1, 3} end
# --- Burgers Equation 2D ---
struct BurgersEquation2D <: ScalarHyperbolicPDE{2} end

struct SimSetting
    tmax::Float64
    dt::Float64
    interpRange::Float64
    interpAlpha::Float64
    saveFreq::Int64
end

