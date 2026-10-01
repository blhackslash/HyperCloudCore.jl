export MLSWeightFunction, AbstractBoundaryCondition, NumericalFluxFunction, MOODStrategy, MOODCriterion
export RealMOOD, AbstractSlopeLimiter, RealSlopeLimiter, DivergenceInterpolator, AbstractPath, EquationRepresentation, NCRepresentation, Conservative
export HyperbolicPDE, UpwindAlgorithm, NoSourceTerm, AbstractSourceTerm, NoGridMover, GridMover
export RKButcherTableau, IMEXButcherTableau, GeneralIMEXTimeStepper, GeneralRKTimeStepper, TimeStepper

"""
    MLSWeightFunction
    GridMover
    AbstractBoundaryCondition
    NumericalFluxFunction
    MOODStrategy
    MOODCriterion
    AbstractSlopeLimiter
    DivergenceInterpolator
    UpwindAlgorithm
    HyperbolicPDE
    TimeStepper
    AbstractSourceTerm

Core abstract types defining the extensible architecture of the mesh-free solver.
"""
abstract type MLSWeightFunction end
abstract type GridMover end
struct NoGridMover <: GridMover end
abstract type AbstractBoundaryCondition end

abstract type NumericalFluxFunction end

abstract type MOODStrategy end
abstract type MOODCriterion end
abstract type RealMOOD <: MOODCriterion end

"""
    MOOD{S <: MOODStrategy, C <: MOODCriterion}

A concrete structure pairing a `MOODStrategy` (which dictates how order reduction propagates) with a `MOODCriterion` (which dictates when order reduction is triggered).
"""
struct MOOD{S <: MOODStrategy, C <: MOODCriterion}
    strategy::S
    criterion::C
end

abstract type AbstractSlopeLimiter end
abstract type RealSlopeLimiter <: AbstractSlopeLimiter end

abstract type DivergenceInterpolator end
abstract type UpwindAlgorithm end 

abstract type AbstractPath end
abstract type PathIntegrator end
struct PathIntegral{P <: AbstractPath, I <: PathIntegrator}
    path::P
    integrator::I
end
"""
    Conservative
    NCRepresentation{P <: AbstractPath}

Representations of the governing equations. 
- `Conservative` indicates a standard divergence form. 
- `NCRepresentation` designates systems containing non-conservative products evaluated along a specific path.
"""
abstract type EquationRepresentation end
struct Conservative <: EquationRepresentation end
abstract type NCRepresentation{P <: AbstractPath} <: EquationRepresentation end

abstract type HyperbolicPDE{D, M, T, R <: EquationRepresentation} end

@inline function evaluate_nc_jump(
    eq::HyperbolicPDE{D, M, T, Conservative}, f_L::Flux{D, M, T}, f_R::Flux{D, M, T}, dist_k::Space{D, T}
) where {D, M, T}
    return zero(Flux{D, M, T})
end

export AbstractSourceTerm, AbstractExplicitSourceTerm, AbstractImplicitSourceTerm
export NoExplicitSource, NoImplicitSource

abstract type AbstractSourceTerm end
abstract type AbstractExplicitSourceTerm <: AbstractSourceTerm end
abstract type AbstractImplicitSourceTerm <: AbstractSourceTerm end

struct NoExplicitSource <: AbstractExplicitSourceTerm end
struct NoImplicitSource <: AbstractImplicitSourceTerm end

abstract type TimeStepper end

## Time Steppers
"""
    InteractionBuffer{D, M, T}

A thread-safe, pre-allocated workspace designed to hold neighbor interaction data during flux evaluations.

# Fields
- `f::Vector{State{M, T}}`: Stores the direct state of neighboring particles.
- `df::Vector{State{M, T}}`: Stores the raw state differences between neighbors and the target particle.
- `df_flux::Vector{Flux{D, M, T}}`: Stores computed numerical flux differences or non-conservative jumps.
- `df_scratch::Vector{State{M, T}}`: An auxiliary buffer for intermediate moving least squares operations.
- `mask::Vector{Bool}`: A boolean array used to filter specific neighbors dynamically during directional stencil building.
"""
struct InteractionBuffer{D, M, T}
    f::Vector{State{M, T}}
    df::Vector{State{M, T}}
    df_flux::Vector{Flux{D, M, T}}
    df_scratch::Vector{State{M, T}} 
    mask::Vector{Bool} 
    
    InteractionBuffer{D, M, T}() where {D, M, T} = new{D, M, T}(
        State{M, T}[], State{M, T}[], Flux{D, M, T}[], State{M, T}[], Bool[]
    )
end

"""
    RKButcherTableau{T}

A structure storing the coefficients for explicit Runge-Kutta time integration.
- Contains the explicit step weights `a`, the final combination weights `b`, and the fractional time steps `c`.
"""
struct RKButcherTableau{T}
    a::Matrix{T}
    b::Vector{T}
    c::Vector{T}
end

"""
    IMEXButcherTableau{T}

A structure storing the paired coefficients for Implicit-Explicit (IMEX) Runge-Kutta time integration.
"""
struct IMEXButcherTableau{T} 
    a::Matrix{T}  
    a_t::Matrix{T} 
    c::Vector{T}   
    c_t::Vector{T} 
    b::Vector{T}   
    b_t::Vector{T}
    
    function IMEXButcherTableau(a::Matrix{T}, a_t::Matrix{T}, c::Vector{T}, c_t::Vector{T}, b::Vector{T}, b_t::Vector{T}) where {T}
        s = size(a, 1) 
        @assert (size(a, 2) == s && size(a_t, 1) == s && size(a_t, 2) == s &&
                 length(c) == s && length(c_t) == s && length(b) == s && length(b_t) == s) "All Butcher tableau components must match number of stages"    
        
        for i in 1:s, j in (i+1):s
            @assert a[i,j] == zero(T) "Implicit matrix A must be lower triangular."
        end
        for i in 1:s, j in i:s 
            @assert a_t[i,j] == zero(T) "Explicit matrix a_t must be strictly lower triangular."
        end
        new{T}(a, a_t, c, c_t, b, b_t)
    end
end

"""
    GeneralRKTimeStepper

A standard explicit Runge-Kutta time integration orchestrator.

# Details
- Couples the physical PDE with the spatial divergence interpolator and explicit Butcher tableau.
- Manages an explicitly-typed tuple of `AbstractExplicitSourceTerm`s.
- Pre-allocates a primary `K_stages` buffer matrix for intermediate derivative evaluations.
"""
struct GeneralRKTimeStepper{D, M, T, PDE <: HyperbolicPDE, G <: DivergenceInterpolator, EST <: Tuple{Vararg{AbstractExplicitSourceTerm}}} <: TimeStepper
    pde::PDE
    divergence_interpolator::G
    tableau::RKButcherTableau{T}
    
    explicit_sources::EST
    
    rho_n::Vector{State{M, T}}
    rho_stage::Vector{State{M, T}}
    K_stages::Vector{Vector{State{M, T}}} 
    int_buffer::InteractionBuffer{D, M, T}

    # Primary strictly-typed constructor
    function GeneralRKTimeStepper(
        pde::HyperbolicPDE{D, M, T}, div_interp::G, 
        explicit_sources::EST, tableau::RKButcherTableau{T}
    ) where {D, M, T, G, EST <: Tuple{Vararg{AbstractExplicitSourceTerm}}}
        s = size(tableau.a, 1)
        new{D, M, T, typeof(pde), G, EST}(
            pde, div_interp, tableau, explicit_sources,
            State{M, T}[], State{M, T}[], 
            [State{M, T}[] for _ in 1:s],
            InteractionBuffer{D, M, T}()
        )
    end
end

"""
    GeneralIMEXTimeStepper

A comprehensive IMEX time integration orchestrator.

# Details
- Manages the physical PDE, spatial divergence interpolator, and strongly-typed explicit/implicit source term tuples.
- Automatically resolves the system dimensions (`M` and `D`) natively from the provided physical/kinetic PDE.
- Allocates and maintains explicit (`K_E_stages`) and implicit (`K_I_stages`) evaluation buffers for all intermediate sub-steps.
"""
struct GeneralIMEXTimeStepper{D, M, T, PDE <: HyperbolicPDE, G <: DivergenceInterpolator, EST <: Tuple{Vararg{AbstractExplicitSourceTerm}}, IST <: Tuple{Vararg{AbstractImplicitSourceTerm}}} <: TimeStepper
    pde::PDE
    divergence_interpolator::G
    tableau::IMEXButcherTableau{T}
    
    explicit_sources::EST
    implicit_sources::IST
    
    rho_n::Vector{State{M, T}}
    Y_stages::Vector{Vector{State{M, T}}}
    K_E_stages::Vector{Vector{State{M, T}}}
    K_I_stages::Vector{Vector{State{M, T}}}
    
    int_buffer::InteractionBuffer{D, M, T}
    num_stages::Int

    # Primary strictly-typed constructor
    function GeneralIMEXTimeStepper(
        pde::HyperbolicPDE{D, M, T}, div_interp::G, 
        explicit_sources::EST, implicit_sources::IST, 
        tableau::IMEXButcherTableau{T}
    ) where {D, M, T, G, EST <: Tuple{Vararg{AbstractExplicitSourceTerm}}, IST <: Tuple{Vararg{AbstractImplicitSourceTerm}}}
        
        s = size(tableau.a, 1)
        
        new{D, M, T, typeof(pde), G, EST, IST}(
            pde, div_interp, tableau, explicit_sources, implicit_sources,
            State{M, T}[], 
            [State{M, T}[] for _ in 1:s], 
            [State{M, T}[] for _ in 1:s], 
            [State{M, T}[] for _ in 1:s], 
            InteractionBuffer{D, M, T}(), 
            s
        )
    end
end
