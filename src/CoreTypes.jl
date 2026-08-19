export MLSWeightFunction, GridMover, AbstractBoundaryCondition, AbstractDomain, NumericalFluxFunction, MOODStrategy, MOODCriterion
export RealMOOD, AbstractSlopeLimiter, RealSlopeLimiter, DivergenceInterpolator, AbstractPath, EquationRepresentation, NCRepresentation
export HyperbolicPDE, UpwindAlgorithm, AbstractImplicitSolver
export RKButcherTableau, IMEXButcherTableau, GeneralIMEXTimeStepper, GeneralRKTimeStepper

abstract type MLSWeightFunction end
abstract type GridMover end
abstract type AbstractBoundaryCondition end
abstract type AbstractDomain{D, T} end

abstract type NumericalFluxFunction end

abstract type MOODStrategy end
abstract type MOODCriterion end
abstract type RealMOOD <: MOODCriterion end
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
abstract type EquationRepresentation end
struct Conservative <: EquationRepresentation end
abstract type NCRepresentation{P <: AbstractPath} <: EquationRepresentation end

abstract type HyperbolicPDE{D, M, T, R <: EquationRepresentation} end

abstract type TimeStepper end
abstract type AbstractSourceTerm end
abstract type AbstractImplicitSolver end

## Time Steppers

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

struct RKButcherTableau{T}
    a::Matrix{T}
    b::Vector{T}
    c::Vector{T}
end

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


struct GeneralIMEXTimeStepper{D, M, T, PDE <: HyperbolicPDE, G <: DivergenceInterpolator, IS <: AbstractImplicitSolver, ST <: AbstractSourceTerm} <: TimeStepper
    pde::PDE
    divergence_interpolator::G
    implicit_solver::IS
    source_term_object::ST
    tableau::IMEXButcherTableau{T}
    
    rho_n::Vector{State{M, T}}
    Y_stages::Vector{Vector{State{M, T}}}
    K_E_stages::Vector{Vector{State{M, T}}}
    K_I_stages::Vector{Vector{State{M, T}}}
    
    int_buffer::InteractionBuffer{D, M, T}
    num_stages::Int

    function GeneralIMEXTimeStepper(
        pde::PDE, div_interp::G, implicit_solver::IS, 
        source_term_object::ST, tableau::IMEXButcherTableau{T}
    ) where {PDE <: HyperbolicPDE, G, IS, ST, T}
        
        s = size(tableau.a, 1)
        
        # Extract D and M natively from the relaxation system
        M = length(source_term_object.scaled_inv_speeds) 
        D = length(source_term_object.scaled_inv_speeds[1])
        
        new{D, M, T, PDE, G, IS, ST}(
            pde, div_interp, implicit_solver, source_term_object, tableau,
            State{M, T}[], 
            [State{M, T}[] for _ in 1:s], 
            [State{M, T}[] for _ in 1:s], 
            [State{M, T}[] for _ in 1:s], 
            InteractionBuffer{D, M, T}(), 
            s
        )
    end
end

struct GeneralRKTimeStepper{D, M, T, PDE <: HyperbolicPDE, G <: DivergenceInterpolator} <: TimeStepper
    pde::PDE
    divergence_interpolator::G
    tableau::RKButcherTableau{T}
    
    rho_n::Vector{State{M, T}}
    rho_stage::Vector{State{M, T}}
    K_stages::Vector{Vector{State{M, T}}} 
    int_buffer::InteractionBuffer{D, M, T}

    function GeneralRKTimeStepper(pde::HyperbolicPDE{D, M, T}, div_interp::G, tableau::RKButcherTableau{T}) where {G, D, M, T}
        s = size(tableau.a, 1)
        new{D, M, T, typeof(pde), G}(
            pde, div_interp, tableau, 
            State{M, T}[], State{M, T}[], 
            [State{M, T}[] for _ in 1:s],
            InteractionBuffer{D, M, T}()
        )
    end
end