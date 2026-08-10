abstract type TimeStepper end
abstract type MeshfreeTimeStepper <: TimeStepper end
abstract type FixedGridTimeStepper <: TimeStepper end
abstract type MeshfreeSystemTimeStepper <: MeshfreeTimeStepper end

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

## ------------------------------- Butcher Tableaus -------------------------------

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

## ------------------------------- Meshfree Direct Steppers -------------------------------

struct GeneralRKTimeStepper{D, M, T, PDE <: HyperbolicPDE, G <: DivergenceInterpolator} <: MeshfreeTimeStepper
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

## ------------------------------- Source Terms -------------------------------
abstract type AbstractSourceTerm end
struct NoSourceTerm <: AbstractSourceTerm end
abstract type KineticSourceTerm <: AbstractSourceTerm end

struct Kin2Macro{NM, NK}
    ranges::NTuple{NM, UnitRange{Int}}
    k_to_m::NTuple{NK, Int}
end

# Added <: KineticSourceTerm
struct RelaxationSourceTerm{D, NM, NK, T} <: KineticSourceTerm
    km::Kin2Macro{NM, NK}
    inv_epsilon::T
    coefficients::State{NM, T}
    scaled_inv_speeds::SVector{NK, Space{D, T}}
end

# Added <: KineticSourceTerm
struct NonLocalRelaxationSourceTerm{D, NM, NK, T} <: KineticSourceTerm
    km::Kin2Macro{NM, NK}
    inv_epsilon::T
    coefficients::State{NM, T}
    scaled_inv_speeds::SVector{NK, Space{D, T}}
    t_potential::Matrix{T}
end

abstract type AbstractImplicitSolver end

# Added <: AbstractImplicitSolver just in case you use it later!
struct PicardIterationSolver{T} <: AbstractImplicitSolver
    max_iters::Int
    tol::T
    s_buffers::Vector{Vector{T}}
    y_buffers::Vector{Vector{T}}
end

struct LinearizedRelaxationImplicitSolver <: AbstractImplicitSolver end

struct GeneralIMEXTimeStepper{D, M, T, PDE <: HyperbolicPDE, G <: DivergenceInterpolator, IS <: AbstractImplicitSolver, ST <: AbstractSourceTerm} <: MeshfreeSystemTimeStepper
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
