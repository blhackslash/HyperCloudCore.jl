abstract type TimeStepper end
abstract type MeshfreeTimeStepper <: TimeStepper end
abstract type FixedGridTimeStepper <: TimeStepper end
abstract type MeshfreeSystemTimeStepper <: MeshfreeTimeStepper end

## ------------------------------- Meshfree Direct Steppers -------------------------------

struct EulerUpwind{T, PDE <: HyperbolicPDE, G1 <: GradientInterpolator, G2 <: GradientInterpolator, M <: MOODCriterion} <: MeshfreeTimeStepper
    pde::PDE
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::M
    
    rhoInit::Vector{T}      
    neighbor_fs::Vector{T}  
    neighbor_dfs::Vector{T} 

    function EulerUpwind(pde::PDE, grad::G1, fallback::G2, mood::M, ::Type{T}=SVector{1, Float64}) where {PDE, G1, G2, M, T}
        new{T, PDE, G1, G2, M}(pde, grad, fallback, mood, T[], T[], T[])
    end
end

struct RalstonRK2{T, PDE <: HyperbolicPDE, G1, G2, MOOD} <: MeshfreeTimeStepper
    pde::PDE
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    
    rhoInit::Vector{T}
    rhos::Vector{T}
    div1::Vector{T}

    neighbor_fs::Vector{T}
    neighbor_dfs::Vector{T}

    function RalstonRK2(pde::PDE, grad::G1, fallback::G2, mood::M, ::Type{T}=SVector{1, Float64}) where {PDE, G1, G2, M, T}
        new{T, PDE, G1, G2, M}(pde, grad, fallback, mood, T[], T[], T[], T[], T[])
    end
end

struct RK3{T, PDE <: HyperbolicPDE, G1, G2, MOOD} <: MeshfreeTimeStepper
    pde::PDE
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    
    rhoInit::Vector{T}
    rhos::Vector{T}
    div1::Vector{T}
    div2::Vector{T}

    neighbor_fs::Vector{T}
    neighbor_dfs::Vector{T}

    function RK3(pde::PDE, grad::G1, fallback::G2, mood::M, ::Type{T}=SVector{1, Float64}) where {PDE, G1, G2, M, T}
        new{T, PDE, G1, G2, M}(pde, grad, fallback, mood, T[], T[], T[], T[], T[], T[])
    end
end

struct RK4{T, PDE <: HyperbolicPDE, G1, G2, MOOD} <: MeshfreeTimeStepper
    pde::PDE
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    
    rhoInit::Vector{T}
    rhos::Vector{T}
    k1::Vector{T}
    k2::Vector{T}
    k3::Vector{T}

    neighbor_fs::Vector{T}
    neighbor_dfs::Vector{T}

    function RK4(pde::PDE, grad::G1, fallback::G2, mood::M, ::Type{T}=SVector{1, Float64}) where {PDE, G1, G2, M, T}
        new{T, PDE, G1, G2, M}(pde, grad, fallback, mood, T[], T[], T[], T[], T[], T[], T[])
    end
end

struct RalstonSwitchRK2{T, PDE <: HyperbolicPDE, G1, G2, MOOD} <: MeshfreeTimeStepper
    pde::PDE
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    
    rhoInit::Vector{T}
    rhos::Vector{T}
    rho_fallback::Vector{T}
    div1::Vector{T}

    # Graph/Topology Propagation Buffers for MOOD Switching
    prop_indices::Vector{Int}
    switched_to_fallback::Vector{Bool}
    mood_indices::Vector{Int}
    tol::Float64

    neighbor_fs::Vector{T}
    neighbor_dfs::Vector{T}

    function RalstonSwitchRK2(pde::PDE, grad::G1, fallback::G2, mood::M, tol::Float64=1e-6, ::Type{T}=SVector{1, Float64}) where {PDE, G1, G2, M, T}
        new{T, PDE, G1, G2, M}(pde, grad, fallback, mood, T[], T[], T[], T[], Int[], Bool[], Int[], tol, T[], T[])
    end
end

## ------------------------------- Butcher Tableaus -------------------------------

struct ButcherTableau
    A::Matrix{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
    A_tilde::Matrix{Float64}
    b_tilde::Vector{Float64}
    c_tilde::Vector{Float64}
end

## ------------------------------- Source Terms -------------------------------

abstract type AbstractSourceTerm end

struct NoSourceTerm <: AbstractSourceTerm end

struct RelaxationSourceTerm{D, NM, NK} <: AbstractSourceTerm
    epsilon::Float64
    inv_epsilon::Float64
    coefficients::NTuple{NM, Float64} # Sized to Macro variables
    inv_relax_speeds::NTuple{NK, SVector{D, Float64}} # Sized to Kinetic, customized dot product!
    interior_factors::NTuple{NK, Float64}
end

mutable struct NonLocalRelaxationSourceTerm{D, NM, NK} <: AbstractSourceTerm
    epsilon::Float64
    inv_epsilon::Float64
    coefficients::NTuple{NM, Float64}
    inv_relax_speeds::NTuple{NK, SVector{D, Float64}}
    interior_factor::Float64
    T_potential::Matrix{Float64}
end

abstract type AbstractImplicitSolver end
struct PicardIterationSolver <: AbstractImplicitSolver
    max_iters::Int
    tol::Float64
end
struct LinearizedRelaxationImplicitSolver <: AbstractImplicitSolver end

struct IMEXButcherTableau{M <: AbstractArray{Float64, 2}, V <: AbstractArray{Float64, 1}}
    A::M  # Implicit coefficient matrix
    At::M # Explicit coefficient matrix (Atilde)
    c::V  # Implicit time nodes
    ct::V # Explicit time nodes (ctilde)
    b::V  # Final weights (assumed same for explicit and implicit parts by your old code's use)
    bt::V
end

struct GeneralIMEXTimeStepper{T, M_comp, G1, G2, M_crit, IS, ST_OBJ, BT} <: MeshfreeSystemTimeStepper
    gradientInterpolators::NTuple{M_comp, G1}
    fallbackInterpolators::NTuple{M_comp, G2}
    mood::M_crit
    implicit_solver::IS
    source_term_object::ST_OBJ
    butcher_tableau::BT
    
    # --- Buffers for local time stepping (Fully Converted to Vector{T}) ---
    U_n::Vector{T}
    Y_stages::Vector{Vector{T}}
    K_E_stages::Vector{Vector{T}}
    K_I_stages::Vector{Vector{T}}
    
    mood_triggered::Matrix{Bool}
    
    # --- Buffers for parallel evaluation ---
    U_n_sys::Vector{T}
    Y_stages_sys::Vector{Vector{T}}
    K_E_stages_sys::Vector{Vector{T}}
    K_I_stages_sys::Vector{Vector{T}}
    
    mood_triggered_sys::Array{Bool, 3}
    
    # Buffers for explicit fused loop 
    all_neighbor_fs::Vector{T}
    all_neighbor_dfs::Vector{T}
    
    num_stages::Int

    function GeneralIMEXTimeStepper(
        gradientInterpolator::G1, fallbackInterpolator::G2, mood::M_crit,
        implicit_solver::IS, source_term_object::ST_OBJ, butcher_tableau::BT,
        ::Type{T}=SVector{1, Float64} # Parameterized by SVector natively
    ) where {G1, G2, M_crit, IS, ST_OBJ, BT, T}
        
        s = size(butcher_tableau.A, 1) # Number of stages
        M_comp = source_term_object.num_total_kinetic_components
        
        new{T, M_comp, G1, G2, M_crit, IS, ST_OBJ, BT}(
            ntuple(_ -> deepcopy(gradientInterpolator), M_comp), 
            ntuple(_ -> deepcopy(fallbackInterpolator), M_comp), 
            mood, implicit_solver, source_term_object, butcher_tableau,
            T[], 
            [T[] for _ in 1:s],
            [T[] for _ in 1:s], 
            [T[] for _ in 1:s], 
            falses(0, M_comp),
            T[], 
            [T[] for _ in 1:s],
            [T[] for _ in 1:s], 
            [T[] for _ in 1:s], 
            falses(0, M_comp, s),
            T[], 
            T[],
            s
        )
    end
end

## ------------------------------- Fixed Grid Direct Stepper -------------------------------

# --- Upwind Method ---
mutable struct Upwind <: FixedGridTimeStepper 
    rho_n::Vector{Float64} # Reusable buffer for the state at time n
    Upwind() = new(Float64[])
end

# --- Lax-Friedrichs Method ---
mutable struct LaxFriedrich <: FixedGridTimeStepper 
    rho_n::Vector{Float64}
    LaxFriedrich() = new(Float64[])
end

# --- Classical Finite Volume Method ---
mutable struct ClassicalTimeStepper <: FixedGridTimeStepper
    numericalFlux::NumericalFluxFunction
    rho_n::Vector{Float64}
    flux_interfaces::Vector{Float64}
end

struct ClassicalRK2LWTimeStepper <: FixedGridTimeStepper
    rhoOld::Vector{Float64}
    rhoPredict_interface::Vector{Float64} # U_{i+1/2}^{n+1/2} - N values for N interfaces
    # No need for fluxPredict as a field, can be local
end

mutable struct ClassicalRichtmyerLWMOOD{M <: MOODCriterion} <: FixedGridTimeStepper
    mood::M
    # --- Reusable Buffers (Workspace) ---
    rho_n::Vector{Float64}
    rho_candidate::Vector{Float64}
    rho_predict_interface::Vector{Float64}
    flux_predict::Vector{Float64}
end