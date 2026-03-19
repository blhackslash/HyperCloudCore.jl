abstract type TimeStepper end
abstract type MeshfreeTimeStepper <: TimeStepper end
abstract type FixedGridTimeStepper <: TimeStepper end
abstract type MeshfreeSystemTimeStepper <: MeshfreeTimeStepper end

## ------------------------------- Meshfree Direct Steppers -------------------------------

struct EulerUpwind{M, PDE <: HyperbolicPDE, G1 <: GradientInterpolator, G2 <: GradientInterpolator, MOOD <: MOODCriterion} <: MeshfreeTimeStepper
    pde::PDE
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    
    rhoInit::Vector{State{M}}      
    neighbor_fs::Vector{State{M}}  
    neighbor_dfs::Vector{State{M}} 

    function EulerUpwind(pde::HyperbolicPDE{D, M}, grad::G1, fallback::G2, mood::MOOD) where {G1, G2, MOOD, D, M}
        new{M, typeof(pde), G1, G2, M}(pde, grad, fallback, mood, State{M}[], State{M}[], State{M}[])
    end
end

struct RalstonRK2{M, PDE, G1, G2, MOOD} <: MeshfreeTimeStepper
    pde::PDE
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    
    rhoInit::Vector{State{M}}
    rhos::Vector{State{M}}
    div1::Vector{State{M}}

    neighbor_fs::Vector{State{M}}
    neighbor_dfs::Vector{State{M}}

    function RalstonRK2(pde::HyperbolicPDE{D, M}, grad::G1, fallback::G2, mood::MOOD) where {G1, G2, MOOD, D, M}
        new{M, typeof(pde), G1, G2, MOOD}(pde, grad, fallback, mood, State{M}[], State{M}[], State{M}[], State{M}[], State{M}[])
    end
end

struct RK3{M, PDE <: HyperbolicPDE, G1, G2, MOOD} <: MeshfreeTimeStepper
    pde::PDE
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    
    rhoInit::Vector{State{M}}
    rhos::Vector{State{M}}
    div1::Vector{State{M}}
    div2::Vector{State{M}}

    neighbor_fs::Vector{State{M}}
    neighbor_dfs::Vector{State{M}}

    function RK3(pde::HyperbolicPDE{D, M}, grad::G1, fallback::G2, mood::MOOD) where {G1, G2, MOOD, D, M}
        new{M, typeof(pde), G1, G2, MOOD}(pde, grad, fallback, mood, State{M}[], State{M}[], State{M}[], State{M}[], State{M}[], State{M}[])
    end
end

struct RK4{M, PDE <: HyperbolicPDE, G1, G2, MOOD} <: MeshfreeTimeStepper
    pde::PDE
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    
    rhoInit::Vector{State{M}}
    rhos::Vector{State{M}}
    k1::Vector{State{M}}
    k2::Vector{State{M}}
    k3::Vector{State{M}}

    neighbor_fs::Vector{State{M}}
    neighbor_dfs::Vector{State{M}}

    function RK4(pde::HyperbolicPDE{D, M}, grad::G1, fallback::G2, mood::MOOD) where {G1, G2, MOOD, D, M}
        new{M, typeof(pde), G1, G2, M}(pde, grad, fallback, mood, State{M}[], State{M}[], State{M}[], State{M}[], State{M}[], State{M}[], State{M}[])
    end
end

struct RalstonSwitchRK2{M, PDE <: HyperbolicPDE, G1, G2, MOOD} <: MeshfreeTimeStepper
    pde::PDE
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    
    rhoInit::Vector{State{M}}
    rhos::Vector{State{M}}
    rho_fallback::Vector{State{M}}
    div1::Vector{State{M}}

    # Graph/Topology Propagation Buffers for MOOD Switching
    prop_indices::Vector{Int}
    switched_to_fallback::Vector{Bool}
    mood_indices::Vector{Int}
    tol::Float64

    neighbor_fs::Vector{State{M}}
    neighbor_dfs::Vector{State{M}}

    function RalstonSwitchRK2(pde::HyperbolicPDE{D, M}, grad::G1, fallback::G2, mood::MOOD, tol::Float64=1e-6) where {G1, G2, MOOD, D, M}
        new{M, typeof(pde), G1, G2, M}(pde, grad, fallback, mood, State{M}[], State{M}[], State{M}[], State{M}[], Int[], Bool[], Int[], tol, State{M}[], State{M}[])
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
struct KineticSourceTerm <: AbstractSourceTerm end

struct Kin2Macro{M}
    ranges::NTuple{M, UnitRange{Int}}
end

struct RelaxationSourceTerm{D, M, K} <: KineticSourceTerm
    kin2macro::Kin2Macro{M}
    epsilon::Float64
    inv_epsilon::Float64
    coefficients::State{M} # Sized to Macro variables
    inv_relax_speeds::SVector{K, Space{D}} # Sized to Kinetic, customized dot product!
    interior_factors::State{K}
end

mutable struct NonLocalRelaxationSourceTerm{D, M, K} <: KineticSourceTerm
    kin2macro::Kin2Macro{M}
    epsilon::Float64
    inv_epsilon::Float64
    coefficients::State{M}
    inv_relax_speeds::SVector{K, Space{D}}
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

struct GeneralIMEXTimeStepper{M, G1, G2, MOOD, IS, ST_OBJ, BT} <: MeshfreeSystemTimeStepper
    gradientInterpolators::NTuple{M, G1}
    fallbackInterpolators::NTuple{M, G2}
    mood::MOOD
    implicit_solver::IS
    source_term_object::ST_OBJ
    butcher_tableau::BT
    
    # --- Buffers for local time stepping (Fully Converted to Vector{State{M}}) ---
    U_n::Vector{State{M}}
    Y_stages::Vector{Vector{State{M}}}
    K_E_stages::Vector{Vector{State{M}}}
    K_I_stages::Vector{Vector{State{M}}}
    
    mood_triggered::Matrix{Bool}
    
    # --- Buffers for parallel evaluation ---
    U_n_sys::Vector{State{M}}
    Y_stages_sys::Vector{Vector{State{M}}}
    K_E_stages_sys::Vector{Vector{State{M}}}
    K_I_stages_sys::Vector{Vector{State{M}}}
    
    mood_triggered_sys::Array{Bool, 3}
    
    # Buffers for explicit fused loop 
    all_neighbor_fs::Vector{State{M}}
    all_neighbor_dfs::Vector{State{M}}
    
    num_stages::Int

    function GeneralIMEXTimeStepper( eq::HyperbolicPDE{D, M},
        gradientInterpolator::G1, fallbackInterpolator::G2, mood::MOOD,
        implicit_solver::IS, source_term_object::ST_OBJ, butcher_tableau::BT,
    ) where {G1, G2, MOOD, IS, ST_OBJ, BT, D, M}
        
        s = size(butcher_tableau.A, 1) # Number of stages
        
        new{M, G1, G2, MOOD, IS, ST_OBJ, BT}(
            ntuple(_ -> deepcopy(gradientInterpolator), M_comp), 
            ntuple(_ -> deepcopy(fallbackInterpolator), M_comp), 
            mood, implicit_solver, source_term_object, butcher_tableau,
            State{M}[], 
            [State{M}[] for _ in 1:s],
            [State{M}[] for _ in 1:s], 
            [State{M}[] for _ in 1:s], 
            falses(0, M_comp),
            State{M}[], 
            [State{M}[] for _ in 1:s],
            [State{M}[] for _ in 1:s], 
            [State{M}[] for _ in 1:s], 
            falses(0, M_comp, s),
            State{M}[], 
            State{M}[],
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

mutable struct ClassicalRichtmyerLWMOOD{MOOD <: MOODCriterion} <: FixedGridTimeStepper
    mood::MOOD
    # --- Reusable Buffers (Workspace) ---
    rho_n::Vector{Float64}
    rho_candidate::Vector{Float64}
    rho_predict_interface::Vector{Float64}
    flux_predict::Vector{Float64}
end