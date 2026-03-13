abstract type TimeStepper end
abstract type MeshfreeTimeStepper <: TimeStepper end
abstract type FixedGridTimeStepper <: TimeStepper end
abstract type MeshfreeSystemTimeStepper <: MeshfreeTimeStepper end

## ------------------------------- Meshfree Direct Stepper -------------------------------

struct EulerUpwind{G1 <: GradientInterpolator, G2 <: GradientInterpolator, M <: MOODCriterion, GM <: GridMover} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::M
    grid_mover::GM
    
    # Buffers are now part of the struct to be reused
    rhoInit::Vector{Float64}      # Stores the state at the beginning of the step
    neighbor_fs::Vector{Float64}  # Pre-gathered neighbor values
    neighbor_dfs::Vector{Float64} # Pre-gathered neighbor differences
end

# No longer needs Nx, Ny. Buffers are sized based on the grid passed during the call.
struct RalstonRK2{G1, G2, MOOD, GM} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    grid_mover::GM
    
    # Buffers are now part of the struct to be reused
    rhoInit::Vector{Float64}
    rhos::Vector{Float64}
    div1::Vector{Float64}

    # Buffers for efficient calculations
    neighbor_fs::Vector{Float64}
    neighbor_dfs::Vector{Float64}

    function RalstonRK2(grad::G1, fallback::G2, mood::M, gm::GM) where {G1 <: GradientInterpolator, G2 <: GradientInterpolator, M <: MOODCriterion, GM <: GridMover}
        # Initialize with empty buffers, they will be resized on the first step
        new{G1, G2, M, GM}(grad, fallback, mood, gm, Float64[], Float64[], Float64[], Float64[], Float64[])
    end
end

struct RalstonRK2SmoothSwitch{G1, G2, MOOD, GM} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    grid_mover::GM
    tol::Float64
    
    # --- Reusable Buffers (Workspace) ---
    rho_n::Vector{Float64}
    rho_stage::Vector{Float64}
    rho_fallback::Vector{Float64}
    div1::Vector{Float64}
    
    # --- Propagation Buffers ---
    mood_indices::Vector{Int}
    prop_indices::Vector{Int}
    
    # Per-step flag to track which particles have been switched to fallback
    switched_to_fallback::BitVector

    # --- Buffers for efficient calculations (like in RK4) ---
    neighbor_fs::Vector{Float64}
    neighbor_dfs::Vector{Float64}
end

struct RK3{G1 <: GradientInterpolator, G2 <: GradientInterpolator, MOOD <: MOODCriterion, GM} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    grid_mover::GM
    
    # --- Reusable Buffers (Workspace) ---
    rho_n::Vector{Float64}      # Stores the solution at the start of the step
    rho_stage1::Vector{Float64} # Stores the result of the first stage
    rho_stage2::Vector{Float64} # Stores the result of the second stage
    
    div1::Vector{Float64} # Stores divergence from stage 1
    div2::Vector{Float64} # Stores divergence from stage 2
    div3::Vector{Float64} # Stores divergence from stage 3

    # --- Buffers for efficient calculations (like in RK4) ---
    neighbor_fs::Vector{Float64}
    neighbor_dfs::Vector{Float64}
end

struct RK4{G1, G2, MOOD, GM} <: MeshfreeTimeStepper
    gradientInterpolator::G1
    fallbackInterpolator::G2
    mood::MOOD
    grid_mover::GM
    
    # --- Reusable Buffers (Workspace) ---
    rho_n::Vector{Float64}
    rho_stage::Vector{Float64} # A single buffer for all intermediate stages
    
    k1::Vector{Float64} # Stores divergence from stage 1
    k2::Vector{Float64} # Stores divergence from stage 2
    k3::Vector{Float64} # Stores divergence from stage 3
    k4::Vector{Float64} # Stores divergence from stage 4

    # --- Buffers for efficient calculations (like in RK2) ---
    neighbor_fs::Vector{Float64}
    neighbor_dfs::Vector{Float64}
end

## ------------------------------- Meshfree IMEX Stepper -------------------------------

## ------------------------------- Source Terms -------------------------------

abstract type AbstractSourceTerm end

struct NoSourceTerm <: AbstractSourceTerm end

struct Kin2Macro{NM}
    ranges::NTuple{NM, UnitRange{Int}}
end

struct RelaxationSourceTerm{D, N, NK, PDE <: HyperbolicPDE{D, N}} <: AbstractSourceTerm
    system_eq::PDE
    epsilon::Float64
    inv_epsilon::Float64
    kin2macro::Kin2Macro{N}

    # Parameters stored as flat tuples of length NK (Number of Kinetic components)
    coefficients::NTuple{NK, Float64}
    relax_speeds::NTuple{NK, Float64}
    interior_factors::NTuple{NK, Float64}
    dimensions::NTuple{NK, Int} # which spatial dimension (flux) this component advects in

    num_total_kinetic_components::Int64
    num_macro_variables::Int64
end

mutable struct NonLocalRelaxationSourceTerm{D, N, NK, PDE <: HyperbolicPDE{D, N}} <: AbstractSourceTerm
    system_eq::PDE
    epsilon::Float64
    inv_epsilon::Float64
    kin2macro::Kin2Macro{N}

    coefficients::NTuple{N, Float64}
    relax_speeds::NTuple{NK, Float64}
    interior_factor::Float64

    T_potential::Matrix{Float64}
    num_total_kinetic_components::Int
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

mutable struct GeneralIMEXTimeStepper{M_comp, G1, G2, M_crit, IS, ST_OBJ, BT, GM} <: TimeStepper
    # User's modular components
    gradientInterpolator::NTuple{M_comp, G1}
    fallbackInterpolator::NTuple{M_comp, G2}
    mood::M_crit
    implicit_solver::IS
    source_term_object::ST_OBJ
    butcher_tableau::BT
    grid_mover::GM  
    
    # --- Reusable Buffers (Workspace) ---
    U_n_sys::Matrix{Float64}
    Y_stages_sys::Vector{Matrix{Float64}}
    K_E_stages_sys::Vector{Matrix{Float64}}
    K_I_stages_sys::Vector{Matrix{Float64}}
    
    mood_triggered::BitArray{3}
    
    # Buffers for explicit fused loop 
    all_neighbor_fs::Matrix{Float64}
    all_neighbor_dfs::Matrix{Float64}
    
    num_stages::Int
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