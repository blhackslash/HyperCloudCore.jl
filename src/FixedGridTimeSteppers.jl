export Upwind, LaxFriedrich, ClassicalTimeStepper, ClassicalRK2LWTimeStepper, ClassicalRichtmyerLWMOOD
using ..Meshfree4ScalarEq.FluxFunctions

# --- In your TimeIntegration.jl file ---

#==============================================================================
  Fixed-Grid Time Steppers (Optimized for SoA Grids)
==============================================================================#

# --- Upwind Method ---
mutable struct Upwind <: FixedGridTimeStepper 
    rho_n::Vector{Float64} # Reusable buffer for the state at time n
    Upwind() = new(Float64[])
end

function (upwind::Upwind)(eq::LinearAdvection{1}, pg::ParticleGrid1D, settings::SimSetting, time::Real, dt::Real)
    @assert pg.regular "Upwind fixed-grid method requires a regular grid."
    N = pg.meta.N
    if length(upwind.rho_n) != N; resize!(upwind.rho_n, N); end
    
    upwind.rho_n .= pg.rhos # Store u^n
    vel = velocity(eq, 0.0) # Velocity is constant for this equation
    λ = vel * dt / pg.meta.dx

    # This method is only defined for periodic BCs
    if vel > 0
        for i in 2:N
            pg.rhos[i] = upwind.rho_n[i] - λ * (upwind.rho_n[i] - upwind.rho_n[i-1])
        end
        pg.rhos[1] = upwind.rho_n[1] - λ * (upwind.rho_n[1] - upwind.rho_n[N]) # Periodic wrap
    else
        for i in 1:(N-1)
            pg.rhos[i] = upwind.rho_n[i] - λ * (upwind.rho_n[i+1] - upwind.rho_n[i])
        end
        pg.rhos[N] = upwind.rho_n[N] - λ * (upwind.rho_n[1] - upwind.rho_n[N]) # Periodic wrap
    end
end

# --- Lax-Friedrichs Method ---
mutable struct LaxFriedrich <: FixedGridTimeStepper 
    rho_n::Vector{Float64}
    LaxFriedrich() = new(Float64[])
end

function (lf::LaxFriedrich)(eq::ScalarHyperbolicPDE{1}, pg::ParticleGrid1D, settings::SimSetting, time::Real, dt::Real)
    @assert pg.regular "Lax-Friedrich fixed-grid method requires a regular grid."
    N = pg.meta.N
    if length(lf.rho_n) != N; resize!(lf.rho_n, N); end
    
    lf.rho_n .= pg.rhos
    λ = dt / (2 * pg.meta.dx)

    # This method is only defined for periodic BCs
    for i in 2:(N-1)
        pg.rhos[i] = 0.5 * (lf.rho_n[i+1] + lf.rho_n[i-1]) - λ * (flux(eq, lf.rho_n[i+1]) - flux(eq, lf.rho_n[i-1]))
    end
    # Periodic boundary updates
    pg.rhos[1] = 0.5 * (lf.rho_n[2] + lf.rho_n[N]) - λ * (flux(eq, lf.rho_n[2]) - flux(eq, lf.rho_n[N]))
    pg.rhos[N] = 0.5 * (lf.rho_n[1] + lf.rho_n[N-1]) - λ * (flux(eq, lf.rho_n[1]) - flux(eq, lf.rho_n[N-1]))
end

# --- Classical Finite Volume Method ---
mutable struct ClassicalTimeStepper <: FixedGridTimeStepper
    numericalFlux::NumericalFluxFunction
    rho_n::Vector{Float64}
    flux_interfaces::Vector{Float64}

    function ClassicalTimeStepper(numFlux::NumericalFluxFunction)
        new(numFlux, Float64[], Float64[])
    end
end

function (cts::ClassicalTimeStepper)(eq::ScalarHyperbolicPDE{1}, pg::ParticleGrid1D, settings::SimSetting, time::Real, dt::Real)
    @assert pg.regular "ClassicalTimeStepper requires a regular grid."
    N = pg.meta.N
    if length(cts.rho_n) != N; resize!(cts.rho_n, N); resize!(cts.flux_interfaces, N); end
    
    cts.rho_n .= pg.rhos
    dtdx = dt / pg.meta.dx

    # This method is only defined for periodic BCs
    # flux_interfaces[k] stores the flux at the right-hand interface of particle k (i.e., F*_{k+1/2})
    for k in 1:N
        u_L = cts.rho_n[k]
        u_R = cts.rho_n[mod1(k + 1, N)] # Periodic neighbor
        cts.flux_interfaces[k] = cts.numericalFlux(u_L, u_R, eq)
    end

    # Update rule: U_i^{n+1} = U_i^n - (dt/dx) * ( F*_{i+1/2} - F*_{i-1/2} )
    for i in 1:N
        F_star_i_plus_half = cts.flux_interfaces[i]
        F_star_i_minus_half = cts.flux_interfaces[mod1(i - 1, N)] # Periodic neighbor
        pg.rhos[i] = cts.rho_n[i] - dtdx * (F_star_i_plus_half - F_star_i_minus_half)
    end
end



# --- NEW: ClassicalRK2LWTimeStepper (Richtmyer two-step Lax-Wendroff) ---
# This one remains the same as it uses the physical flux F(U) on predicted states,
# not a generic NumericalFluxFunction for its core logic.
struct ClassicalRK2LWTimeStepper <: FixedGridTimeStepper
    rhoOld::Vector{Float64}
    rhoPredict_interface::Vector{Float64} # U_{i+1/2}^{n+1/2} - N values for N interfaces
    # No need for fluxPredict as a field, can be local

    function ClassicalRK2LWTimeStepper(Nx::Integer)
        new(Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx))
    end
end

function (crk2::ClassicalRK2LWTimeStepper)(eq::ScalarHyperbolicPDE{1}, pg::ParticleGrid1D, settings::SimSetting, time::Real, dt::Real)
    map!(particle -> particle.rho, crk2.rhoOld, pg.grid)
    N = pg.meta.N
    dx = pg.meta.dx
    
    # --- Predictor Step: Calculate U_{i+1/2}^{n+1/2} ---
    # This is stored in crk2.rhoPredict_interface[i] (representing interface i+1/2)
    for i in 1:N
        idx_plus_1 = mod1(i + 1, N) # Periodic neighbor for U_{i+1}^n
        
        Ui_n = crk2.rhoOld[i]
        Uip1_n = crk2.rhoOld[idx_plus_1]
        
        F_Ui_n = flux(eq, Ui_n)         # Physical flux F(U_i^n)
        F_Uip1_n = flux(eq, Uip1_n)     # Physical flux F(U_{i+1}^n)
        
        # U_{i+1/2}^{n+1/2}
        crk2.rhoPredict_interface[i] = 0.5 * (Ui_n + Uip1_n) - (dt / (2.0 * dx)) * (F_Uip1_n - F_Ui_n)
    end

    # --- Corrector Step ---
    # Temporary storage for fluxes F(U_{i+1/2}^{n+1/2})
    flux_of_predicted_interface_states = Vector{Float64}(undef, N)
    for i in 1:N
        # flux_of_predicted_interface_states[i] is F^*_{i+1/2}
        flux_of_predicted_interface_states[i] = flux(eq, crk2.rhoPredict_interface[i])
    end

    # Update U_i^{n+1}
    for i in 1:N
        idx_minus_1 = mod1(i - 1, N) # For F^*_{i-1/2}
        
        F_star_ip_half = flux_of_predicted_interface_states[i]             # This is F(U_{i+1/2}^{n+1/2})
        F_star_im_half = flux_of_predicted_interface_states[idx_minus_1]   # This is F(U_{(i-1)+1/2}^{n+1/2}) = F(U_{i-1/2}^{n+1/2})
        
        pg.grid[i].rho = crk2.rhoOld[i] - (dt / dx) * (F_star_ip_half - F_star_im_half)
    end
end


# --- Helper function for Lax-Friedrichs dissipation ---
# This can be a local helper function if only used here.
"""
    _max_abs_speed_classical(eq, uL, uR)

Calculates the maximum absolute wavespeed for the state between uL and uR.
Used as the dissipation coefficient for the Lax-Friedrichs flux.
"""
function _max_abs_speed_classical(eq::LinearAdvection, uL::Real, uR::Real)
    return abs(eq.vel[1])
end

function _max_abs_speed_classical(eq::BurgersEquation, uL::Real, uR::Real)
    # For Burger's equation, the characteristic speed is u.
    return max(abs(uL), abs(uR))
end
# Add methods for other equations like Euler if needed.


# --- NEW: Classical Richtmyer Lax-Wendroff with MOOD ---

# Helper function to estimate the max wave speed for the Rusanov/LF flux
function _max_abs_speed_classical(eq::ScalarHyperbolicPDE, u_L, u_R)
    # This is a simple implementation; more advanced versions might use Roe averages
    vel_L = velocity(eq, u_L)
    vel_R = velocity(eq, u_R)
    return max(abs(vel_L), abs(vel_R))
end


mutable struct ClassicalRichtmyerLWMOOD{M <: MOODCriterion} <: FixedGridTimeStepper
    mood::M
    # --- Reusable Buffers (Workspace) ---
    rho_n::Vector{Float64}
    rho_candidate::Vector{Float64}
    rho_predict_interface::Vector{Float64}
    flux_predict::Vector{Float64}

    function ClassicalRichtmyerLWMOOD(; mood::M = NoMOOD()) where {M <: MOODCriterion}
        new{M}(mood, Float64[], Float64[], Float64[], Float64[])
    end
end

function initTimeStepper(cts_mood::ClassicalRichtmyerLWMOOD, pg::ParticleGrid, settings::SimSetting)
    # This function is now primarily for ensuring buffers are sized.
    # The main functor also checks this, so this function is optional but good practice.
    N = pg.meta.N
    if length(cts_mood.rho_n) != N
        resize!.((cts_mood.rho_n, cts_mood.rho_candidate, cts_mood.rho_predict_interface, cts_mood.flux_predict), N)
    end
end

function (cts_mood::ClassicalRichtmyerLWMOOD)(
    eq::ScalarHyperbolicPDE{1}, 
    pg::ParticleGrid1D, 
    settings::SimSetting, 
    time::Real, 
    dt::Real
)
    @assert pg.regular "ClassicalRichtmyerLWMOOD requires a regular grid."
    
    N = pg.meta.N
    # --- Ensure buffers are correctly sized for the current grid ---
    if length(cts_mood.rho_n) != N
        resize!.((cts_mood.rho_n, cts_mood.rho_candidate, cts_mood.rho_predict_interface, cts_mood.flux_predict), N)
    end

    cts_mood.rho_n .= pg.rhos
    dx = pg.meta.dx
    dtdx = dt / dx
    
    # This method is only defined for periodic BCs
    interior = pg.interior_indices # Should be 1:N for periodic

    # --- 1. Predictor Step: Calculate U_{i+1/2}^{n+1/2} at all interfaces ---
    for i in 1:N
        idx_plus_1 = mod1(i + 1, N)
        Ui_n = cts_mood.rho_n[i]
        Uip1_n = cts_mood.rho_n[idx_plus_1]
        
        F_Ui_n = flux(eq, Ui_n)
        F_Uip1_n = flux(eq, Uip1_n)
        
        cts_mood.rho_predict_interface[i] = 0.5 * (Ui_n + Uip1_n) - (dt / (2.0 * dx)) * (F_Uip1_n - F_Ui_n)
    end

    # --- 2. Corrector Step: Calculate high-order candidate solution ---
    map!(rho -> flux(eq, rho), cts_mood.flux_predict, cts_mood.rho_predict_interface)

    for i in interior
        F_star_ip_half = cts_mood.flux_predict[i]
        F_star_im_half = cts_mood.flux_predict[mod1(i - 1, N)]
        
        cts_mood.rho_candidate[i] = cts_mood.rho_n[i] - dtdx * (F_star_ip_half - F_star_im_half)
    end

    # --- 3. MOOD Detection and Final Update ---
    for i in interior
        if cts_mood.mood(pg, i, cts_mood.rho_n, cts_mood.rho_candidate[i])
            # MOOD triggered: Fallback to Lax-Friedrichs/Rusanov update for this cell
            u_L_right = cts_mood.rho_n[i]
            u_R_right = cts_mood.rho_n[mod1(i + 1, N)]
            alpha_right = _max_abs_speed_classical(eq, u_L_right, u_R_right)
            F_star_ip_half_LF = 0.5 * (flux(eq, u_L_right) + flux(eq, u_R_right)) - 0.5 * alpha_right * (u_R_right - u_L_right)

            u_L_left = cts_mood.rho_n[mod1(i - 1, N)]
            u_R_left = cts_mood.rho_n[i]
            alpha_left = _max_abs_speed_classical(eq, u_L_left, u_R_left)
            F_star_im_half_LF = 0.5 * (flux(eq, u_L_left) + flux(eq, u_R_left)) - 0.5 * alpha_left * (u_R_left - u_L_left)
            
            pg.rhos[i] = cts_mood.rho_n[i] - dtdx * (F_star_ip_half_LF - F_star_im_half_LF)
        else
            # Candidate is good, accept it.
            pg.rhos[i] = cts_mood.rho_candidate[i]
        end
    end
end
