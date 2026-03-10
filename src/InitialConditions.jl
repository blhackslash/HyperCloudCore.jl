module InitialConditions

# Import necessary types from your main module. Adjust the path as needed.
using ..HyperbolicPDEs
using ..ParticleGrids
using ..SourceTerms
using LinearAlgebra
using StaticArrays

export InitialCondition, SmoothInitialCondition, ShockInitialCondition, setInitialConditions!,
       Gauss, Box, Sine, Riemann, EulerSmooth, EulerShockTube,SRiemann,
       getInitialCondition, get_discontinuity_points, euler1D_physical_fluxes, GAS_GAMMA_EULER


# --- Helper functions for Euler Equations ---
const GAS_GAMMA_EULER = 1.4

function pressure_from_euler_conserved(rho::Real, m::Real, E::Real)::Float64
    if rho < 1e-9; return 1e-9; end
    pressure = (GAS_GAMMA_EULER - 1.0) * (E - 0.5 * m^2 / rho)
    return max(pressure, 1e-9)
end

function euler1D_physical_fluxes(rho::Real, m::Real, E::Real)::NTuple{3, Float64}
    if rho < 1e-9; return (0.0, pressure_from_euler_conserved(1e-9,0.0,0.0), 0.0); end
    ux = m / rho
    p = pressure_from_euler_conserved(rho, m, E)
    return (m, m * ux + p, (E + p) * ux)
end

# --- 1. Abstract Type Hierarchy ---
abstract type InitialCondition end
abstract type SmoothInitialCondition <: InitialCondition end
abstract type ShockInitialCondition <: InitialCondition end
# --- 2. Concrete Structs and Functors for t=0 ---
# --- GENERALIZED, PARAMETRIC STRUCTS ---

# --- Set Initial Conditions ---
function setInitialConditions!(pg::ParticleGrid{1}, IC::InitialCondition)
    pg.rhos .= IC.(get_positions(pg))
end
function setInitialConditions!(pg::ParticleGrid{2}, IC::InitialCondition)
    map!(x -> IC(x[1],x[2]), pg.rhos, get_positions(pg))
end

# --- Base Case: Standard PDE (No Source Term, No Systems) ---
function setInitialConditions!(pg::ParticleGrid{D, M}, IC::InitialCondition) where {D, M}
    for i in 1:pg.meta.N
        val = IC(get_positions(pg)[i]...)
        
        # Handles both scalar IC outputs and tuple IC outputs seamlessly
        if M == 1
            pg.rhos[i, 1] = val
        else
            for m in 1:M
                pg.rhos[i, m] = val[m]
            end
        end
    end
end

# --- LOCAL Relaxation Initialization ---
function setInitialConditions!(
    pg::ParticleGrid{D, NK}, 
    st::RelaxationSourceTerm{D, N, NK, PDE},     
    IC::InitialCondition                              
) where {D, N, NK, PDE}
    
    for p_idx in 1:pg.meta.N
        # 1. Get macroscopic state at this position
        u_val = IC(get_positions(pg)[p_idx]...)
        
        # 2. Evaluate physical flux exactly once for the macro state
        flux_vals = flux(st.system_eq, u_val)
        
        # 3. Initialize each kinetic component to local equilibrium
        for k in 1:NK
            m_idx = st.kin2macro(k)
            dim = st.dimensions[k]
            
            f_val = SourceTerms.get_flux_component(flux_vals, m_idx, dim, Val(D))
            
            # Inline Maxwellian Initialization
            Mk = st.coefficients[k] * (u_val[m_idx] + st.interior_factors[k] * f_val / st.relax_speeds[k])
            pg.rhos[p_idx, k] = Mk
        end
    end
    return nothing
end

# --- NON-LOCAL Relaxation Initialization ---
function setInitialConditions!(
    pg::ParticleGrid{D, NK},
    st::NonLocalRelaxationSourceTerm{D, N, NK, PDE},
    IC::InitialCondition
) where {D, N, NK, PDE}
    N_particles = pg.meta.N
    
    # 1. Initialize grid to LOCAL equilibrium (V_k = c_k * U_m)
    for p_idx in 1:N_particles
        u_val = IC(get_positions(pg)[p_idx]...) 
        for k in 1:NK
            m_idx = st.kin2macro(k)
            pg.rhos[p_idx, k] = st.coefficients[m_idx] * u_val[m_idx]
        end
    end

    # 2. Compute the true initial potential T_0 using current grid state
    update_nonlocal_potential!(st, pg.rhos, pg)

    # 3. Re-initialize kinetic grids to the NON-LOCAL equilibrium: V_0 = M(U_0, T_0)
    for p_idx in 1:N_particles
        u_val = IC(get_positions(pg)[p_idx]...)
        for k in 1:NK
            m_idx = st.kin2macro(k)
            T_val = st.T_potential[p_idx, m_idx]
            
            # The correct Maxwellian formulation
            pg.rhos[p_idx, k] = st.coefficients[m_idx] * (
                u_val[m_idx] + st.interior_factor * T_val / st.relax_speeds[k]
            )
        end
    end
    
    @info "Initialized Non-Local Equilibrium (Max Potential: $(maximum(abs.(st.T_potential))))"
    return nothing
end
function update_potential_from_matrix!(st, system_pg, U_matrix)
    N_particles = size(U_matrix, 1)
    ensure_buffer_size!(st, N_particles)
    N_macro = size(U_matrix, 2)

    # Parallel path integration
    Threads.@threads for i in 2:N_particles
        uL = ntuple(m -> U_matrix[i-1, m], Val(N_macro))
        uR = ntuple(m -> U_matrix[i, m],   Val(N_macro))
        jump = path_integral(st.system_eq, uL, uR) # [cite: 108]
        for k in 1:N_macro
            st.T_potential[i, k] = jump[k]
        end
    end

    # Serial accumulation
    for k in 1:N_macro; st.T_potential[1, k] = 0.0; end
    for i in 2:N_particles
        for k in 1:N_macro
            st.T_potential[i, k] += st.T_potential[i-1, k]
        end
    end
end

"Gaussian distribution for scalar or system states."
struct Gauss{T, S} <: SmoothInitialCondition
    a::S      # Amplitude (can be a scalar or a vector/tuple)
    b::T      # Center (Float64 for 1D, NTuple for 2D)
    width::Float64
end
# Functors work for both scalar and system types due to broadcasting (vector * scalar)
(ic::Gauss{Float64, S})(x::Real) where S = ic.a .* exp(-((x - ic.b) / ic.width)^2)
(ic::Gauss{NTuple{2, Float64}, S})(x::Real, y::Real) where S = ic.a .* exp(-(((x - ic.b[1])^2 + (y - ic.b[2])^2) / ic.width^2))


"Box (Top-hat) distribution for scalar or system states."
struct Box{S} <: ShockInitialCondition
    u_background::S
    u_box::S
    x_start::Float64
    x_end::Float64
    y_start::Union{Float64, Nothing}
    y_end::Union{Float64, Nothing}

    # 1D Constructor
    Box(bg::S, val::S, xs, xe) where S = new{S}(bg, val, xs, xe, nothing, nothing)
    # 2D Constructor
    Box(bg::S, val::S, xs, xe, ys, ye) where S = new{S}(bg, val, xs, xe, ys, ye)
end
(ic::Box)(x::Real) = ic.x_start <= x <= ic.x_end ? ic.u_box : ic.u_background
(ic::Box)(x::Real, y::Real) = (ic.x_start <= x <= ic.x_end && !isnothing(ic.y_start) && ic.y_start <= y <= ic.y_end) ? ic.u_box : ic.u_background


"Sine wave for scalar states (systems would require more specific definition)."
struct Sine <: SmoothInitialCondition
    a::Float64
    b_period::Float64
    c_offset::Float64
end
(ic::Sine)(x::Real) = ic.a * sin(2.0 * pi * x / ic.b_period) + ic.c_offset


"Riemann problem (shock/rarefaction) for scalar or system states in 1D or 2D."
struct Riemann{T, S} <: ShockInitialCondition
    uL::S
    uR::S
    p0::T  # 1D: x0 position. 2D: point on line.
    n::T   # 1D: defaults to 1.0. 2D: normal vector.

    # 1D Constructor
    function Riemann(uL::S, uR::S, x0::Real) where S
        new{Float64, S}(uL, uR, Float64(x0), 1.0)
    end

    # 2D Constructor
    function Riemann(uL::S, uR::S, p0::NTuple{2, Real}, n_vec::NTuple{2, Real}) where S
        norm_n = LinearAlgebra.norm(n_vec)
        if norm_n < 1e-14; error("Normal vector for Riemann cannot be a zero vector."); end
        n_normalized = (n_vec[1] / norm_n, n_vec[2] / norm_n)
        p0_float = (Float64(p0[1]), Float64(p0[2]))
        new{NTuple{2, Float64}, S}(uL, uR, p0_float, n_normalized)
    end
end
(ic::Riemann{Float64, S})(x::Real) where S = x < ic.p0 ? ic.uL : ic.uR
(ic::Riemann{NTuple{2, Float64}, S})(x::Real, y::Real) where S = dot((x - ic.p0[1], y - ic.p0[2]), ic.n) < 0 ? ic.uL : ic.uR

"Smoothed Riemann problem (arctan) for scalar or system states."
struct SRiemann{T, S} <: SmoothInitialCondition
    uL::S
    uR::S
    x0::T      # Center of the transition
    width::T   # Smoothing width (steepness)
end

# Functor for t=0
# Uses broadcasting (.*, .+, .-) to handle both scalar (Burgers) and vector (Euler) states
function (ic::SRiemann)(x::Real)
    # Formula: u(x) = Avg - Diff/pi * atan((x-x0)/w)
    # Limits: x->-inf => uL; x->+inf => uR
    return @. 0.5 * (ic.uL + ic.uR) - (ic.uL - ic.uR) / pi * atan((x - ic.x0) / ic.width)
end

# --- NEW: Generalized Quadrant-based Riemann Problem ---
struct QuadrantRiemann{D, M, T} <: ShockInitialCondition
    u_states::NTuple{D,NTuple{M,Float64}} # Vector of states for each quadrant
    p0::T               # Center point of the quadrants

    function QuadrantRiemann(u_states::NTuple{D,NTuple{M,Float64}}, p0::T) where {D, M, T}
        if D != 2^(length(p0))
            error("For a D-dimensional problem!")
        end
        new{D, M, T}(u_states, p0)
    end
end

# 1D Functor (2 states: left, right)
function (ic::QuadrantRiemann{2, M, Float64})(x::Real) where M
    return x < ic.p0 ? ic.u_states[1] : ic.u_states[2]
end

# 2D Functor (4 states: BL, BR, TL, TR)
function (ic::QuadrantRiemann{4, M, NTuple{2, Float64}})(x::Real, y::Real) where M
    x0, y0 = ic.p0
    if x < x0 && y < y0       # Bottom-Left
        return ic.u_states[1]
    elseif x >= x0 && y < y0  # Bottom-Right
        return ic.u_states[2]
    elseif x < x0 && y >= y0  # Top-Left
        return ic.u_states[3]
    else # x >= x0 && y >= y0 # Top-Right
        return ic.u_states[4]
    end
end

# --- EULER SYSTEM ICS (1D) ---
struct EulerSmooth <: SmoothInitialCondition
    rho_spec::NamedTuple
    u_spec::NamedTuple
    p_spec::NamedTuple
end
function (ic::EulerSmooth)(x::Real)
    rho_s, u_s, p_s = ic.rho_spec, ic.u_spec, ic.p_spec
    rho_val = rho_s.off + rho_s.amp * exp(-((x - rho_s.mean) / rho_s.width)^2)
    u_val   = u_s.off   + u_s.amp   * aexp(-((x - u_s.mean) / u_s.width)^2)
    p_val   = p_s.off   + p_s.amp   * exp(-((x - p_s.mean) / p_s.width)^2)
    rho_val = max(rho_val, 1e-6); p_val = max(p_val, 1e-6)
    m_val = rho_val * u_val
    E_val = p_val / (GAS_GAMMA_EULER - 1.0) + 0.5 * rho_val * u_val^2
    return (rho_val, m_val, E_val)
end

struct EulerShockTube <: ShockInitialCondition
    stateL::NTuple{3, Float64} # (rho, u, p)
    stateR::NTuple{3, Float64} # (rho, u, p)
    x0::Float64
end
function (ic::EulerShockTube)(x::Real)
    rho_val, u_val, p_val = x < ic.x0 ? ic.stateL : ic.stateR
    rho_val = max(rho_val, 1e-6); p_val = max(p_val, 1e-6)
    m_val = rho_val * u_val
    E_val = p_val / (GAS_GAMMA_EULER - 1.0) + 0.5 * rho_val * u_val^2
    return (rho_val, m_val, E_val)
end

# --- 2. IC Functors
# --- Functors for t=0 ---
# (ic::Gauss{Float64})(x::Real) = ic.a * exp(-((x - ic.b) / ic.width)^2)
# (ic::Gauss{NTuple{2,Float64}})(x::Real, y::Real) = ic.a * exp(-(((x - ic.b[1])^2 + (y - ic.b[2])^2) / ic.width^2))

# (ic::Box)(x::Real) = (isnothing(ic.y_start) && ic.x_start <= x <= ic.x_end) ? ic.u_box : ic.u_background
# (ic::Box)(x::Real, y::Real) = (!isnothing(ic.y_start) && ic.x_start <= x <= ic.x_end && ic.y_start <= y <= ic.y_end) ? ic.u_box : ic.u_background

# (ic::Sine)(x::Real) = ic.a * sin(2.0 * pi * x / ic.b_period) + ic.c_offset

# # Functors for the new unified Riemann struct
# (ic::Riemann{Float64})(x::Real) = (x - ic.p0) * ic.n >= 0.0 ? ic.uR : ic.uL
# function (ic::Riemann{NTuple{2, Float64}})(x::Real, y::Real)
#     p_vec = (x - ic.p0[1], y - ic.p0[2])
#     dot_product = p_vec[1] * ic.n[1] + p_vec[2] * ic.n[2]
#     return dot_product >= 0.0 ? ic.uR : ic.uL
# end

# --- 3. Analytical Solution Functors (t>0) using Multiple Dispatch ---
function (ic::InitialCondition)(x::Real, t::Real, eq::HyperbolicPDE{D,N}, pg::ParticleGrid1D) where {D,N}
    if hasfield(ic,:reference)
        return reference(x,t)
    else
        error("No analytic solution implemented! A reference solution has to be given!")
    end
end

# --- For Linear Advection (General Solution) ---
function (ic::InitialCondition)(x::Real, t::Real, eq::LinearAdvection{1}, pg::ParticleGrid1D)
    x0 = x - eq.vel[1] * t
    if pg.bc == :periodic
        domain_length = pg.xmax - pg.xmin
        x0_mapped = pg.xmin + mod(x0 - pg.xmin, domain_length)
        return ic(x0_mapped)
    else # :fixed or :outflow (infinite domain assumption)
        return ic(x0)
    end
end



function (ic::InitialCondition)(x::Real, y::Real, t::Real, eq::LinearAdvection{2}, pg::ParticleGrid2D)
    x0 = x - eq.vel[1] * t
    y0 = y - eq.vel[2] * t
    if pg.bc == :periodic
        x0_wrapped = pg.xmin + mod(x0 - pg.xmin, pg.xmax - pg.xmin)
        y0_wrapped = pg.ymin + mod(y0 - pg.ymin, pg.ymax - pg.ymin)
        return ic(x0_wrapped, y0_wrapped)
    else 
        return ic(x0, y0)
    end
end

function (ic::InitialCondition)(pos::Union{Tuple{<:Real,<:Real},SVector{2,<:Real}}, t::Real, eq::LinearAdvection{2}, pg::ParticleGrid2D)
    x = pos[1]
    y = pos[2]
    x0 = x - eq.vel[1] * t
    y0 = y - eq.vel[2] * t
    if pg.bc == :periodic
        x0_wrapped = pg.xmin + mod(x0 - pg.xmin, pg.xmax - pg.xmin)
        y0_wrapped = pg.ymin + mod(y0 - pg.ymin, pg.ymax - pg.ymin)
        return ic(x0_wrapped, y0_wrapped)
    else 
        return ic(x0, y0)
    end
end

(ic::InitialCondition)(x::Real, y::Real, t::Real, eq::LinearAdvection{1}, pg::ParticleGrid) = ic(x-eq.vel[1]*t, y) # Dispatch to 2D functor


# --- For Burger's Equation (Specific to each IC Type) ---
function (ic::Sine)(x::Real, t::Real, eq::BurgersEquation, pg::ParticleGrid1D; tol::Real = 1e-10, max_iter::Int = 100)
    if t <= 1e-12; return ic(x); end
    u_current = ic(x)
    for _ in 1:max_iter
        u_next = ic.a * sin(2.0 * pi * (x - u_current * t) / ic.b_period) + ic.c_offset
        if abs(u_next - u_current) < tol; return u_next; end
        u_current = u_next
    end
    @warn "sineInitAna: Fixed-point iteration did not converge at x=$x, t=$t."
    return u_current
end

function (ic::Box)(x::Real, t::Real, eq::BurgersEquation, pg::ParticleGrid1D)
    # --- Pre-computation and edge cases ---
    if t <= 1e-12; return ic(x); end
    if abs(ic.u_box - ic.u_background) < 1e-12; return ic.u_background; end

    if ic.u_box > ic.u_background
        # --- Top-hat case: Rarefaction at left, Shock at right ---
        
        # Time of interaction: when the head of the rarefaction fan catches the shock
        # This occurs when the plateau of u_box disappears.
        delta_u = ic.u_box - ic.u_background
        t_interaction = 2.0 * (ic.x_end - ic.x_start) / delta_u
        
        if t < t_interaction
            # --- Phase 1: Waves evolve independently ---
            s_shock = (ic.u_box + ic.u_background) / 2.0
            x_shock_front = ic.x_end + s_shock * t
            x_fan_head = ic.x_start + ic.u_box * t
            
            if x < ic.x_start + ic.u_background * t
                return ic.u_background
            elseif x < x_fan_head
                # Inside rarefaction fan
                return (x - ic.x_start) / t
            elseif x < x_shock_front
                # Plateau region
                return ic.u_box
            else 
                # Behind shock
                return ic.u_background
            end
        else
            # --- Phase 2: Shock has merged with rarefaction fan ---
            # The shock path is now x_s(t) = x_start + u_background*t + C*sqrt(t)
            C = sqrt(2.0 * (ic.x_end - ic.x_start) * delta_u)
            x_shock_interacting = ic.x_start + ic.u_background * t + C * sqrt(t)

            if x < ic.x_start + ic.u_background * t
                return ic.u_background
            elseif x < x_shock_interacting
                # Inside rarefaction fan, up to the interacting shock
                return (x - ic.x_start) / t
            else 
                # Behind the interacting shock
                return ic.u_background
            end
        end

    else # u_box < u_background
        # --- Well case: Shock at left, Rarefaction at right ---

        # Time of interaction: when the shock front catches the tail of the rarefaction fan
        delta_u = ic.u_background - ic.u_box
        t_interaction = 2.0 * (ic.x_end - ic.x_start) / delta_u

        if t < t_interaction
            # --- Phase 1: Waves evolve independently ---
            s_shock = (ic.u_background + ic.u_box) / 2.0
            x_shock_front = ic.x_start + s_shock * t
            x_fan_tail = ic.x_end + ic.u_box * t
            
            if x < x_shock_front
                return ic.u_background
            elseif x < x_fan_tail
                return ic.u_box
            elseif x < ic.x_end + ic.u_background * t
                # Inside rarefaction fan
                return (x - ic.x_end) / t
            else
                return ic.u_background
            end
        else
            # --- Phase 2: Shock has entered the rarefaction fan ---
            # The shock path is now x_s(t) = x_end + u_background*t - C*sqrt(t)
            C = sqrt(2.0 * (ic.x_end - ic.x_start) * delta_u)
            x_shock_interacting = ic.x_end + ic.u_background * t - C * sqrt(t)

            if x < x_shock_interacting
                return ic.u_background
            elseif x < ic.x_end + ic.u_background * t
                # Inside rarefaction fan, to the right of the interacting shock
                return (x - ic.x_end) / t
            else
                return ic.u_background
            end
        end
    end
end

"""
Analytical solution for the 1D Shock Tube (Physical Euler) using primitive variables.
This follows the iterative approach to find the star-region pressure p_star.
"""
function (ic::Riemann{Float64, NTuple{3, Float64}})(x::Real, t::Real, eq::LEuler1D, pg::ParticleGrid1D)
    if t <= 1e-10; return ic(x); end

    gamma = GAS_GAMMA_EULER
    rho_L, u_L, p_L = ic.uL
    rho_R, u_R, p_R = ic.uR
    x0 = ic.p0

    # 1. Iterative solve for Star-Region Pressure (p_star)
    c_L = sqrt(gamma * p_L / rho_L)
    c_R = sqrt(gamma * p_R / rho_R)

    # Function to solve f(p_star) = 0
    f_wave(p_s, p_k, rho_k, c_k) = p_s > p_k ? 
        (p_s - p_k) * sqrt((2.0/((gamma+1)*rho_k)) / (p_s + p_k*(gamma-1)/(gamma+1))) : # Shock
        (2c_k/(gamma-1)) * ((p_s/p_k)^((gamma-1)/(2gamma)) - 1.0)                      # Rarefaction

    p_star = 0.5 * (p_L + p_R) # Initial guess
    for _ in 1:50
        f = f_wave(p_star, p_L, rho_L, c_L) + f_wave(p_star, p_R, rho_R, c_R) + (u_R - u_L)
        if abs(f) < 1e-8; break; end
        # Numerical derivative for Newton step
        df = (f_wave(p_star * 1.01, p_L, rho_L, c_L) + f_wave(p_star * 1.01, p_R, rho_R, c_R) + (u_R - u_L) - f) / (0.01 * p_star)
        p_star = max(1e-9, p_star - f/df)
    end

    # 2. Velocity in Star Region
    u_star = 0.5 * (u_L + u_R) + 0.5 * (f_wave(p_star, p_R, rho_R, c_R) - f_wave(p_star, p_L, rho_L, c_L))

    # 3. Sample the solution at (x, t)
    s = (x - x0) / t
    rho, u, p = 0.0, 0.0, 0.0

    if s < u_star # Left of Contact Discontinuity
        if p_star > p_L # Left Shock
            s_shock = u_L - c_L * sqrt(((gamma+1)/(2gamma))*(p_star/p_L) + (gamma-1)/(2gamma))
            if s < s_shock; (rho, u, p) = (rho_L, u_L, p_L)
            else; (rho, u, p) = (rho_L * (p_star/p_L + (gamma-1)/(gamma+1)) / (1 + (p_star/p_L)*(gamma-1)/(gamma+1)), u_star, p_star); end
        else # Left Rarefaction
            s_head = u_L - c_L
            s_tail = u_star - c_L * (p_star/p_L)^((gamma-1)/(2gamma))
            if s < s_head; (rho, u, p) = (rho_L, u_L, p_L)
            elseif s > s_tail; (rho, u, p) = (rho_L * (p_star/p_L)^(1/gamma), u_star, p_star)
            else # Inside fan
                u = (2.0/(gamma+1)) * (c_L + (gamma-1)/2.0 * u_L + s)
                c = c_L - (gamma-1)/2.0 * (u - u_L)
                rho = rho_L * (c/c_L)^(2.0/(gamma-1))
                p = p_L * (rho/rho_L)^gamma
            end
        end
    else # Right of Contact Discontinuity
        if p_star > p_R # Right Shock
            s_shock = u_R + c_R * sqrt(((gamma+1)/(2gamma))*(p_star/p_R) + (gamma-1)/(2gamma))
            if s > s_shock; (rho, u, p) = (rho_R, u_R, p_R)
            else; (rho, u, p) = (rho_R * (p_star/p_R + (gamma-1)/(gamma+1)) / (1 + (p_star/p_R)*(gamma-1)/(gamma+1)), u_star, p_star); end
        else # Right Rarefaction
            s_head = u_R + c_R
            s_tail = u_star + c_R * (p_star/p_R)^((gamma-1)/(2gamma))
            if s > s_head; (rho, u, p) = (rho_R, u_R, p_R)
            elseif s < s_tail; (rho, u, p) = (rho_R * (p_star/p_R)^(1/gamma), u_star, p_star)
            else # Inside fan
                u = (2.0/(gamma+1)) * (-c_R + (gamma-1)/2.0 * u_R + s)
                c = c_R + (gamma-1)/2.0 * (u - u_R)
                rho = rho_R * (c/c_R)^(2.0/(gamma-1))
                p = p_R * (rho/rho_R)^gamma
            end
        end
    end

    return (rho, u, p)
end

function (ic::Riemann)(x::Real, t::Real, eq::BurgersEquation, pg::ParticleGrid1D)
    if t <= 1e-12; return ic(x); end
    if ic.uL > ic.uR # Shock
        s = (ic.uL + ic.uR) / 2.0
        shock_pos = ic.p0 + s * t
        return x < shock_pos ? ic.uL : ic.uR
    else # Rarefaction
        x_fan_tail = ic.p0 + ic.uL * t
        x_fan_head = ic.p0 + ic.uR * t
        if x < x_fan_tail; return ic.uL;
        elseif x > x_fan_head; return ic.uR;
        else return (x - ic.p0) / t; end
    end
end

function (ic::Riemann)(x::Real, t::Real, eq::TestU3Equation, pg::ParticleGrid1D)
    if t <= 1e-12; return ic(x); end
    if ic.uL > ic.uR # Shock
        s = (ic.uL^2 + ic.uL * ic.uR + ic.uR^2) / 3.0
        shock_pos = ic.p0 + s * t
        return x < shock_pos ? ic.uL : ic.uR
    else # Rarefaction
        error("uL < uR not implemented yet!")
    end
end

# --- Analytical Solution for SRiemann (delegates to sharp Riemann) ---
function (ic::SRiemann)(x::Real, t::Real, eq::BurgersEquation, pg::ParticleGrid1D)
    # Construct an equivalent sharp Riemann problem to get the "Real" solution
    sharp_ic = Riemann(ic.uL, ic.uR, ic.x0)
    return sharp_ic(x, t, eq, pg)
end

# --- Discontinuity Tracking for SRiemann ---
# Essential for QuadGK to know where the shock/fans are expected to be
function get_discontinuity_points(ic::SRiemann, eq::BurgersEquation, t::Real, pg::ParticleGrid1D)
    sharp_ic = Riemann(ic.uL, ic.uR, ic.x0)
    return get_discontinuity_points(sharp_ic, eq, t, pg)
end

# --- NEW: Analytical Solution for 2D Burgers with Planar Riemann IC ---
function (ic::Riemann{NTuple{2, Float64}})(pos::Union{NTuple{2,Float64},SVector{2,Float64}}, t::Real, eq::BurgersEquation2D, pg::ParticleGrid2D)
    x = pos[1]
    y = pos[2]
    
    if t <= 1e-12; return ic(x, y); end

    # Project the problem onto the 1D normal direction
    # d is the perpendicular distance from the initial line
    d = dot((x - ic.p0[1], y - ic.p0[2]), ic.n)
    
    # The effective 1D characteristic speed is u_n = u * (n_x + n_y)
    n_sum = ic.n[1] + ic.n[2]
    
    if ic.uL > ic.uR # --- Shock Wave ---
        # Rankine-Hugoniot shock speed in the normal direction
        s = 0.5 * (ic.uL + ic.uR) * n_sum
        shock_pos_d = s * t
        return d < shock_pos_d ? ic.uL : ic.uR
    else # --- Rarefaction Wave ---
        fan_tail_d = ic.uL * n_sum * t
        fan_head_d = ic.uR * n_sum * t
        
        if d < fan_tail_d
            return ic.uL
        elseif d > fan_head_d
            return ic.uR
        else # Inside the rarefaction fan
            if abs(t * n_sum) < 1e-14
                return 0.5 * (ic.uL + ic.uR) # Avoid division by zero at t=0 or if n_sum=0
            end
            return d / (t * n_sum)
        end
    end
end

# Fallback for ICs without a specific analytical solution for Burger's
(ic::Gauss)(x::Real, t::Real, eq::BurgersEquation, pg::ParticleGrid1D) = NaN


# --- 3. Analytical Solution Functors (t>0) ---

# --- For Euler Equations ---
(ic::InitialCondition)(x::Real, t::Real, eq::Euler1D, pg::ParticleGrid1D) = error("Analytical solution for this Euler IC is not implemented.")


# --- Analytical Solution Functor for Euler Shock Tube (t>0) ---
function (ic::EulerShockTube)(x::Real, t::Real, eq::Euler1D, pg::ParticleGrid1D)
    if pg.bc == :periodic
        @warn "Analytical Riemann solver for Euler is not defined for periodic BCs."
        return (NaN, NaN, NaN)
    end
    if t <= 1e-9; return ic(x); end

    # --- 1. Extract Initial States and Parameters ---
    gamma = GAS_GAMMA_EULER
    rho_L, u_L, p_L = ic.stateL
    rho_R, u_R, p_R = ic.stateR
    x0 = ic.x0
    
    # --- 2. Solve for Pressure in the Star Region (p_star) ---
    c_L = sqrt(gamma * p_L / rho_L)
    c_R = sqrt(gamma * p_R / rho_R)
    
    function pressure_func(p_star_guess::Real)
        local f_L, f_R
        # Left wave
        if p_star_guess > p_L # Shock
            A_L = 2.0 / ((gamma + 1.0) * rho_L); B_L = p_L * (gamma - 1.0) / (gamma + 1.0)
            f_L = (p_star_guess - p_L) * sqrt(A_L / (p_star_guess + B_L))
        else # Rarefaction
            f_L = (2.0 * c_L / (gamma - 1.0)) * ((p_star_guess / p_L)^((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        end
        # Right wave
        if p_star_guess > p_R # Shock
            A_R = 2.0 / ((gamma + 1.0) * rho_R); B_R = p_R * (gamma - 1.0) / (gamma + 1.0)
            f_R = (p_star_guess - p_R) * sqrt(A_R / (p_star_guess + B_R))
        else # Rarefaction
            f_R = (2.0 * c_R / (gamma - 1.0)) * ((p_star_guess / p_R)^((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        end
        return f_L + f_R + (u_R - u_L)
    end

    p_star = 0.5 * (p_L + p_R) # Initial guess
    for _ in 1:100 # Newton-Raphson iterations
        f_p = pressure_func(p_star)
        if abs(f_p) < 1e-9; break; end
        dfdp = (pressure_func(p_star * 1.001) - f_p) / (p_star * 0.001)
        p_star -= f_p / (dfdp + 1e-9)
        if p_star < 0; p_star = 1e-9; end
    end

    # --- 3. Calculate Star Region Velocity (u_star) ---
    local f_L_final
    if p_star > p_L # Left shock
        A_L = 2.0 / ((gamma + 1.0) * rho_L); B_L = p_L * (gamma - 1.0) / (gamma + 1.0)
        f_L_final = (p_star - p_L) * sqrt(A_L / (p_star + B_L))
    else # Left rarefaction
        f_L_final = (2.0 * c_L / (gamma - 1.0)) * ((p_star / p_L)^((gamma - 1.0) / (2.0 * gamma)) - 1.0)
    end
    u_star = u_L - f_L_final

    # --- 4. Determine Wave Speeds and Regions ---
    local rho_star_L, rho_star_R, S_L, S_R, S_head_L, S_tail_L, S_head_R, S_tail_R
    
    if p_star > p_L # Left Shock
        S_L = u_L - c_L * sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / p_L) + (gamma - 1.0) / (2.0 * gamma))
        rho_star_L = rho_L * ((p_star / p_L) + (gamma - 1.0) / (gamma + 1.0)) / (1.0 + (p_star / p_L) * (gamma - 1.0) / (gamma + 1.0))
    else # Left Rarefaction
        S_head_L = u_L - c_L
        c_star_L = c_L * (p_star / p_L)^((gamma - 1.0) / (2.0 * gamma))
        S_tail_L = u_star - c_star_L
        rho_star_L = rho_L * (p_star / p_L)^(1.0 / gamma)
    end

    if p_star > p_R # Right Shock
        S_R = u_R + c_R * sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / p_R) + (gamma - 1.0) / (2.0 * gamma))
        rho_star_R = rho_R * ((p_star / p_R) + (gamma - 1.0) / (gamma + 1.0)) / (1.0 + (p_star / p_R) * (gamma - 1.0) / (gamma + 1.0))
    else # Right Rarefaction
        S_head_R = u_R + c_R
        c_star_R = c_R * (p_star / p_R)^((gamma - 1.0) / (2.0 * gamma))
        S_tail_R = u_star + c_star_R
        rho_star_R = rho_R * (p_star / p_R)^(1.0 / gamma)
    end

    S_contact = u_star

    # --- 5. Find Solution at Query Point (x,t) ---
    s_query = (x - x0) / t
    local rho_final, u_final, p_final

    if s_query <= S_contact # Left of contact
        if p_star > p_L # Left Shock
            rho_final, u_final, p_final = s_query <= S_L ? (rho_L, u_L, p_L) : (rho_star_L, u_star, p_star)
        else # Left Rarefaction
            if s_query <= S_head_L
                rho_final, u_final, p_final = rho_L, u_L, p_L
            elseif s_query >= S_tail_L
                rho_final, u_final, p_final = rho_star_L, u_star, p_star
            else # Inside rarefaction fan
                u_final = (2.0 / (gamma + 1.0)) * (c_L + (gamma - 1.0) / 2.0 * u_L + s_query)
                c_final = c_L - (gamma - 1.0) / 2.0 * (u_final - u_L)
                rho_final = rho_L * (c_final / c_L)^(2.0 / (gamma - 1.0))
                p_final = p_L * (rho_final / rho_L)^gamma
            end
        end
    else # Right of contact
        if p_star > p_R # Right Shock
            rho_final, u_final, p_final = s_query >= S_R ? (rho_R, u_R, p_R) : (rho_star_R, u_star, p_star)
        else # Right Rarefaction
            if s_query >= S_head_R
                rho_final, u_final, p_final = rho_R, u_R, p_R
            elseif s_query <= S_tail_R
                rho_final, u_final, p_final = rho_star_R, u_star, p_star
            else # Inside rarefaction fan
                u_final = (2.0 / (gamma + 1.0)) * (-c_R + (gamma - 1.0) / 2.0 * u_R + s_query)
                c_final = c_R + (gamma - 1.0) / 2.0 * (u_final - u_R)
                rho_final = rho_R * (c_final / c_R)^(2.0 / (gamma - 1.0))
                p_final = p_R * (rho_final / rho_R)^gamma
            end
        end
    end

    # --- 6. Convert final primitive variables to conserved variables ---
    m_final = rho_final * u_final
    E_final = p_final / (gamma - 1.0) + 0.5 * rho_final * u_final^2
    
    return (rho_final, m_final, E_final)
end

# --- 4. Factory Function (SIMPLIFIED) ---
function getInitialCondition(name::String, params::Tuple)
    if name == "gauss"; return Gauss(params...);
    elseif name == "box"; return Box(params...);
    elseif name == "sine"; return Sine(params...);
    elseif name == "riemann"; return Riemann(params...);
    elseif name == "s_riemann"; return SRiemann(params...);
    elseif name == "q_riemann"; return QuadrantRiemann(params...);
    elseif name == "eulerSmooth"; return EulerSmooth(params...);
    elseif name == "eulerShockTube"; return EulerShockTube(params...);
    else error("Unknown initFunc name: $name"); end
end

"""
    get_discontinuity_points(ic::EulerShockTube, eq::Euler1D, t::Real, pg::ParticleGrid1D)

Calculates the positions of the shock, contact, and rarefaction fan edges
for the Euler shock tube problem at a given time `t`. This is essential for
providing accurate integration points to `QuadGK`.
"""
function get_discontinuity_points(ic::EulerShockTube, eq::Euler1D, t::Real, pg::ParticleGrid1D)
    # --- 1. Extract Initial States and Parameters ---
    gamma = GAS_GAMMA_EULER
    rho_L, u_L, p_L = ic.stateL
    rho_R, u_R, p_R = ic.stateR
    x0 = ic.x0
    
    # --- 2. Solve for Pressure and Velocity in the Star Region ---
    c_L = sqrt(gamma * p_L / rho_L)
    c_R = sqrt(gamma * p_R / rho_R)
    
    function pressure_func(p_star_guess::Real)
        local f_L, f_R
        # Left wave
        if p_star_guess > p_L # Shock
            A_L = 2.0 / ((gamma + 1.0) * rho_L); B_L = p_L * (gamma - 1.0) / (gamma + 1.0)
            f_L = (p_star_guess - p_L) * sqrt(A_L / (p_star_guess + B_L))
        else # Rarefaction
            f_L = (2.0 * c_L / (gamma - 1.0)) * ((p_star_guess / p_L)^((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        end
        # Right wave
        if p_star_guess > p_R # Shock
            A_R = 2.0 / ((gamma + 1.0) * rho_R); B_R = p_R * (gamma - 1.0) / (gamma + 1.0)
            f_R = (p_star_guess - p_R) * sqrt(A_R / (p_star_guess + B_R))
        else # Rarefaction
            f_R = (2.0 * c_R / (gamma - 1.0)) * ((p_star_guess / p_R)^((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        end
        return f_L + f_R + (u_R - u_L)
    end

    p_star = 0.5 * (p_L + p_R) # Initial guess
    for _ in 1:100 # Newton-Raphson-like iterations
        f_p = pressure_func(p_star)
        if abs(f_p) < 1e-9; break; end
        # Use a simple secant method to approximate the derivative
        dfdp = (pressure_func(p_star * 1.001) - f_p) / (p_star * 0.001)
        p_star -= f_p / (dfdp + 1e-9) # Add epsilon for stability
        if p_star < 0; p_star = 1e-9; end
    end

    local f_L_final
    if p_star > p_L # Left shock
        A_L = 2.0 / ((gamma + 1.0) * rho_L); B_L = p_L * (gamma - 1.0) / (gamma + 1.0)
        f_L_final = (p_star - p_L) * sqrt(A_L / (p_star + B_L))
    else # Left rarefaction
        f_L_final = (2.0 * c_L / (gamma - 1.0)) * ((p_star / p_L)^((gamma - 1.0) / (2.0 * gamma)) - 1.0)
    end
    u_star = u_L - f_L_final

    # --- 3. Calculate Wave Positions ---
    points = Float64[]
    
    # Left Wave Position(s)
    if p_star > p_L # Left Shock
        S_L = u_L - c_L * sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / p_L) + (gamma - 1.0) / (2.0 * gamma))
        push!(points, x0 + S_L * t)
    else # Left Rarefaction
        S_head_L = u_L - c_L
        c_star_L = c_L * (p_star / p_L)^((gamma - 1.0) / (2.0 * gamma))
        S_tail_L = u_star - c_star_L
        push!(points, x0 + S_head_L * t, x0 + S_tail_L * t)
    end

    # Contact Discontinuity Position
    push!(points, x0 + u_star * t)

    # Right Wave Position(s)
    if p_star > p_R # Right Shock
        S_R = u_R + c_R * sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / p_R) + (gamma - 1.0) / (2.0 * gamma))
        push!(points, x0 + S_R * t)
    else # Right Rarefaction
        S_head_R = u_R + c_R
        c_star_R = c_R * (p_star / p_R)^((gamma - 1.0) / (2.0 * gamma))
        S_tail_R = u_star + c_star_R
        push!(points, x0 + S_head_R * t, x0 + S_tail_R * t)
    end
    
    return unique(sort(points))
end


# --- 5. Discontinuity Finder ---
"""
    get_discontinuity_points(ic, eq, t, pg)

Calculates the current positions of any discontinuities for QuadGK.
Uses multiple dispatch on the initial condition type and equation type.
"""
# Default case, no interval splitting
get_discontinuity_points(ic::InitialCondition, eq::HyperbolicPDE{D,N}, t::Real, pg::ParticleGrid{D}) where {D,N} = Float64[]

# For shock ICs with linear advection -> track the initial jumps
function get_discontinuity_points(ic::ShockInitialCondition, eq::LinearAdvection{1}, t::Real, pg::ParticleGrid1D)
    points = Float64[]
    xmin, xmax = pg.xmin, pg.xmax
    domain_length = xmax - xmin
    
    initial_points = if ic isa Box; [ic.x_start, ic.x_end]; else [ic.p0]; end
    
    for pt in initial_points
        advected_pos = pt + eq.vel[1] * t
        if pg.bc == :periodic
            advected_pos = xmin + mod(advected_pos - xmin, domain_length)
        end
        push!(points, advected_pos)
    end
    return unique(sort(points))
end

# For Riemann with Burger's -> track shock or rarefaction edges
function get_discontinuity_points(ic::Riemann, eq::BurgersEquation, t::Real, pg::ParticleGrid1D)
    if ic.uL > ic.uR # Shock
        s = (ic.uL + ic.uR) / 2.0
        return [ic.p0 + s * t]
    else # Rarefaction
        return [ic.p0 + ic.uL * t, ic.p0 + ic.uR * t]
    end
end

# For Box with Burger's -> track shock and rarefaction edges
function get_discontinuity_points(ic::Box, eq::BurgersEquation, t::Real, pg::ParticleGrid1D)
    s_shock = (ic.u_box + ic.u_background) / 2.0
    if ic.u_box > ic.u_background # Top-hat
        res = [ic.x_start + ic.u_background * t]
        rare_pos = ic.x_start + ic.u_box * t
        shock_pos = ic.x_end + s_shock * t
        if rare_pos < shock_pos
            push!(res, rare_pos)
        end
        push!(res, shock_pos)
        return res
    else # Well
        return [ic.x_start + s_shock * t, ic.x_end + ic.u_box * t, ic.x_end + ic.u_background * t]
    end
end

# For Sine with Burger's -> track the characteristic from the point of steepest descent
function get_discontinuity_points(ic::Sine, eq::BurgersEquation, t::Real, pg::ParticleGrid1D)
    # Shock forms at the point with the most negative slope.
    # For a*sin(2pi*x/b), this is at x = b/2 (if a>0) or x=0 (if a<0)
    x_break = ic.a > 0 ? ic.b_period / 2.0 : 0.0
    u_at_break = ic(x_break)
    return [x_break + u_at_break * t]
end

# For Gauss with Burger's -> track the characteristic from the inflection point
function get_discontinuity_points(ic::Gauss, eq::BurgersEquation, t::Real, pg::ParticleGrid1D)
    # Shock forms at the point with the most negative slope (an inflection point).
    x_break = ic.a > 0 ? ic.b + ic.width / sqrt(2.0) : ic.b - ic.width / sqrt(2.0)
    u_at_break = ic(x_break)
    return [x_break + u_at_break * t]
end


end # module InitialConditions