using StaticArrays
using LinearAlgebra

# =========================================================================
# 1. SET INITIAL CONDITIONS
# =========================================================================

function setInitialConditions!(pg::ParticleGrid{D, M}, eq::HyperbolicPDE, IC::InitialCondition) where {D, M}
    positions = get_positions(pg)
    
    @inbounds for i in 1:pg.meta.N
        # Strict assignment: IC(pos) must return an State{M}
        pg.rhos[i] = IC(positions[i])
    end
    return nothing
end

(ic::Gauss)(pos::Space{D}) where {D} = @. ic.a * exp(-sum(abs2, pos - ic.b) / ic.width^2)
(ic::Box)(pos::Space{D}) where {D} = all(ic.mins .<= pos .<= ic.maxs) ? ic.u_box : ic.u_bg
(ic::Sine)(pos::Space{D}) where {D} = ic.a * sin(2.0 * pi * sum(pos ./ ic.period)) + ic.c_offset
(ic::Riemann)(pos::Space{D}) where {D} = dot(pos - ic.p0, ic.n) < 0 ? ic.uL : ic.uR
function (ic::SRiemann)(pos::Space{D}) where {D}
    dist = dot(pos - ic.p0, ic.n)
    return @. 0.5 * (ic.uL + ic.uR) - (ic.uL - ic.uR) / pi * atan(dist / ic.width)
end
function (ic::QuadrantRiemann{D, M, N})(pos::Space{D}) where {D, M, N}
    # Binary encoding: Left/Bottom adds 0, Right/Top adds 2^(d-1)
    idx = 1
    for d in 1:D
        if pos[d] >= ic.p0[d]
            idx += 2^(d-1)
        end
    end
    return ic.u_states[idx]
end
function (ic::EulerShockTube)(pos::Space{D}) where {D}
    is_left = dot(pos - ic.p0, ic.n) < 0
    rho_val, u_val, p_val = is_left ? ic.uL : ic.uR
    
    rho_val = max(rho_val, 1e-6)
    p_val = max(p_val, 1e-6)
    
    m_val = rho_val * u_val
    E_val = p_val / (GAS_GAMMA_EULER - 1.0) + 0.5 * rho_val * u_val^2
    
    return SVector{3, Float64}(rho_val, m_val, E_val)
end

# =========================================================================
# 3. FACTORY FUNCTION
# =========================================================================
function getInitialCondition(name::String, p::Tuple)
    if name == "gauss"
        return Gauss(param2uvec(p[1]), param2xvec(p[2]), Float64(p[3]))
        
    elseif name == "box"
        if length(p) == 4
            # 1D/Multi-D unified format: Box(bg, box, mins, maxs)
            return Box(param2uvec(p[1]), param2uvec(p[2]), param2xvec(p[3]), param2xvec(p[4]))
        elseif length(p) == 6
            # Backwards compatibility for old 2D format: Box(bg, box, xmin, xmax, ymin, ymax)
            mins = param2xvec((p[3], p[5]))
            maxs = param2xvec((p[4], p[6]))
            return Box(param2uvec(p[1]), param2uvec(p[2]), mins, maxs)
        else
            error("Invalid number of parameters for Box")
        end
        
    elseif name == "sine"
        return Sine(param2uvec(p[1]), param2xvec(p[2]), param2uvec(p[3]))
        
    elseif name == "riemann"
        uL = param2uvec(p[1])
        uR = param2uvec(p[2])
        p0 = param2xvec(p[3])
        # If normal vector not provided, default to pointing in +X direction
        n  = length(p) > 3 ? normalize(param2xvec(p[4])) : SVector{length(p0), Float64}(ntuple(i -> i==1 ? 1.0 : 0.0, length(p0)))
        
        return Riemann(uL, uR, p0, n)
        
    elseif name == "s_riemann"
        uL = param2uvec(p[1])
        uR = param2uvec(p[2])
        p0 = param2xvec(p[3])
        if length(p) == 4
            n = SVector{length(p0), Float64}(ntuple(i -> i==1 ? 1.0 : 0.0, length(p0)))
            return SRiemann(uL, uR, p0, n, Float64(p[4]))
        else
            n = normalize(param2xvec(p[4]))
            return SRiemann(uL, uR, p0, n, Float64(p[5]))
        end
        
    elseif name == "q_riemann"
        states = ntuple(i -> param2uvec(p[1][i]), length(p[1]))
        p0 = param2xvec(p[2])
        return QuadrantRiemann(states, p0)
        
    elseif name == "eulerShockTube"
        uL = param2uvec(p[1])
        uR = param2uvec(p[2])
        p0 = param2xvec(p[3])
        n  = length(p) > 3 ? normalize(param2xvec(p[4])) : SVector{length(p0), Float64}(ntuple(i -> i==1 ? 1.0 : 0.0, length(p0)))
        
        return EulerShockTube(uL, uR, p0, n)
        
    else 
        error("Unknown initFunc name: $name")
    end
end

# =========================================================================
# ANALYTICAL SOLUTIONS (t > 0)
# =========================================================================

# Fallback
(ic::InitialCondition)(pos::Space{D}, t::Real, eq::HyperbolicPDE, pg::ParticleGrid{D}) where {D} = error("No analytic solution implemented for this combination!")

# --- LINEAR ADVECTION (Unified N-Dimensional) ---
function (ic::InitialCondition)(pos::Space{D}, t::Real, eq::LinearAdvection{M, D}, pg::ParticleGrid{D}) where {M, D}
    # Assume M=1 velocity for the purely advective shift
    vel = Space{D}(ntuple(d -> eq.vel[d][1], Val(D)))
    pos0 = pos - vel * t
    
    if pg.meta.bc == :periodic
        pos0 = pg.meta.mins .+ mod.(pos0 .- pg.meta.mins, pg.meta.maxs .- pg.meta.mins)
    end
    
    return ic(pos0)
end

# --- BURGERS EQUATION 1D ---
function (ic::Sine)(pos::Space{1}, t::Real, eq::BurgersEquation, pg::ParticleGrid{1}; tol=1e-10, max_iter=100)
    if t <= 1e-12; return ic(pos); end
    x = pos[1]
    u_curr = ic(pos)[1]
    
    for _ in 1:max_iter
        u_next = ic.a * sin(2.0 * pi * (x - u_curr * t) / ic.period[1]) + ic.c_offset
        if abs(u_next - u_curr) < tol; return State{1}(u_next); end
        u_curr = u_next
    end
    @warn "Fixed-point iteration did not converge at x=$x, t=$t."
    return State{1}(u_curr)
end

function (ic::Riemann)(pos::Space{1}, t::Real, eq::BurgersEquation, pg::ParticleGrid{1})
    if t <= 1e-12; return ic(pos); end
    x = pos[1]; x0 = ic.p0[1]
    uL = ic.uL[1]; uR = ic.uR[1]
    
    if uL > uR # Shock
        s = 0.5 * (uL + uR)
        return x < x0 + s * t ? ic.uL : ic.uR
    else # Rarefaction
        if x < x0 + uL * t; return ic.uL
        elseif x > x0 + uR * t; return ic.uR
        else return State{1}((x - x0) / t)
        end
    end
end

function (ic::SRiemann)(pos::Space{1}, t::Real, eq::BurgersEquation, pg::ParticleGrid{1})
    # SRiemann Analytical defaults to sharp Riemann
    return Riemann(ic.uL, ic.uR, ic.p0, ic.n)(pos, t, eq, pg)
end

function (ic::Box)(pos::Space{1}, t::Real, eq::BurgersEquation, pg::ParticleGrid{1})
    if t <= 1e-12; return ic(pos); end
    x = pos[1]; xs = ic.mins[1]; xe = ic.maxs[1]
    ub = ic.u_box[1]; ug = ic.u_bg[1]
    
    if abs(ub - ug) < 1e-12; return ic.u_bg; end

    if ub > ug # Top-hat case
        t_int = 2.0 * (xe - xs) / (ub - ug)
        if t < t_int
            s_shock = 0.5 * (ub + ug)
            if x < xs + ug * t; return ic.u_bg
            elseif x < xs + ub * t; return State{1}((x - xs) / t)
            elseif x < xe + s_shock * t; return ic.u_box
            else return ic.u_bg
            end
        else
            C = sqrt(2.0 * (xe - xs) * (ub - ug))
            x_shock = xs + ug * t + C * sqrt(t)
            if x < xs + ug * t; return ic.u_bg
            elseif x < x_shock; return State{1}((x - xs) / t)
            else return ic.u_bg
            end
        end
    else # Well case
        t_int = 2.0 * (xe - xs) / (ug - ub)
        if t < t_int
            s_shock = 0.5 * (ug + ub)
            if x < xs + s_shock * t; return ic.u_bg
            elseif x < xe + ub * t; return ic.u_box
            elseif x < xe + ug * t; return State{1}((x - xe) / t)
            else return ic.u_bg
            end
        else
            C = sqrt(2.0 * (xe - xs) * (ug - ub))
            x_shock = xe + ug * t - C * sqrt(t)
            if x < x_shock; return ic.u_bg
            elseif x < xe + ug * t; return State{1}((x - xe) / t)
            else return ic.u_bg
            end
        end
    end
end

# --- BURGERS EQUATION 2D (Planar) ---
function (ic::Riemann)(pos::Space{2}, t::Real, eq::BurgersEquation2D, pg::ParticleGrid{2})
    if t <= 1e-12; return ic(pos); end
    
    d = dot(pos - ic.p0, ic.n)
    n_sum = ic.n[1] + ic.n[2]
    uL = ic.uL[1]; uR = ic.uR[1]
    
    if uL > uR # Shock
        s = 0.5 * (uL + uR) * n_sum
        return d < s * t ? ic.uL : ic.uR
    else # Rarefaction
        if d < uL * n_sum * t; return ic.uL
        elseif d > uR * n_sum * t; return ic.uR
        else 
            if abs(t * n_sum) < 1e-14; return State{1}(0.5 * (uL + uR)); end
            return State{1}(d / (t * n_sum))
        end
    end
end

# --- EULER EQUATIONS 1D ---
function (ic::EulerShockTube)(pos::Space{1}, t::Real, eq::Euler1D, pg::ParticleGrid{1})
    if t <= 1e-9; return ic(pos); end
    
    x = pos[1]; x0 = ic.p0[1]
    gamma = GAS_GAMMA_EULER
    rho_L, m_L, E_L = ic.uL
    rho_R, m_R, E_R = ic.uR
    
    u_L = m_L / rho_L; p_L = (gamma - 1.0) * (E_L - 0.5 * rho_L * u_L^2)
    u_R = m_R / rho_R; p_R = (gamma - 1.0) * (E_R - 0.5 * rho_R * u_R^2)
    
    c_L = sqrt(gamma * p_L / rho_L)
    c_R = sqrt(gamma * p_R / rho_R)
    
    function pressure_func(p_star_guess::Real)
        f_L = p_star_guess > p_L ? (p_star_guess - p_L) * sqrt((2.0 / ((gamma + 1.0) * rho_L)) / (p_star_guess + p_L * (gamma - 1.0) / (gamma + 1.0))) : (2.0 * c_L / (gamma - 1.0)) * ((p_star_guess / p_L)^((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        f_R = p_star_guess > p_R ? (p_star_guess - p_R) * sqrt((2.0 / ((gamma + 1.0) * rho_R)) / (p_star_guess + p_R * (gamma - 1.0) / (gamma + 1.0))) : (2.0 * c_R / (gamma - 1.0)) * ((p_star_guess / p_R)^((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        return f_L + f_R + (u_R - u_L)
    end

    p_star = 0.5 * (p_L + p_R)
    for _ in 1:100
        f_p = pressure_func(p_star)
        if abs(f_p) < 1e-9; break; end
        dfdp = (pressure_func(p_star * 1.001) - f_p) / (p_star * 0.001)
        p_star = max(1e-9, p_star - f_p / (dfdp + 1e-9))
    end

    f_L_final = p_star > p_L ? (p_star - p_L) * sqrt((2.0 / ((gamma + 1.0) * rho_L)) / (p_star + p_L * (gamma - 1.0) / (gamma + 1.0))) : (2.0 * c_L / (gamma - 1.0)) * ((p_star / p_L)^((gamma - 1.0) / (2.0 * gamma)) - 1.0)
    u_star = u_L - f_L_final

    s_query = (x - x0) / t
    rho_final, u_final, p_final = 0.0, 0.0, 0.0

    if s_query <= u_star # Left of contact
        if p_star > p_L # Left Shock
            S_L = u_L - c_L * sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / p_L) + (gamma - 1.0) / (2.0 * gamma))
            rho_star_L = rho_L * ((p_star / p_L) + (gamma - 1.0) / (gamma + 1.0)) / (1.0 + (p_star / p_L) * (gamma - 1.0) / (gamma + 1.0))
            rho_final, u_final, p_final = s_query <= S_L ? (rho_L, u_L, p_L) : (rho_star_L, u_star, p_star)
        else # Left Rarefaction
            S_head_L = u_L - c_L
            S_tail_L = u_star - c_L * (p_star / p_L)^((gamma - 1.0) / (2.0 * gamma))
            if s_query <= S_head_L; rho_final, u_final, p_final = rho_L, u_L, p_L
            elseif s_query >= S_tail_L; rho_final, u_final, p_final = rho_L * (p_star / p_L)^(1.0 / gamma), u_star, p_star
            else
                u_final = (2.0 / (gamma + 1.0)) * (c_L + (gamma - 1.0) / 2.0 * u_L + s_query)
                c_final = c_L - (gamma - 1.0) / 2.0 * (u_final - u_L)
                rho_final = rho_L * (c_final / c_L)^(2.0 / (gamma - 1.0))
                p_final = p_L * (rho_final / rho_L)^gamma
            end
        end
    else # Right of contact
        if p_star > p_R # Right Shock
            S_R = u_R + c_R * sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / p_R) + (gamma - 1.0) / (2.0 * gamma))
            rho_star_R = rho_R * ((p_star / p_R) + (gamma - 1.0) / (gamma + 1.0)) / (1.0 + (p_star / p_R) * (gamma - 1.0) / (gamma + 1.0))
            rho_final, u_final, p_final = s_query >= S_R ? (rho_R, u_R, p_R) : (rho_star_R, u_star, p_star)
        else # Right Rarefaction
            S_head_R = u_R + c_R
            S_tail_R = u_star + c_R * (p_star / p_R)^((gamma - 1.0) / (2.0 * gamma))
            if s_query >= S_head_R; rho_final, u_final, p_final = rho_R, u_R, p_R
            elseif s_query <= S_tail_R; rho_final, u_final, p_final = rho_R * (p_star / p_R)^(1.0 / gamma), u_star, p_star
            else
                u_final = (2.0 / (gamma + 1.0)) * (-c_R + (gamma - 1.0) / 2.0 * u_R + s_query)
                c_final = c_R + (gamma - 1.0) / 2.0 * (u_final - u_R)
                rho_final = rho_R * (c_final / c_R)^(2.0 / (gamma - 1.0))
                p_final = p_R * (rho_final / rho_R)^gamma
            end
        end
    end

    return SVector{3, Float64}(rho_final, rho_final * u_final, p_final / (gamma - 1.0) + 0.5 * rho_final * u_final^2)
end

# =========================================================================
# DISCONTINUITY TRACKING (1D QuadGK Integration Constraints)
# =========================================================================

# Default fallback
get_discontinuity_points(ic::InitialCondition, eq::HyperbolicPDE{D, M}, t::Real, pg::ParticleGrid{D}) where {D, M} = Float64[]

# --- Linear Advection ---
function get_discontinuity_points(ic::Union{Box, Riemann, SRiemann, EulerShockTube}, eq::LinearAdvection{M, 1}, t::Real, pg::ParticleGrid{1}) where M
    vel = eq.vel[1][1]
    domain_length = pg.meta.maxs[1] - pg.meta.mins[1]
    
    initial_points = ic isa Box ? [ic.mins[1], ic.maxs[1]] : [ic.p0[1]]
    points = Float64[]
    
    for pt in initial_points
        advected_pos = pt + vel * t
        if pg.meta.bc == :periodic
            advected_pos = pg.meta.mins[1] + mod(advected_pos - pg.meta.mins[1], domain_length)
        end
        push!(points, advected_pos)
    end
    return unique(sort(points))
end

# --- Burgers Equation ---
function get_discontinuity_points(ic::Riemann, eq::BurgersEquation, t::Real, pg::ParticleGrid{1})
    uL = ic.uL[1]; uR = ic.uR[1]; x0 = ic.p0[1]
    return uL > uR ? [x0 + 0.5 * (uL + uR) * t] : [x0 + uL * t, x0 + uR * t]
end

function get_discontinuity_points(ic::SRiemann, eq::BurgersEquation, t::Real, pg::ParticleGrid{1})
    return get_discontinuity_points(Riemann(ic.uL, ic.uR, ic.p0, ic.n), eq, t, pg)
end

function get_discontinuity_points(ic::Box, eq::BurgersEquation, t::Real, pg::ParticleGrid{1})
    xs = ic.mins[1]; xe = ic.maxs[1]
    ub = ic.u_box[1]; ug = ic.u_bg[1]
    s_shock = 0.5 * (ub + ug)
    
    if ub > ug # Top-hat
        res = [xs + ug * t]
        rare_pos = xs + ub * t
        shock_pos = xe + s_shock * t
        if rare_pos < shock_pos; push!(res, rare_pos); end
        push!(res, shock_pos)
        return res
    else # Well
        return [xs + s_shock * t, xe + ub * t, xe + ug * t]
    end
end

function get_discontinuity_points(ic::Sine, eq::BurgersEquation, t::Real, pg::ParticleGrid{1})
    x_break = ic.a > 0 ? ic.period[1] / 2.0 : 0.0
    u_at_break = ic(Space{1}(x_break))[1]
    return [x_break + u_at_break * t]
end

function get_discontinuity_points(ic::Gauss, eq::BurgersEquation, t::Real, pg::ParticleGrid{1})
    x_break = ic.a > 0 ? ic.b[1] + ic.width / sqrt(2.0) : ic.b[1] - ic.width / sqrt(2.0)
    u_at_break = ic(Space{1}(x_break))[1]
    return [x_break + u_at_break * t]
end

# --- Euler Equations ---
function get_discontinuity_points(ic::EulerShockTube, eq::Euler1D, t::Real, pg::ParticleGrid{1})
    gamma = GAS_GAMMA_EULER
    rho_L, m_L, E_L = ic.uL
    rho_R, m_R, E_R = ic.uR
    x0 = ic.p0[1]
    
    p_L = (gamma - 1.0) * (E_L - 0.5 * m_L^2 / rho_L)
    p_R = (gamma - 1.0) * (E_R - 0.5 * m_R^2 / rho_R)
    c_L = sqrt(gamma * p_L / rho_L)
    c_R = sqrt(gamma * p_R / rho_R)
    
    # Run the exact same Star-Region iterative solver to find wave speeds
    function pressure_func(p_star_guess::Real)
        f_L = p_star_guess > p_L ? (p_star_guess - p_L) * sqrt((2.0 / ((gamma + 1.0) * rho_L)) / (p_star_guess + p_L * (gamma - 1.0) / (gamma + 1.0))) : (2.0 * c_L / (gamma - 1.0)) * ((p_star_guess / p_L)^((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        f_R = p_star_guess > p_R ? (p_star_guess - p_R) * sqrt((2.0 / ((gamma + 1.0) * rho_R)) / (p_star_guess + p_R * (gamma - 1.0) / (gamma + 1.0))) : (2.0 * c_R / (gamma - 1.0)) * ((p_star_guess / p_R)^((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        return f_L + f_R + (m_R/rho_R - m_L/rho_L)
    end

    p_star = 0.5 * (p_L + p_R)
    for _ in 1:100 
        f_p = pressure_func(p_star)
        if abs(f_p) < 1e-9; break; end
        p_star = max(1e-9, p_star - f_p / ((pressure_func(p_star * 1.001) - f_p) / (p_star * 0.001) + 1e-9))
    end

    f_L_final = p_star > p_L ? (p_star - p_L) * sqrt((2.0 / ((gamma + 1.0) * rho_L)) / (p_star + p_L * (gamma - 1.0) / (gamma + 1.0))) : (2.0 * c_L / (gamma - 1.0)) * ((p_star / p_L)^((gamma - 1.0) / (2.0 * gamma)) - 1.0)
    u_star = (m_L/rho_L) - f_L_final

    points = Float64[x0 + u_star * t] # Contact wave
    
    if p_star > p_L # Left Shock
        push!(points, x0 + ((m_L/rho_L) - c_L * sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / p_L) + (gamma - 1.0) / (2.0 * gamma))) * t)
    else # Left Rarefaction
        push!(points, x0 + ((m_L/rho_L) - c_L) * t, x0 + (u_star - c_L * (p_star / p_L)^((gamma - 1.0) / (2.0 * gamma))) * t)
    end

    if p_star > p_R # Right Shock
        push!(points, x0 + ((m_R/rho_R) + c_R * sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / p_R) + (gamma - 1.0) / (2.0 * gamma))) * t)
    else # Right Rarefaction
        push!(points, x0 + ((m_R/rho_R) + c_R) * t, x0 + (u_star + c_R * (p_star / p_R)^((gamma - 1.0) / (2.0 * gamma))) * t)
    end
    
    return unique(sort(points))
end