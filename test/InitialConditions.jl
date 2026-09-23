# =========================================================================
# INITIAL CONDITION STRUCTS
# =========================================================================
abstract type InitialCondition end
abstract type SmoothInitialCondition <: InitialCondition end

struct Gauss{D, M} <: SmoothInitialCondition
    a::State{M}
    b::Space{D}
    width::Float64
end

struct Box{D, M} <: InitialCondition
    u_bg::State{M}
    u_box::State{M}
    mins::Space{D}
    maxs::Space{D}
end

struct Sine{D, M} <: SmoothInitialCondition
    a::State{M}
    period::Space{D}
    c_offset::State{M}
end

struct Riemann{D, M} <: InitialCondition
    uL::State{M}
    uR::State{M}
    p0::Space{D}
    n::Space{D}
end

struct SRiemann{D, M} <: SmoothInitialCondition
    uL::State{M}
    uR::State{M}
    p0::Space{D}
    n::Space{D}
    width::Float64
end

struct QuadrantRiemann{D, M, N_states} <: InitialCondition
    u_states::NTuple{N_states, State{M}} 
    p0::Space{D}
end

@inline function flux_dot(F::Flux{D, NM, T}, m_idx::Int, scaled_inv_speed::Space{D, T}) where {D, NM, T}
    return sum(ntuple(d -> F[d][m_idx] * scaled_inv_speed[d], Val(D)))
end

# =========================================================================
# 1. SET INITIAL CONDITIONS
# =========================================================================

function setInitialConditions!(pg::ParticleGrid{D, M}, eq::HyperbolicPDE, IC::InitialCondition) where {D, M}
    positions = pg.core.positions
    
    @inbounds for i in 1:pg.meta.N
        # Strict assignment: IC(pos) must return an State{M}
        pg.rhos[i] = IC(positions[i])
    end
    return nothing
end

# --- LOCAL Relaxation Initialization ---
function setInitialConditions!(
    pg::ParticleGrid{D, NK}, 
    st::RelaxationSourceTerm{D, NM, NK},     
    IC::InitialCondition,
    eq_macro::HyperbolicPDE{D}
) where {D, NM, NK}
    
    for p_idx in 1:pg.meta.N
        u_val = IC(pg.core.positions[p_idx]) # Macro State{NM}
        flux_vals = flux(eq_macro, u_val)    # Flux{D, NM}
        
        # Build the initial kinetic SVector component-by-component
        pg.rhos[p_idx] = State{NK}(ntuple(Val(NK)) do k
            m_idx = st.km(k)
            f_dot_inv_lambda = flux_dot(flux_vals, m_idx, st.scaled_inv_speeds[k])
            
            # Inline Maxwellian Initialization
            return st.coefficients[m_idx] * (u_val[m_idx] + f_dot_inv_lambda)
        end)
    end
    return nothing
end

# --- NON-LOCAL Relaxation Initialization ---
function setInitialConditions!(
    pg::ParticleGrid{D, NK},
    st::NonLocalRelaxationSourceTerm{D, NM, NK},
    IC::InitialCondition,
    eq_macro::HyperbolicPDE{D}
) where {D, NM, NK}
    
    N_particles = pg.meta.N
    
    # 1. Initialize grid to LOCAL equilibrium (V_k = c_m * U_m)
    for p_idx in 1:N_particles
        u_val = IC(pg.core.positions[p_idx]) 
        pg.rhos[p_idx] = State{NK}(ntuple(Val(NK)) do k
            st.coefficients[st.km(k)] * u_val[st.km(k)]
        end)
    end

    # 2. Compute the true initial potential T_0 using current grid state
    update_nonlocal_potential!(st, pg.rhos, pg, eq_macro)

    # 3. Re-initialize kinetic grids to the NON-LOCAL equilibrium: V_0 = M(U_0, T_0)
    for p_idx in 1:N_particles
        u_val = IC(pg.core.positions[p_idx])
        
        pg.rhos[p_idx] = State{NK}(ntuple(Val(NK)) do k
            m_idx = st.km(k)
            T_val = st.T_potential[p_idx, m_idx]
            T_dot_inv_lambda = T_val * st.scaled_inv_speeds[k][1]
            
            return st.coefficients[m_idx] * (u_val[m_idx] + T_dot_inv_lambda)
        end)
    end
    
    @info "Initialized Non-Local Equilibrium (Max Potential: $(maximum(abs.(st.T_potential))))"
    return nothing
end

(ic::Gauss)(pos::Space{D}) where {D} = ic.a * exp(-sum(abs2, pos - ic.b) / ic.width^2)
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
        
    else 
        error("Unknown initFunc name: $name")
    end
end