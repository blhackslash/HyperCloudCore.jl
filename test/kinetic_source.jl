"""
    Kin2Macro{NM, NK}

A mapping structure bridging kinetic (`NK`) and macroscopic (`NM`) state components.
"""
struct Kin2Macro{NM, NK}
    ranges::NTuple{NM, UnitRange{Int}}
    k_to_m::NTuple{NK, Int}
end

function Kin2Macro(edges::Union{Vector{Int},Tuple})
    NM = length(edges) - 1
    NK = edges[end] - 1
    ranges = ntuple(i -> edges[i]:(edges[i+1]-1), NM)
    
    k_to_m_array = zeros(Int, NK)
    for m in 1:NM
        for k in ranges[m]
            k_to_m_array[k] = m
        end
    end
    
    return Kin2Macro{NM,NK}(ranges, Tuple(k_to_m_array))
end

@inline (km::Kin2Macro{NM,NK})(v::AbstractVector) where {NM,NK} = State{NM}(ntuple(i -> sum(v[k] for k in km.ranges[i]), Val(NM)))
@inline (km::Kin2Macro{NM,NK})(k::Int) where {NM,NK} = km.k_to_m[k]

@inline function flux_dot(F::Flux{D, NM, T}, m_idx::Int, scaled_inv_speed::Space{D, T}) where {D, NM, T}
    return sum(ntuple(d -> F[d][m_idx] * scaled_inv_speed[d], Val(D)))
end

# =========================================================================
# LOCAL RELAXATION SOURCE TERM
# =========================================================================

struct RelaxationSourceTerm{D, NM, NK, T, MEQ <: HyperbolicPDE} <: AbstractImplicitSourceTerm
    km::Kin2Macro{NM, NK}
    macro_eq::MEQ
    inv_epsilon::T
    coefficients::State{NM, T}
    scaled_inv_speeds::SVector{NK, Space{D, T}}
end

function RelaxationSourceTerm(
    km::Kin2Macro{NM, NK}, 
    eps::T, 
    coeffs::State{NM, T}, 
    macro_eq::HyperbolicPDE{D, NM, T},
    eq_kin::HyperbolicPDE{D, NK, T}, 
    interior_factor::T = T(D)
) where {D, NM, NK, T}
    
    # We now extract speeds using the generic kinetic_wave_speed API instead of hardcoded .vel
    scaled_inv_speeds = ntuple(Val(NK)) do k
        Space{D, T}(ntuple(Val(D)) do d
            v = kinetic_wave_speed(eq_kin, d, k)
            abs(v) > T(1e-14) ? interior_factor / v : zero(T)
        end)
    end
    
    return RelaxationSourceTerm(
        km, macro_eq, one(T) / eps, coeffs, SVector{NK, Space{D, T}}(scaled_inv_speeds)
    )
end

@inline function evaluate_source(rs::RelaxationSourceTerm{D, NM, NK, T}, U_kinetic::State{NK, T}, p_idx::Int, pg::ParticleGrid, t::Real) where {D, NM, NK, T}
    u_macro = rs.km(U_kinetic)
    flux_vals = flux(rs.macro_eq, u_macro)
    
    return State{NK, T}(ntuple(Val(NK)) do k
        m_idx = rs.km(k)
        f_dot_inv_lambda = flux_dot(flux_vals, m_idx, rs.scaled_inv_speeds[k])
        
        Mk = rs.coefficients[m_idx] * (u_macro[m_idx] + f_dot_inv_lambda)
        (Mk - U_kinetic[k]) * rs.inv_epsilon
    end)
end

# =========================================================================
# NON-LOCAL RELAXATION SOURCE TERM
# =========================================================================

mutable struct NonLocalRelaxationSourceTerm{D, NM, NK, T, MEQ <: HyperbolicPDE} <: AbstractImplicitSourceTerm
    km::Kin2Macro{NM, NK}
    macro_eq::MEQ
    inv_epsilon::T
    coefficients::State{NM, T}
    scaled_inv_speeds::SVector{NK, Space{D, T}}
    t_potential::Matrix{T}
end

function NonLocalRelaxationSourceTerm(
    km::Kin2Macro{NM, NK}, 
    eps::T, 
    coeffs::State{NM, T}, 
    macro_eq::HyperbolicPDE{D, NM, T},
    eq_kin::HyperbolicPDE{D, NK, T}, 
    interior_factor::T = T(D)
) where {D, NM, NK, T}
    
    scaled_inv_speeds = ntuple(Val(NK)) do k
        Space{D, T}(ntuple(Val(D)) do d
            v = kinetic_wave_speed(eq_kin, d, k)
            abs(v) > T(1e-14) ? interior_factor / v : zero(T)
        end)
    end
    
    t_potential = Matrix{T}(undef, 0, NM)
    return NonLocalRelaxationSourceTerm(
        km, macro_eq, one(T) / eps, coeffs, SVector{NK, Space{D, T}}(scaled_inv_speeds), t_potential
    )
end

function ensure_buffer_size!(st::NonLocalRelaxationSourceTerm{D, NM, NK, T}, N_particles::Int) where {D, NM, NK, T}
    if size(st.t_potential, 1) != N_particles
        st.t_potential = Matrix{T}(undef, N_particles, NM)
    end
end

function pre_solve_update!(st::NonLocalRelaxationSourceTerm{D, NM, NK, T}, stage_data::AbstractVector{State{NK, T}}, pg::ParticleGrid, t::Real) where {D, NM, NK, T}
    N_particles = pg.meta.N
    ensure_buffer_size!(st, N_particles)
    
    km = st.km
    macro_eq = st.macro_eq
    
    # 1. Compute jump potentials iteratively along the array
    Threads.@threads for i in 2:N_particles
        v_L = stage_data[i-1]
        v_R = stage_data[i]    
        
        u_L = km(v_L)
        u_R = km(v_R)
        
        jump = path_integral(macro_eq, u_L, u_R)
        
        for m in 1:NM
            st.t_potential[i, m] = jump[m]
        end
    end
    
    # Base condition for particle 1
    for m in 1:NM; st.t_potential[1, m] = zero(T); end
    
    # 2. Cumulative summation to build the topological field
    for i in 2:N_particles
        for m in 1:NM
            st.t_potential[i, m] += st.t_potential[i-1, m]
        end
    end
    return nothing
end

@inline function evaluate_source(st::NonLocalRelaxationSourceTerm{D, NM, NK, T}, V_kin::State{NK, T}, p_idx::Int, pg::ParticleGrid, t::Real) where {D, NM, NK, T}
    u_macro = st.km(V_kin)
    
    return State{NK, T}(ntuple(Val(NK)) do k
        m_idx = st.km(k)
        T_val = st.t_potential[p_idx, m_idx]
        
        T_dot_inv_lambda = T_val * st.scaled_inv_speeds[k][1]
        Mk_val = st.coefficients[m_idx] * (u_macro[m_idx] + T_dot_inv_lambda)
        
        (Mk_val - V_kin[k]) * st.inv_epsilon
    end)
end

@inline function implicit_solve(
    rs::RelaxationSourceTerm{D, NM, NK, T}, 
    Y_in::State{NK, T},                
    dt_coeff::Real,              
    p_idx::Int,
    pg::ParticleGrid,
    t::Real      
) where {D, NM, NK, T}
    
    dt_over_eps = T(dt_coeff) * rs.inv_epsilon
    denom = one(T) / (one(T) + dt_over_eps)

    u_macro = rs.km(Y_in)
    flux_vals = flux(rs.macro_eq, u_macro)

    return State{NK, T}(ntuple(Val(NK)) do k
        v_k_base = Y_in[k]
        m_idx = rs.km(k)
        
        f_dot_inv_lambda = flux_dot(flux_vals, m_idx, rs.scaled_inv_speeds[k])
        Mk_val = rs.coefficients[m_idx] * (u_macro[m_idx] + f_dot_inv_lambda)
        
        (v_k_base + dt_over_eps * Mk_val) * denom
    end)
end

# =========================================================================
# 1. SET INITIAL CONDITIONS
# =========================================================================

# --- LOCAL Relaxation Initialization ---
function set_initial_conditions!(
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
function set_initial_conditions!(
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
