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
