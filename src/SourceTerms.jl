module SourceTerms

using ..HyperbolicPDEs
using ..ParticleGrids
using ..CoreUtils
using StaticArrays 

export AbstractSourceTerm, RelaxationSourceTerm, update_nonlocal_potential!, NonLocalRelaxationSourceTerm, Kin2Macro
export ensure_buffer_size!

abstract type AbstractSourceTerm end

struct NoSourceTerm <: AbstractSourceTerm end

"""
    Kin2Macro{NM}

A simplified mapper storing the ranges for each of the NM macro variables.
"""
struct Kin2Macro{NM}
    ranges::NTuple{NM, UnitRange{Int}}
end

function Kin2Macro(edges::Union{AbstractVector{Int},Tuple})
    NM = length(edges) - 1
    ranges = ntuple(i -> edges[i]:(edges[i+1]-1), NM)
    return Kin2Macro{NM}(ranges)
end

# Functor 1: Reconstruct Macro Tuple (v_kinetic -> u_macro)
@inline function (km::Kin2Macro{NM})(v::AbstractVector) where {NM}
    return ntuple(i -> sum(v[k] for k in km.ranges[i]), Val(NM))
end

# Functor 2: Returns the macroscopic index 'm' that owns kinetic component 'k'
@inline function (km::Kin2Macro{NM})(k::Int) where {NM}
    for (i, range) in enumerate(km.ranges)
        if k in range
            return i 
        end
    end
    @warn "Could not match given kinetic index to macro variable!"
    return 1
end

# --- Flux Helpers ---
@inline get_flux_component(flux_result::Tuple, i_macro::Int, i_dim::Int, ::Val{1}) = flux_result[i_macro]
@inline get_flux_component(flux_result::Tuple, i_macro::Int, i_dim::Int, ::Val{2}) = flux_result[i_dim][i_macro]
@inline get_flux_component(flux_result::Float64, i_dim::Int, ::Val{1}) = flux_result
@inline get_flux_component(flux_result::Tuple, i_dim::Int, ::Val{2}) = flux_result[i_dim]


# =========================================================================
# LOCAL RELAXATION SOURCE TERM
# =========================================================================
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

function RelaxationSourceTerm(
    system_eq::PDE,
    epsilon::Float64,
    km::Kin2Macro{N},
    coeffs::NTuple{NK, Float64},
    speeds::NTuple{NK, Float64},
    int_factors::NTuple{NK, Float64},
    dims::NTuple{NK, Int}
) where {D, N, NK, PDE <: HyperbolicPDE{D, N}}
    
    return RelaxationSourceTerm{D, N, NK, PDE}(
        system_eq, epsilon, 1.0 / epsilon, km,
        coeffs, speeds, int_factors, dims,
        NK, N
    )
end

function (rs::RelaxationSourceTerm{D, N, NK, PDE})(
    S_out_particle::AbstractVector{Float64},
    U_kinetic_particle::AbstractVector{Float64},
    p_idx::Int,
    particle_pos::Any, 
    time::Real             
) where {D, N, NK, PDE}
    
    # 1. Reconstruct Macro State
    u_macro = rs.kin2macro(U_kinetic_particle)
    
    # 2. Evaluate physical flux exactly ONCE for the macro state
    flux_vals = flux(rs.system_eq, u_macro)
    
    # 3. Compute relaxation for each kinetic component
    for k in 1:NK
        m_idx = rs.kin2macro(k)
        dim = rs.dimensions[k]
        
        # Extract correct flux scalar
        f_val = get_flux_component(flux_vals, m_idx, dim, Val(D))
        
        # Inline Maxwellian Equilibrium
        Mk = rs.coefficients[k] * (u_macro[m_idx] + rs.interior_factors[k] * f_val / rs.relax_speeds[k])
        
        S_out_particle[k] = (Mk - U_kinetic_particle[k]) * rs.inv_epsilon
    end
end


# =========================================================================
# NON-LOCAL RELAXATION SOURCE TERM
# =========================================================================
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

    function NonLocalRelaxationSourceTerm(
        eq::PDE, epsilon::Float64, km::Kin2Macro{N},
        coeffs::NTuple{N, Float64}, speeds::NTuple{NK, Float64}, int_factor::Float64
    ) where {D, N, NK, PDE <: HyperbolicPDE{D, N}}
        new{D, N, NK, PDE}(
            eq, epsilon, 1.0/epsilon, km, coeffs, speeds, int_factor,
            Matrix{Float64}(undef, 0, 0), NK
        )
    end
end

function ensure_buffer_size!(st::NonLocalRelaxationSourceTerm{D,N,NK,PDE}, N_particles::Int) where {D,N,NK,PDE}
    if size(st.T_potential, 1) != N_particles
        st.T_potential = Matrix{Float64}(undef, N_particles, N)
    end
end

function update_nonlocal_potential!(
    st::NonLocalRelaxationSourceTerm{D,N,NK,PDE}, 
    stage_data::AbstractMatrix{Float64},
    pg::ParticleGrid
) where {D,N,NK,PDE}
    
    N_particles = pg.meta.N
    ensure_buffer_size!(st, N_particles)
    
    # Phase 1: Parallel jump calculation 
    Threads.@threads for i in 2:N_particles
        v_L = @view stage_data[i-1, :]
        v_R = @view stage_data[i, :]    
        
        u_L = st.kin2macro(v_L)
        u_R = st.kin2macro(v_R)
        
        jump = path_integral(st.system_eq, u_L, u_R)
        
        for k in 1:N
            st.T_potential[i, k] = jump[k]
        end
    end

    # Phase 2: Serial accumulation
    for k in 1:N
        st.T_potential[1, k] = 0.0 
    end
    
    for i in 2:N_particles
        for k in 1:N
            st.T_potential[i, k] += st.T_potential[i-1, k]
        end
    end
end

function (st::NonLocalRelaxationSourceTerm{D, N, NK})(
    S_out::AbstractVector{Float64}, 
    V_kin::AbstractVector{Float64}, 
    p_idx::Int, 
    particle_pos::Any,
    time::Any = 0.0
) where {D, N, NK}
    
    u_macro = st.kin2macro(V_kin)

    for k in 1:NK
        m_idx = st.kin2macro(k)
        T_val = st.T_potential[p_idx, m_idx]
        
        Mk_val = st.coefficients[m_idx] * (u_macro[m_idx] + st.interior_factor * T_val / st.relax_speeds[k])
        S_out[k] = (Mk_val - V_kin[k]) * st.inv_epsilon
    end
end

end # Module SourceTerms