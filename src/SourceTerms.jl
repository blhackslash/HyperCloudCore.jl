# ============== Suggested new file: SourceTerms.jl ==============
module SourceTerms

using ..HyperbolicPDEs
using ..ParticleGrids

export AbstractSourceTerm, RelaxationSourceTerm1D, RelaxationSourceTerm, MaxwellianFunctor, update_nonlocal_potential!, NonLocalRelaxationSourceTerm, Kin2Macro
export ensure_buffer_size!
abstract type AbstractSourceTerm end

"""
    Kin2Macro{NM, NK, KI}

A simplified mapper that stores the starting kinetic index for each of the NM macro variables.
NM: Number of Macroscopic variables.
NK: Total number of Kinetic components.
"""
struct Kin2Macro{N}
    ranges::NTuple{N,UnitRange{Int}}
end

# Helper to support iteration/indexing like a Tuple (backward compatibility)
Base.getindex(km::Kin2Macro, i::Int) = km.ranges[i]
Base.length(km::Kin2Macro) = length(km.ranges)
Base.iterate(km::Kin2Macro, state=1) = iterate(km.ranges, state)
Base.eachindex(km::Kin2Macro) = eachindex(km.ranges)

# Constructor from the original nested list logic
function Kin2Macro(edges::Union{AbstractVector{Int},Tuple})
    N = length(edges) - 1
    ranges = ntuple(i -> edges[i]:(edges[i+1]-1), N)
    # Extract only the first element of each group as the 'start' index
    return Kin2Macro{N}(ranges)
end

# Functor 1: Reconstruct Macro Tuple (v_kinetic -> u_macro)
@inline function (km::Kin2Macro{N})(v::Union{Tuple,AbstractVector}) where {N}
    ntuple(i -> sum(v[km.ranges[i]]),Val(N))
    # return ntuple(Val(NM)) do i
    #     val = 0.0
    #     for k in km.range[i]
    #         val += v[k]
    #     end
    #     val
    # end
end

# Functor 2: Overloaded Mapping (k_index -> m_index)
# Returns the macroscopic index 'm' that owns kinetic component 'k'
@inline function (km::Kin2Macro{N})(k::Int) where {N}
    for (i,range) = enumerate(km.ranges)
        if k in range; return i end
    end
    @warn "Could not match given kinetic index to macro variable!"
end

# Helper for 1D PDEs: flux_result is a single tuple
@inline function get_flux_component(flux_result::Tuple, i_macro::Int, i_dim::Int, ::Val{1})
    return flux_result[i_macro]
end

# Helper for 2D PDEs: flux_result is a tuple of tuples
@inline function get_flux_component(flux_result::Tuple, i_macro::Int, i_dim::Int, ::Val{2})
    return flux_result[i_dim][i_macro]
end

# Helper for 1D PDEs: flux_result is a single tuple
@inline function get_flux_component(flux_result::Float64, i_dim::Int, ::Val{1})
    return flux_result
end

# Helper for 2D PDEs: flux_result is a tuple of tuples
@inline function get_flux_component(flux_result::Tuple, i_dim::Int, ::Val{2})
    return flux_result[i_dim]
end

# F is a parameter for the concrete type of the flux function.
struct MaxwellianFunctor{D,N,E <: HyperbolicPDE{D, N}}
    system_eq::E
    i_macro::Int64
    i_dim::Int64
    relax_speed::Float64
    coefficient::Float64
    interior_factor::Float64
end

# --- Method 1: Specialized for SCALAR PDEs (N=1) ---

# This method is only called if the functor's type is MaxwellianFunctor{D, 1, E}
function (m::MaxwellianFunctor{D, 1, E})(U)::Float64 where {D, E}
    # For a scalar equation, U is a 1-tuple, e.g., (rho,). We extract the value.
    u_scalar = U[1]
    
    # The flux function for a scalar PDE expects a single Float64
    flux_result = flux(m.system_eq, u_scalar)
    
    # Use the helper to handle 1D (scalar) vs 2D (tuple) flux results
    flux_val = get_flux_component(flux_result, m.i_dim, Val(D))

    # Note: For scalar relaxation, U[m.i_macro] is just u_scalar.
    return m.coefficient * (u_scalar + m.interior_factor * flux_val / m.relax_speed)
end


# --- Method 2: General for SYSTEM PDEs (any N, any D) ---

# This method is called for any MaxwellianFunctor, but because we defined a more
# specific one for N=1, this one will be used for all N > 1 cases.
function (m::MaxwellianFunctor{D, N, E})(U)::Float64 where {D, N, E}
    macro_val = U[m.i_macro]
    
    # The flux function for a system PDE expects a tuple
    flux_result = flux(m.system_eq, U)
    
    # Use the helper to handle 1D vs 2D system flux results
    flux_val = get_flux_component(flux_result, m.i_macro, m.i_dim, Val(D))

    return m.coefficient * (macro_val + m.interior_factor * flux_val / m.relax_speed)
end



"""
    (source::AbstractSourceTerm)(
        S_out_particle::AbstractVector{Float64}, # Output vector S(U)
        U_particle::AbstractVector{Float64},     # Input state vector U
        particle_pos::Float64,
        time::Real
    )

Functor interface for source terms. Modifies `S_out_particle` in place.
"""
function (source::AbstractSourceTerm)(
    S_out_particle::AbstractVector{Float64},
    U_particle::AbstractVector{Float64},
    particle_pos::Any,
    time::Any
)
    error("Functor () not implemented for source term type $(typeof(source))")
end

# --- MODIFIED Struct ---
struct RelaxationSourceTerm{MF <: Tuple, KI <: Tuple} <: AbstractSourceTerm
    maxwellians::MF
    kinetic_indices::KI
    epsilon::Float64
    inv_epsilon::Float64
    num_total_kinetic_components::Int64
    num_macro_variables::Int64
    # Buffer is now a list of buffers, one per thread
    thread_macro_buffers::Vector{Vector{Float64}}
end

# --- MODIFIED Constructor ---
function RelaxationSourceTerm(
    maxwellian_functions_input::AbstractVector{<:MaxwellianFunctor},
    epsilon::Float64,
    kinetic_indices_input::AbstractVector{<:AbstractVector{Int}}
)   
    maxwellian_tuple = Tuple(maxwellian_functions_input)
    kinetic_indices_tuple = Tuple(Tuple(indices) for indices in kinetic_indices_input)
    num_total_kin = length(maxwellian_tuple)
    num_macro_vars = length(kinetic_indices_tuple)
    
    # --- Create one buffer for each thread ---
    n_threads = Threads.nthreads()
    thread_buffers = [Vector{Float64}(undef, num_macro_vars) for _ in 1:n_threads]
    
    return RelaxationSourceTerm(
        maxwellian_tuple, 
        kinetic_indices_tuple,
        epsilon,
        1/epsilon,
        num_total_kin,
        num_macro_vars,
        thread_buffers # <-- Pass the list of buffers
    )
end

# --- MODIFIED Functor ---
function (rs::RelaxationSourceTerm{MF,KI})(
    S_out_particle::AbstractVector{Float64},
    U_kinetic_particle::AbstractVector{Float64},
    p_idx,
    particle_pos, 
    time             
) where {MF <: Tuple, KI <: Tuple}
    
    if length(U_kinetic_particle) != rs.num_total_kinetic_components || length(S_out_particle) != rs.num_total_kinetic_components
        error("Dimension mismatch in RelaxationSourceTerm functor.")
    end

    # --- Get the correct buffer for this thread ---
    tid = Threads.threadid()
    # Use mod1 to handle potential dynamic changes in thread count if Julia is started with -t auto
    safe_tid = mod1(tid, length(rs.thread_macro_buffers))
    macro_buffer = rs.thread_macro_buffers[safe_tid] # <-- THREAD-SAFE
    
    # This loop now writes to a thread-local buffer
    for i = 1:rs.num_macro_variables
        # Use @inbounds for a slight speedup if you are confident
        macro_buffer[i] = sum(U_kinetic_particle[k] for k in rs.kinetic_indices[i])
    end
    
    # This loop now reads from a thread-local buffer
    for (k_global_comp, maxwellian) in enumerate(rs.maxwellians)
        mk_of_U_macro = maxwellian(macro_buffer)
        
        S_out_particle[k_global_comp] = (mk_of_U_macro - U_kinetic_particle[k_global_comp]) * rs.inv_epsilon
    end
end

# In SourceTerms.jl

mutable struct NonLocalRelaxationSourceTerm{D, N, NK, PDE <: HyperbolicPDE{D, N}} <: AbstractSourceTerm
    system_eq::PDE
    epsilon::Float64
    inv_epsilon::Float64
    
    kin2macro::Kin2Macro{N}

    # Stored Parameters (length NK)
    coefficients::NTuple{N, Float64}
    relax_speeds::NTuple{NK, Float64}
    interior_factor::Float64

    # Potential buffer T_j: Size (N_particles, N) [cite: 240, 241, 249]
    T_potential::Matrix{Float64}
    num_total_kinetic_components::Int

    function NonLocalRelaxationSourceTerm(
        eq::PDE, 
        epsilon::Float64, 
        km::Kin2Macro{N},
        coeffs::NTuple{N, Float64},
        speeds::NTuple{NK, Float64},
        int_factor::Float64
    ) where {D, N, NK, PDE <: HyperbolicPDE{D, N}}
        new{D, N, NK, PDE}(
            eq, epsilon, 1.0/epsilon, km,
            coeffs, speeds, int_factor,
            Matrix{Float64}(undef, 0, 0), NK
        )
    end
end

"""
Equation (38) Implementation: T_j = Σ_{i ≤ j} ∫ A(Φ) ∂sΦ ds.
Calculates jumps in parallel using Kin2Macro and accumulates in serial[cite: 249, 252].
"""
function update_nonlocal_potential!(
    st::NonLocalRelaxationSourceTerm{D,N,NK,PDE}, 
    system_pg::ParticleGridSystem
) where {D,N,NK,PDE<:HyperbolicPDESystem{D,N}}
    N_particles = system_pg[1].N
    ensure_buffer_size!(st, N_particles)
    
    # Phase 1: Parallel jump calculation using ntuples (No Allocations)
    for i in 2:N_particles
        # 1. Reconstruct kinetic state vectors as views or temporary arrays
        # Note: system_pg[k].rhos holds the k-th kinetic component [cite: 939, 1043]
        v_L = ntuple(k -> system_pg[k].rhos[i-1], Val(NK))
        v_R = ntuple(k -> system_pg[k].rhos[i],   Val(NK))

        # 2. Map Kinetic -> Macro using our new struct
        u_L = st.kin2macro(v_L)
        u_R = st.kin2macro(v_R)
        
        # 3. Path integral ∫ A(Φ(uL, uR)) ds [cite: 801, 803]
        jump = path_integral(st.system_eq, u_L, u_R)
        
        for k in 1:N
            st.T_potential[i, k] = jump[k]
        end
    end
    
    # Phase 2: Serial accumulation to form the non-local potential T_j [cite: 281]
    for k in 1:N
        st.T_potential[1, k] = 0.0 
    end

    for i in 2:N_particles
        for k in 1:N
            st.T_potential[i, k] += st.T_potential[i-1, k]
        end
    end
end

# Overload to update potential from a Matrix (current stage values)
function update_nonlocal_potential!(
    st::NonLocalRelaxationSourceTerm{D,N,NK,PDE}, 
    stage_data::AbstractMatrix{Float64},
    system_pg::ParticleGridSystem
) where {D,N,NK,PDE<:HyperbolicPDESystem{D,N}}
    N_particles = system_pg[1].N
    ensure_buffer_size!(st, N_particles)
    
    for i in 2:N_particles
        # Read from stage_data matrix instead of pgs.grids
        v_L = ntuple(k -> stage_data[i-1, k], Val(NK))
        v_R = ntuple(k -> stage_data[i, k],   Val(NK))
        println("v_R",v_R)       
        u_L = st.kin2macro(v_L)
        u_R = st.kin2macro(v_R)
        
        jump = path_integral(st.system_eq, u_L, u_R)
        for k in 1:N; st.T_potential[i, k] = jump[k]; end
    end
    # Phase 2: Serial accumulation to form the non-local potential T_j [cite: 281]
    for k in 1:N
        st.T_potential[1, k] = 0.0 
    end

    for i in 2:N_particles
        for k in 1:N
            st.T_potential[i, k] += st.T_potential[i-1, k]
        end
    end
end

"""
Ensures the T_potential buffer is correctly sized for the current particle count.
"""
function ensure_buffer_size!(st::NonLocalRelaxationSourceTerm{D,N,NK,PDE}, N_particles::Int) where {D,N,NK,PDE<:HyperbolicPDE{D,N}}
    if size(st.T_potential, 1) != N_particles
        # Resize to (N_particles, N_components)
        st.T_potential = Matrix{Float64}(undef, N_particles, N)
    end
end

# In SourceTerms.jl

function (st::NonLocalRelaxationSourceTerm{D, N, NK})(
    S_out::AbstractVector{Float64}, 
    V_kin::AbstractVector{Float64}, 
    p_idx::Int, 
    particle_pos,
    time::Any = 0.0
) where {D, N, NK}
    # V_kin is the kinetic state vector for the current particle at the current stage
    
    # 1. Reconstruct the Macroscopic State U for this particle
    # We use the Kin2Macro ranges to sum components
    u_macro = ntuple(Val(N)) do m
        val = 0.0
        for k_idx in st.kin2macro.ranges[m]
            val += V_kin[k_idx]
        end
        val
    end

    # 2. Calculate the Source Term K_I = (M - V) / epsilon
    @inbounds for k in 1:NK
        # Determine which macro variable this kinetic component belongs to
        m_idx = 1
        for i in 1:N
            if k in st.kin2macro.ranges[i]
                m_idx = i
                break
            end
        end
        
        T_val = st.T_potential[p_idx, m_idx]
        
        # The correct Maxwellian Equilibrium for non-conservative products:
        # Mk = coeff * (U_macro + factor * T / lambda)
        Mk_val = st.coefficients[k] * (u_macro[m_idx] + st.interior_factor * T_val / st.relax_speeds[k])
        
        # Compute the relaxation tendency
        S_out[k] = (Mk_val - V_kin[k]) * st.inv_epsilon
    end
end


# """
# Equation (36): Solves the implicit source term part (T_j - V) / epsilon[cite: 242, 260].
# """
# function (st::NonLocalRelaxationSourceTerm)(
#     S_out::AbstractVector{Float64},
#     V_kinetic::AbstractVector{Float64}, 
#     p_idx::Int,
#     particle_pos,
#     time::Any = 0.0
# )
#     @inbounds for k in eachindex(S_out)
#         # T_potential[p_idx, k] now holds the fully accumulated sum T_j [cite: 249]
#         S_out[k] = (st.T_potential[p_idx, k] - V_kinetic[k]) * st.inv_epsilon
#     end
# end

end # Module SourceTerms