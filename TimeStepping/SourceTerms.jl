function Kin2Macro(edges::Union{AbstractVector{Int},Tuple})
    NM = length(edges) - 1
    NK = edges[end] - 1
    ranges = ntuple(i -> edges[i]:(edges[i+1]-1), NM)
    
    # Pre-compute the inverse map for instant O(1) lookups
    k_to_m_array = zeros(Int, NK)
    for m in 1:NM
        for k in ranges[m]
            k_to_m_array[k] = m
        end
    end
    
    return Kin2Macro{NM,NK}(ranges, Tuple(k_to_m_array))
end

# Functor 1: Reconstruct Macro State natively into State{NM}
@inline (km::Kin2Macro{NM,NK})(v::AbstractVector) where {NM,NK} = State{NM}(ntuple(i -> sum(v[k] for k in km.ranges[i]), Val(NM)))

# Functor 2: Returns the macroscopic index 'm' that owns kinetic component 'k'
@inline (km::Kin2Macro{NM,NK})(k::Int) where {NM,NK} = km.k_to_m[k]


# =========================================================================
# TYPE-STABLE FLUX DOT PRODUCT
# =========================================================================

# Natively handles the Flux{D, NM} (SVector{D, State{NM}}) structure without Tuples!
@inline function flux_dot(F::Flux{D, NM}, m_idx::Int, scaled_inv_speed::Space{D}) where {D, NM}
    return sum(ntuple(d -> F[d][m_idx] * scaled_inv_speed[d], Val(D)))
end


# =========================================================================
# LOCAL RELAXATION SOURCE TERM
# =========================================================================



function RelaxationSourceTerm(
    km::Kin2Macro{NM, NK}, 
    eps::Float64, 
    coeffs::State{NM}, 
    eq_kin::LinearAdvection{D, NK}, 
    interior_factor::Float64 = Float64(D)
) where {D, NM, NK}
    
    # Pre-calculate (interior_factor / v) for every kinetic component
    scaled_inv_speeds = ntuple(Val(NK)) do k
        Space{D}(ntuple(Val(D)) do d
            v = eq_kin.vel[d][k]
            abs(v) > 1e-14 ? interior_factor / v : 0.0
        end)
    end
    
    return RelaxationSourceTerm{D, NM, NK}(km, 1.0 / eps, coeffs, SVector{NK, Space{D}}(scaled_inv_speeds))
end

function (rs::RelaxationSourceTerm{D, NM, NK})(
    S_out_particle::AbstractVector{Float64},
    U_kinetic_particle::AbstractVector{Float64},
    p_idx::Int,
    eq_macro::HyperbolicPDE{D},
    km::Kin2Macro{NM, NK}
) where {D, NM, NK}
    
    u_macro = km(U_kinetic_particle)
    flux_vals = flux(eq_macro, u_macro) # Native Flux{D, NM}
    
    @inbounds for k in 1:NK
        m_idx = km(k)
        
        # Native dot product of the m-th macro flux with the k-th scaled inverse speed
        f_dot_inv_lambda = flux_dot(flux_vals, m_idx, rs.scaled_inv_speeds[k])
        
        # Note: coefficients[k] is used instead of coefficients[m_idx] 
        # to ensure correct BGK weighting if multiple kinetic variables map to one macro variable!
        Mk = rs.coefficients[m_idx] * (u_macro[m_idx] + f_dot_inv_lambda)
        S_out_particle[k] = (Mk - U_kinetic_particle[k]) * rs.inv_epsilon
    end
end


# =========================================================================
# NON-LOCAL RELAXATION SOURCE TERM
# =========================================================================



function NonLocalRelaxationSourceTerm(
    km::Kin2Macro{NM, NK}, 
    eps::Float64, 
    coeffs::State{NM}, 
    eq_kin::LinearAdvection{D, NK}, 
    interior_factor::Float64 = Float64(D)
) where {D, NM, NK}
    
    scaled_inv_speeds = ntuple(Val(NK)) do k
        Space{D}(ntuple(Val(D)) do d
            v = eq_kin.vel[d][k]
            abs(v) > 1e-14 ? interior_factor / v : 0.0
        end)
    end
    
    T_pot = Matrix{Float64}(undef, 0, NM)
    return NonLocalRelaxationSourceTerm{D, NM, NK}(km, 1.0 / eps, coeffs, SVector{NK, Space{D}}(scaled_inv_speeds), T_pot)
end

function ensure_buffer_size!(st::NonLocalRelaxationSourceTerm{D, NM, NK}, N_particles::Int) where {D, NM, NK}
    if size(st.T_potential, 1) != N_particles
        st.T_potential = Matrix{Float64}(undef, N_particles, NM)
    end
end

function update_nonlocal_potential!(
    st::NonLocalRelaxationSourceTerm{D, NM, NK}, 
    stage_data::AbstractMatrix{Float64},
    pg::ParticleGrid,
    eq::HyperbolicPDE{D}
) where {D, NM, NK}
    
    N_particles = pg.meta.N
    ensure_buffer_size!(st, N_particles)
    km = st.km
    
    Threads.@threads for i in 2:N_particles
        v_L = @view stage_data[i-1, :]
        v_R = @view stage_data[i, :]    
        
        u_L = km(v_L)
        u_R = km(v_R)
        
        jump = path_integral(eq, u_L, u_R)
        
        for m in 1:NM
            st.T_potential[i, m] = jump[m]
        end
    end
    
    for m in 1:NM; st.T_potential[1, m] = 0.0; end
    
    for i in 2:N_particles
        for m in 1:NM
            st.T_potential[i, m] += st.T_potential[i-1, m]
        end
    end
end

function (st::NonLocalRelaxationSourceTerm{D, NM, NK})(
    S_out::AbstractVector{Float64}, 
    V_kin::AbstractVector{Float64}, 
    p_idx::Int, 
    eq::HyperbolicPDE{D},
    km::Kin2Macro{NM, NK}
) where {D, NM, NK}
    
    u_macro = km(V_kin)
    @inbounds for k in 1:NK
        m_idx = km(k)
        T_val = st.T_potential[p_idx, m_idx]
        
        # Non-local uses the path integral jump (T_val).
        # We dot it with the primary direction (usually X, d=1).
        T_dot_inv_lambda = T_val * st.scaled_inv_speeds[k][1]
        
        Mk_val = st.coefficients[m_idx] * (u_macro[m_idx] + T_dot_inv_lambda)
        S_out[k] = (Mk_val - V_kin[k]) * st.inv_epsilon
    end
end