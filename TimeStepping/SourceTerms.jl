function Kin2Macro(edges::Union{Vector{Int},Tuple})
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
@inline function flux_dot(F::Flux{D, NM, T}, m_idx::Int, scaled_inv_speed::Space{D, T}) where {D, NM, T}
    return sum(ntuple(d -> F[d][m_idx] * scaled_inv_speed[d], Val(D)))
end

function RelaxationSourceTerm(
    km::Kin2Macro{NM, NK}, 
    eps::T, 
    coeffs::State{NM, T}, 
    eq_kin::LinearAdvection{D, NK, T}, 
    interior_factor::T = T(D)
) where {D, NM, NK, T}
    
    scaled_inv_speeds = ntuple(Val(NK)) do k
        Space{D, T}(ntuple(Val(D)) do d
            v = eq_kin.vel[d][k]
            abs(v) > T(1e-14) ? interior_factor / v : zero(T)
        end)
    end
    
    return RelaxationSourceTerm{D, NM, NK, T}(km, one(T) / eps, coeffs, SVector{NK, Space{D, T}}(scaled_inv_speeds))
end

@inline function (rs::RelaxationSourceTerm{D, NM, NK, T})(
    U_kinetic::State{NK, T},
    p_idx::Int,
    eq_macro::HyperbolicPDE{D, NM, T},
    km::Kin2Macro{NM, NK}
) where {D, NM, NK, T}
    
    u_macro = km(U_kinetic)
    flux_vals = flux(eq_macro, u_macro)
    
    return State{NK, T}(ntuple(Val(NK)) do k
        m_idx = km(k)
        f_dot_inv_lambda = flux_dot(flux_vals, m_idx, rs.scaled_inv_speeds[k])
        
        Mk = rs.coefficients[m_idx] * (u_macro[m_idx] + f_dot_inv_lambda)
        (Mk - U_kinetic[k]) * rs.inv_epsilon
    end)
end

function NonLocalRelaxationSourceTerm(
    km::Kin2Macro{NM, NK}, 
    eps::T, 
    coeffs::State{NM, T}, 
    eq_kin::LinearAdvection{D, NK, T}, 
    interior_factor::T = T(D)
) where {D, NM, NK, T}
    
    scaled_inv_speeds = ntuple(Val(NK)) do k
        Space{D, T}(ntuple(Val(D)) do d
            v = eq_kin.vel[d][k]
            abs(v) > T(1e-14) ? interior_factor / v : zero(T)
        end)
    end
    
    t_potential = Matrix{T}(undef, 0, NM)
    return NonLocalRelaxationSourceTerm{D, NM, NK, T}(km, one(T) / eps, coeffs, SVector{NK, Space{D, T}}(scaled_inv_speeds), t_potential)
end

function ensure_buffer_size!(st::NonLocalRelaxationSourceTerm{D, NM, NK, T}, N_particles::Int) where {D, NM, NK, T}
    if size(st.t_potential, 1) != N_particles
        st.t_potential = Matrix{T}(undef, N_particles, NM)
    end
end

function update_nonlocal_potential!(
    st::NonLocalRelaxationSourceTerm{D, NM, NK, T}, 
    stage_data::AbstractMatrix{T},
    pg::ParticleGrid,
    eq::HyperbolicPDE{D, NM, T}
) where {D, NM, NK, T}
    
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
            st.t_potential[i, m] = jump[m]
        end
    end
    
    for m in 1:NM; st.t_potential[1, m] = zero(T); end
    
    for i in 2:N_particles
        for m in 1:NM
            st.t_potential[i, m] += st.t_potential[i-1, m]
        end
    end
end

@inline function (st::NonLocalRelaxationSourceTerm{D, NM, NK, T})(
    V_kin::State{NK, T}, 
    p_idx::Int, 
    eq::HyperbolicPDE{D, NM, T},
    km::Kin2Macro{NM, NK}
) where {D, NM, NK, T}
    
    u_macro = km(V_kin)
    
    return State{NK, T}(ntuple(Val(NK)) do k
        m_idx = km(k)
        T_val = st.t_potential[p_idx, m_idx]
        
        T_dot_inv_lambda = T_val * st.scaled_inv_speeds[k][1]
        Mk_val = st.coefficients[m_idx] * (u_macro[m_idx] + T_dot_inv_lambda)
        
        (Mk_val - V_kin[k]) * st.inv_epsilon
    end)
end