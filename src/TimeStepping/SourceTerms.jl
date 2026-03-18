# --- Custom Type-Stable Flux Dot Products ---
# 1D: f is a Tuple of scalars (e.g., NTuple{3, Float64})
@inline flux_dot(f::Tuple, m_idx::Int, inv_speed::SVector{1, Float64}) = f[m_idx] * inv_speed[1]

# 2D: f is a Tuple of Tuples (F, G)
@inline flux_dot(f::Tuple, m_idx::Int, inv_speed::SVector{2, Float64}) = f[1][m_idx] * inv_speed[1] + f[2][m_idx] * inv_speed[2]

# =========================================================================
# LOCAL RELAXATION SOURCE TERM
# =========================================================================

# Notice the signature now accepts `eq` and `km` from the caller!
function (rs::RelaxationSourceTerm{D, M, NK})(
    S_out_particle::AbstractVector{Float64},
    U_kinetic_particle::AbstractVector{Float64},
    p_idx::Int,
    eq::HyperbolicPDE{D},
    km::Kin2Macro{M}
) where {D, M, NK}
    
    u_macro = km(U_kinetic_particle)
    flux_vals = flux(eq, u_macro)
    
    for k in 1:NK
        m_idx = km(k)
        
        # Zero-allocation custom dot product replaces the old dimension indexing!
        f_dot_inv_lambda = flux_dot(flux_vals, m_idx, rs.inv_relax_speeds[k])
        
        Mk = rs.coefficients[m_idx] * (u_macro[m_idx] + rs.interior_factors[k] * f_dot_inv_lambda)
        S_out_particle[k] = (Mk - U_kinetic_particle[k]) * rs.inv_epsilon
    end
end

# =========================================================================
# NON-LOCAL RELAXATION SOURCE TERM
# =========================================================================

function ensure_buffer_size!(st::NonLocalRelaxationSourceTerm{D, M, NK}, N_particles::Int) where {D, M, NK}
    if size(st.T_potential, 1) != N_particles
        st.T_potential = Matrix{Float64}(undef, N_particles, M)
    end
end

function update_nonlocal_potential!(
    st::NonLocalRelaxationSourceTerm{D, M, NK}, 
    stage_data::AbstractMatrix{Float64},
    pg::ParticleGrid,
    eq::HyperbolicPDE{D}
) where {D, M, NK}
    
    N_particles = pg.meta.N
    ensure_buffer_size!(st, N_particles)
    km = pg.kin2macro
    
    Threads.@threads for i in 2:N_particles
        v_L = @view stage_data[i-1, :]
        v_R = @view stage_data[i, :]    
        
        u_L = km(v_L)
        u_R = km(v_R)
        
        jump = path_integral(eq, u_L, u_R)
        
        for m in 1:M
            st.T_potential[i, m] = jump[m]
        end
    end

    for m in 1:M; st.T_potential[1, m] = 0.0; end
    
    for i in 2:N_particles
        for m in 1:M
            st.T_potential[i, m] += st.T_potential[i-1, m]
        end
    end
end

function (st::NonLocalRelaxationSourceTerm{D, M, NK})(
    S_out::AbstractVector{Float64}, 
    V_kin::AbstractVector{Float64}, 
    p_idx::Int, 
    eq::HyperbolicPDE{D},
    km::Kin2Macro{M}
) where {D, M, NK}
    
    u_macro = km(V_kin)

    for k in 1:NK
        m_idx = km(k)
        T_val = st.T_potential[p_idx, m_idx]
        
        # In 1D, T_val is a scalar jump. Dot it with the 1D inverse speed
        T_dot_inv_lambda = T_val * st.inv_relax_speeds[k][1]

        Mk_val = st.coefficients[m_idx] * (u_macro[m_idx] + st.interior_factor * T_dot_inv_lambda)
        S_out[k] = (Mk_val - V_kin[k]) * st.inv_epsilon
    end
end