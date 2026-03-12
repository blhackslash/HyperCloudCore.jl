function PicardIterationSolver(;max_iters::Int = 20, tol::Float64 = 1e-8)
    PicardIterationSolver(max_iters, tol)
end

function solve!(
    solver::PicardIterationSolver,
    Y_out_particle::AbstractVector{Float64}, 
    RHS_const_particle::AbstractVector{Float64},
    dt_coefficient_for_S::Float64,
    source_term_object,
    particle_pos::Any,
    time_for_S_eval::Real,
    N_components::Int
)::Bool
    if N_components == 0 && length(Y_out_particle) == 0
        return true 
    end

    # Use stack allocation for local S evaluation
    S_eval_local = MVector{N_components, Float64}(undef)
    Y_prev_iter = MVector{N_components, Float64}(undef)
    
    converged = false
    norm_diff::Float64 = Inf 
    
    for iter in 1:solver.max_iters
        Y_prev_iter .= Y_out_particle
        source_term_object(S_eval_local, Y_out_particle, particle_pos, time_for_S_eval)
        
        for k_comp in 1:N_components
            Y_out_particle[k_comp] = RHS_const_particle[k_comp] + dt_coefficient_for_S * S_eval_local[k_comp]
        end
        
        norm_diff = 0.0
        for k_comp in 1:N_components
            norm_diff = max(norm_diff, abs(Y_out_particle[k_comp] - Y_prev_iter[k_comp]))
        end

        if norm_diff < solver.tol
            converged = true
            break
        end
    end

    if !converged
        @warn "PicardIterationSolver did not converge at pos $particle_pos, time $time_for_S_eval. Max Diff: $norm_diff"
    end
    return converged
end

# Solve for LOCAL Relaxation Source Term
function solve!(
    ::LinearizedRelaxationImplicitSolver,
    Y_out_particle::AbstractVector{Float64},         
    dt_coeff::Float64,              
    rs::RelaxationSourceTerm{D, N, NK, PDE},     
    p_idx::Int,
    particle_pos::Any,           
    time_for_S_eval::Real,                      
    args...      
)::Bool where {D, N, NK, PDE}
    
    epsilon = rs.epsilon
    coeff_sum_inv = 1.0 / (epsilon + dt_coeff)

    # 1. Reconstruct Macro State
    u_macro = rs.kin2macro(Y_out_particle)
    
    # 2. Evaluate physical flux ONCE
    flux_vals = flux(rs.system_eq, u_macro)

    # 3. Update kinetic components implicitly
    for k in 1:NK
        v_k_base_kinetic = Y_out_particle[k]
        
        m_idx = rs.kin2macro(k)
        dim = rs.dimensions[k]
        
        f_val = get_flux_component(flux_vals, m_idx, dim, Val(D))
        
        # Inline Maxwellian
        Mk_val = rs.coefficients[k] * (u_macro[m_idx] + rs.interior_factors[k] * f_val / rs.relax_speeds[k])
        
        Y_out_particle[k] = (epsilon * v_k_base_kinetic + dt_coeff * Mk_val) * coeff_sum_inv
    end
    
    return true 
end

# Solve for NON-LOCAL Relaxation Source Term
function solve!(
    ::LinearizedRelaxationImplicitSolver,
    V_out::AbstractVector{Float64},       
    dt_coeff::Float64,              
    st::NonLocalRelaxationSourceTerm{D, N, NK, PDE},     
    p_idx::Int, 
    args...
)::Bool where {D, N, NK, PDE}
    
    epsilon = st.epsilon
    coeff_sum_inv = 1.0 / (epsilon + dt_coeff)
    
    # Kin2Macro handles the slice directly now
    u_macro = st.kin2macro(V_out)
    
    for k in 1:NK
        v_star = V_out[k]
        m_idx = st.kin2macro(k)
        
        T_val = st.T_potential[p_idx, m_idx]
        Mk_val = st.coefficients[m_idx] * (u_macro[m_idx] + st.interior_factor * T_val / st.relax_speeds[k])
        
        V_out[k] = (epsilon * v_star + dt_coeff * Mk_val) * coeff_sum_inv
    end
    
    return true 
end