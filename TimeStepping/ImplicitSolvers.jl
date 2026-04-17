

function PicardIterationSolver(max_components::Int = 100; max_iters::Int = 20, tol::Float64 = 1e-8)
    n_threads = Threads.nthreads()
    S_buffers = [zeros(Float64, max_components) for _ in 1:n_threads]
    Y_buffers = [zeros(Float64, max_components) for _ in 1:n_threads]
    return PicardIterationSolver(max_iters, tol, S_buffers, Y_buffers)
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
    if N_components == 0 && length(Y_out_particle) == 0; return true; end

    # Fetch thread-local buffers natively
    tid = mod1(Threads.threadid(), Threads.nthreads())
    S_eval_local = solver.S_buffers[tid]
    Y_prev_iter  = solver.Y_buffers[tid]
    
    converged = false
    norm_diff::Float64 = Inf 
    
    for iter in 1:solver.max_iters
        # Native array copying for the slice
        for k in 1:N_components; Y_prev_iter[k] = Y_out_particle[k]; end
        
        source_term_object(S_eval_local, Y_out_particle, particle_pos, time_for_S_eval)
        
        norm_diff = 0.0
        for k in 1:N_components
            Y_out_particle[k] = RHS_const_particle[k] + dt_coefficient_for_S * S_eval_local[k]
            norm_diff = math_max(norm_diff, abs(Y_out_particle[k] - Y_prev_iter[k])) # Using our branchless max!
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
@inline function solve(
    ::LinearizedRelaxationImplicitSolver,
    Y_in::State{NK},                
    dt_coeff::Float64,              
    rs::RelaxationSourceTerm{D, NM, NK},     
    p_idx::Int,
    eq::HyperbolicPDE{D},
    km::Kin2Macro{NM, NK}      
) where {D, NM, NK}
    
    dt_over_eps = dt_coeff * rs.inv_epsilon
    denom = 1.0 / (1.0 + dt_over_eps)

    u_macro = km(Y_in)
    flux_vals = flux(eq, u_macro)

    # Generate the SVector entirely in the CPU registers
    return State{NK}(ntuple(Val(NK)) do k
        v_k_base = Y_in[k]
        m_idx = km(k)
        
        f_dot_inv_lambda = flux_dot(flux_vals, m_idx, rs.scaled_inv_speeds[k])
        Mk_val = rs.coefficients[m_idx] * (u_macro[m_idx] + f_dot_inv_lambda)
        
        (v_k_base + dt_over_eps * Mk_val) * denom
    end)
end

# Solve for NON-LOCAL Relaxation Source Term
@inline function solve(
    ::LinearizedRelaxationImplicitSolver,
    V_in::State{NK},       
    dt_coeff::Float64,              
    st::NonLocalRelaxationSourceTerm{D, NM, NK},     
    p_idx::Int, 
    eq::HyperbolicPDE{D},
    km::Kin2Macro{NM, NK}
) where {D, NM, NK}
    
    dt_over_eps = dt_coeff * st.inv_epsilon
    denom = 1.0 / (1.0 + dt_over_eps)
    
    u_macro = km(V_in)
    
    return State{NK}(ntuple(Val(NK)) do k
        v_star = V_in[k]
        m_idx = km(k)
        
        T_val = st.T_potential[p_idx, m_idx]
        T_dot_inv_lambda = T_val * st.scaled_inv_speeds[k][1]

        Mk_val = st.coefficients[m_idx] * (u_macro[m_idx] + T_dot_inv_lambda)
        
        (v_star + dt_over_eps * Mk_val) * denom
    end)
end