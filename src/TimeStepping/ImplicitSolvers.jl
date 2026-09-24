export LinearizedRelaxationImplicitSolver, PicardIterationSolver
# Added <: AbstractImplicitSolver just in case you use it later!
"""
    PicardIterationSolver{T}(max_components=100; max_iters=20, tol=1e-8)

An implicit solver utilizing Picard iteration to resolve general stiff source terms.

# Fields
- `max_iters::Int`: The maximum allowed iterations before issuing a convergence warning.
- `tol::T`: The tolerance threshold used to evaluate convergence against the maximum difference between iterations.
- `s_buffers::Vector{Vector{T}}` and `y_buffers::Vector{Vector{T}}`: Pre-allocated evaluation buffers initialized per available thread to ensure thread-safe execution.
"""
struct PicardIterationSolver{T} <: AbstractImplicitSolver
    max_iters::Int
    tol::T
    s_buffers::Vector{Vector{T}}
    y_buffers::Vector{Vector{T}}
end
"""
    LinearizedRelaxationImplicitSolver()

An optimized, non-iterative implicit solver specifically designed to resolve linearized relaxation source terms analytically. 

# Details
- Computes an exact update using a scalar denominator defined as `1 / (1 + dt / eps)`.
- Dispatches custom analytical solutions for both standard `RelaxationSourceTerm` and `NonLocalRelaxationSourceTerm` configurations by utilizing the flux and macroscopic state definitions.
"""
struct LinearizedRelaxationImplicitSolver <: AbstractImplicitSolver end

function PicardIterationSolver(::Type{T}, max_components::Int = 100; max_iters::Int = 20, tol::T = T(1e-8)) where {T}
    n_threads = Threads.nthreads()
    s_buffers = [zeros(T, max_components) for _ in 1:n_threads]
    y_buffers = [zeros(T, max_components) for _ in 1:n_threads]
    return PicardIterationSolver{T}(max_iters, tol, s_buffers, y_buffers)
end

"""
    solve!(solver::PicardIterationSolver, Y_out_particle, RHS_const_particle, dt_coefficient_for_S, source_term_object, particle_pos, time_for_S_eval, N_components)

Executes the Picard iteration to implicitly solve the source term update for a single particle.

# Returns
- A `Bool` indicating whether the iteration converged within the solver's defined tolerance. If convergence fails, it logs a warning containing the position, time, and maximum observed difference.
"""
function solve!(
    solver::PicardIterationSolver{T},
    Y_out_particle::AbstractVector{T}, 
    RHS_const_particle::AbstractVector{T},
    dt_coefficient_for_S::T,
    source_term_object,
    particle_pos::Any,
    time_for_S_eval::Real,
    N_components::Int
)::Bool where {T}
    if N_components == 0 && length(Y_out_particle) == 0; return true; end

    tid = mod1(Threads.threadid(), Threads.nthreads())
    s_eval_local = solver.s_buffers[tid]
    y_prev_iter  = solver.y_buffers[tid]
    
    converged = false
    norm_diff::T = Inf 
    
    for iter in 1:solver.max_iters
        for k in 1:N_components; y_prev_iter[k] = Y_out_particle[k]; end
        
        source_term_object(s_eval_local, Y_out_particle, particle_pos, time_for_S_eval)
        
        norm_diff = zero(T)
        for k in 1:N_components
            Y_out_particle[k] = RHS_const_particle[k] + dt_coefficient_for_S * s_eval_local[k]
            norm_diff = math_max(norm_diff, abs(Y_out_particle[k] - y_prev_iter[k]))
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

@inline function solve(
    ::LinearizedRelaxationImplicitSolver,
    Y_in::State{NK, T},                
    dt_coeff::T,              
    rs::RelaxationSourceTerm{D, NM, NK, T},     
    p_idx::Int,
    eq::HyperbolicPDE{D, NM, T},
    km::Kin2Macro{NM, NK}      
) where {D, NM, NK, T}
    
    dt_over_eps = dt_coeff * rs.inv_epsilon
    denom = one(T) / (one(T) + dt_over_eps)

    u_macro = km(Y_in)
    flux_vals = flux(eq, u_macro)

    return State{NK, T}(ntuple(Val(NK)) do k
        v_k_base = Y_in[k]
        m_idx = km(k)
        
        f_dot_inv_lambda = flux_dot(flux_vals, m_idx, rs.scaled_inv_speeds[k])
        Mk_val = rs.coefficients[m_idx] * (u_macro[m_idx] + f_dot_inv_lambda)
        
        (v_k_base + dt_over_eps * Mk_val) * denom
    end)
end

@inline function solve(
    ::LinearizedRelaxationImplicitSolver,
    V_in::State{NK, T},       
    dt_coeff::T,              
    st::NonLocalRelaxationSourceTerm{D, NM, NK, T},     
    p_idx::Int, 
    eq::HyperbolicPDE{D, NM, T},
    km::Kin2Macro{NM, NK}
) where {D, NM, NK, T}
    
    dt_over_eps = dt_coeff * st.inv_epsilon
    denom = one(T) / (one(T) + dt_over_eps)
    
    u_macro = km(V_in)
    
    return State{NK, T}(ntuple(Val(NK)) do k
        v_star = V_in[k]
        m_idx = km(k)
        
        T_val = st.t_potential[p_idx, m_idx]
        T_dot_inv_lambda = T_val * st.scaled_inv_speeds[k][1]

        Mk_val = st.coefficients[m_idx] * (u_macro[m_idx] + T_dot_inv_lambda)
        
        (v_star + dt_over_eps * Mk_val) * denom
    end)
end