# ============== Suggested new file: ImplicitSolvers.jl ==============
module ImplicitSolvers

using ..SourceTerms
using ..CoreUtils

# To use AbstractSourceTerm here, ensure it's accessible:
# using ..SourceTerms # If SourceTerms.jl is at the same level or correctly pathed

# For this example, we'll use `Any` for source_term_object and rely on its functor interface
# A more type-safe approach would be: source_term_object::ST where ST <: AbstractSourceTerm

export AbstractImplicitSolver, PicardIterationSolver, LinearizedRelaxationImplicitSolver, solve!

abstract type AbstractImplicitSolver end

struct PicardIterationSolver <: AbstractImplicitSolver
    max_iters::Int
    tol::Float64

    function PicardIterationSolver(;max_iters::Int = 20, tol::Float64 = 1e-8)
        new(max_iters, tol)
    end
end

function solve!(
    solver::PicardIterationSolver,
    Y_out_particle::AbstractVector{Float64}, 
    RHS_const_particle::AbstractVector{Float64},
    dt_coefficient_for_S::Float64,
    source_term_object, # Can be AbstractSourceTerm or just a callable
    particle_pos::Float64,
    time_for_S_eval::Real,
    N_components::Int
)::Bool
    if N_components == 0 && length(Y_out_particle) == 0
        return true 
    end
    if length(Y_out_particle) != N_components || length(RHS_const_particle) != N_components
        error("Vector size mismatch in PicardIterationSolver.solve!")
    end

    S_eval_local = Vector{Float64}(undef, N_components)
    Y_prev_iter = similar(Y_out_particle)
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
        @warn "PicardIterationSolver did not converge for particle at pos $particle_pos, time $time_for_S_eval after $(solver.max_iters) iterations. Max Diff: $norm_diff"
    end
    return converged
end

# --- NEW: LinearizedRelaxationImplicitSolver ---
"""
    LinearizedRelaxationImplicitSolver <: AbstractImplicitSolver

Solves the implicit part for a relaxation source term of the form 
S_k = (M_k(rho_base) - U_k_base)/epsilon using a linearized, direct update:
U_k_new = (U_k_base + (dt'/epsilon)*M_k(rho_base)) / (1 + dt'/epsilon),
where rho_base = sum(U_k_base) from the state before this implicit step.
This is a non-iterative update for U_k once M_k(rho_base) is computed.
"""
struct LinearizedRelaxationImplicitSolver <: AbstractImplicitSolver
    # No internal fields needed if all info comes from source_term_object and arguments
    function LinearizedRelaxationImplicitSolver()
        new()
    end
end

function solve!(
    solver::LinearizedRelaxationImplicitSolver,
    Y_out_particle::AbstractVector{Float64},         
    dt_coefficient_for_S::Float64,              
    source_term_object::RelaxationSourceTerm,     
    p_idx::Int,
    particle_pos::Any,                      
    time_for_S_eval::Real,                      
    N_total_kinetic_components_arg::Int      
)::Bool
    # ... (checks) ...
    epsilon = source_term_object.epsilon
    maxwellians = source_term_object.maxwellians

    coeff_sum_inv = 1.0 / (epsilon + dt_coefficient_for_S)

    # General coupled case: reconstruct U_macro_base as a TUPLE
    kinetic_map = source_term_object.kinetic_indices
    N_macro_vars = source_term_object.num_macro_variables

    # --- Get the correct buffer for this thread ---
    tid = Threads.threadid()
    # Use mod1 to handle potential dynamic changes in thread count if Julia is started with -t auto
    safe_tid = mod1(tid, length(source_term_object.thread_macro_buffers))
    macro_buffer = source_term_object.thread_macro_buffers[safe_tid] # <-- THREAD-SAFE
    for i = 1:N_macro_vars
        macro_buffer[i] = sum(Y_out_particle[k] for k in kinetic_map[i])
    end

    for k_global_comp in 1:N_total_kinetic_components_arg
        v_k_base_kinetic = Y_out_particle[k_global_comp]
        
        # The Maxwellian now receives the TUPLE, which can be splatted efficiently
        Mk_val = maxwellians[k_global_comp](macro_buffer)
        
        Y_out_particle[k_global_comp] = (epsilon * v_k_base_kinetic + dt_coefficient_for_S * Mk_val) * coeff_sum_inv
    end
    @pebug "Implicit Solver" group=:implicit V=@view(Y_out_particle[1:NK]) u_macro=macro_buffer
    return true 
end

# In ImplicitSolvers.jl

# In ImplicitSolvers.jl

function solve!(
    ::LinearizedRelaxationImplicitSolver,
    V_out::AbstractVector{Float64},         
    dt_coeff::Float64,              
    st::NonLocalRelaxationSourceTerm{D, N, NK,  PDE},     
    p_idx::Int, 
    args...
)::Bool where {D, N, NK, PDE}
    epsilon = st.epsilon
    coeff_sum_inv = 1.0 / (epsilon + dt_coeff)
    u_macro = st.kin2macro(V_out)
    for k in 1:NK
        v_star = V_out[NK]
        m_idx = st.kin2macro(k)
        T_val = st.T_potential[p_idx, m_idx]
        # The correct equilibrium: Mk = coeff * (U + factor * T / lambda)
        Mk_val = st.coefficients[m_idx] * (u_macro[m_idx] + st.interior_factor * T_val / st.relax_speeds[k])
        
        # Standard implicit relaxation update
        V_out[k] = (epsilon * v_star + dt_coeff * Mk_val) * coeff_sum_inv
    end
    @pebug "Implicit Solver" group=:implicit V=@view(V_out[1:NK]) u_macro=u_macro
    return true 
end

end # Module ImplicitSolvers