export LinearizedRelaxationImplicitSolver

struct LinearizedRelaxationImplicitSolver <: AbstractImplicitSolver end

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