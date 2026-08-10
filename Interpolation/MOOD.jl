@inline function evaluate_mood_and_halo!(
    main_grad, pg, rk, stage, dt, rho_stage
)
    return false
end

@inline function evaluate_mood_and_halo!(
    main_grad::MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, MOOD{S, C}, INTERPS, L, NF}, 
    pg, rk, stage, dt, rho_stage
) where {D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, S <: MOODStrategy, C <: RealMOOD, INTERPS, L, NF}
    
    mood_fun = main_grad.mood
    N = pg.meta.N
    nb_slices = pg.neighbor.ranges
    is_boundary = pg.core.is_boundary
    
    int_buffer = rk.int_buffer
    orders = main_grad.particle_orders
    mood_triggered = main_grad.mood_triggered
    needs_recalc = pg.shared.bit_buffer
    
    any_triggered = false
    fill!(mood_triggered, false)

    @batch for p_idx in 1:N
        if is_boundary[p_idx] || !needs_recalc[p_idx]; continue; end
        
        fi = rho_stage[p_idx]
        nb_slice = nb_slices[p_idx]
        div_val = rk.K_stages[stage][p_idx]

        s = length(rk.K_stages)
        base_rho = rk.rho_n[p_idx]
        A_coef = stage < s ? rk.tableau.a[stage+1, stage] : rk.tableau.b[stage]
        
        if stage < s
            for j in 1:(stage-1)
                a_val = rk.tableau.a[stage+1, j]
                if a_val != zero(T); base_rho -= dt * a_val * rk.K_stages[j][p_idx]; end
            end
        else
            for j in 1:(s-1)
                b_val = rk.tableau.b[j]
                if b_val != zero(T); base_rho -= dt * b_val * rk.K_stages[j][p_idx]; end
            end
        end
        
        rho_candidate = base_rho - dt * A_coef * div_val
        
        if mood_fun(main_grad, p_idx, fi, nb_slice, rho_candidate, pg, int_buffer.f)
            mood_triggered[p_idx] = true
        end
    end
    
    fill!(needs_recalc, false)

    for p_idx in 1:N
        if mood_triggered[p_idx] && orders[p_idx] > 1
            any_triggered = true
            orders[p_idx] -= 1
            needs_recalc[p_idx] = true 
            
            trigger_halo!(main_grad.mood.strategy, p_idx, pg, needs_recalc, orders)
        end
    end
    
    return any_triggered
end

@inline function evaluate_mood_and_halo!(
    main_grad::MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, MOOD{S, C}, INTERPS, L, NF}, 
    pg, imex_ts::GeneralIMEXTimeStepper, i, dt, current_Y_i
) where {D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, S <: MOODStrategy, C <: RealMOOD, INTERPS, L, NF}
    
    mood_fun = main_grad.mood
    N = pg.meta.N
    nb_slices = pg.neighbor.ranges
    is_boundary = pg.core.is_boundary
    
    int_buffer = imex_ts.int_buffer
    orders = main_grad.particle_orders
    mood_triggered = main_grad.mood_triggered
    needs_recalc = pg.shared.bit_buffer
    bt = imex_ts.tableau
    
    any_triggered = false
    fill!(mood_triggered, false)

    @batch for p_idx in 1:N
        if is_boundary[p_idx] || !needs_recalc[p_idx]; continue; end
        
        fi = current_Y_i[p_idx]
        nb_slice = nb_slices[p_idx]
        
        div_val = -imex_ts.K_E_stages[i][p_idx] 
        
        s = imex_ts.num_stages
        Y_local = imex_ts.rho_n[p_idx]
        
        if i < s
            for j in 1:(i-1)
                if bt.a_t[i+1, j] != zero(T)
                    Y_local += (dt * bt.a_t[i+1, j]) * imex_ts.K_E_stages[j][p_idx]
                end
                if bt.a[i+1, j] != zero(T)
                    Y_local += (dt * bt.a[i+1, j]) * imex_ts.K_I_stages[j][p_idx]
                end
            end
            Y_local += (dt * bt.a[i+1, i]) * imex_ts.K_I_stages[i][p_idx]
            Y_local += (dt * bt.a_t[i+1, i]) * (-div_val)
        else
            for j in 1:(s-1)
                if bt.b_t[j] != zero(T)
                    Y_local += (dt * bt.b_t[j]) * imex_ts.K_E_stages[j][p_idx]
                end
                if bt.b[j] != zero(T)
                    Y_local += (dt * bt.b[j]) * imex_ts.K_I_stages[j][p_idx]
                end
            end
            Y_local += (dt * bt.b[i]) * imex_ts.K_I_stages[i][p_idx]
            Y_local += (dt * bt.b_t[i]) * (-div_val)
        end
        
        if mood_fun(main_grad, p_idx, fi, nb_slice, Y_local, pg, int_buffer.f)
            mood_triggered[p_idx] = true
        end
    end
    
    fill!(needs_recalc, false)

    for p_idx in 1:N
        if mood_triggered[p_idx] && orders[p_idx] > 1
            any_triggered = true
            orders[p_idx] -= 1
            needs_recalc[p_idx] = true 
            
            trigger_halo!(main_grad.mood.strategy, p_idx, pg, needs_recalc, orders)
        end
    end
    
    return any_triggered
end

# --- Effective Order Evaluators & Halo Triggers ---
# (Remaining logic unchanged except for types if applicable)
@inline get_effective_order(::EPD1, orders, i, nb_slice, nb_indices) = orders[i]
@inline function get_effective_order(::EPD2, orders, i, nb_slice, nb_indices)
    eff = orders[i]
    @inbounds for k in nb_slice
        @inbounds idx = nb_indices[k]
        @inbounds order = orders[idx]
        eff = min(eff, order)
    end
    return eff
end
@inline get_effective_order(::EPD0, orders, i, nb_slice, nb_indices) = orders[i]
@inline get_effective_order(::StrictEPD0, orders, i, nb_slice, nb_indices) = orders[i]

@inline trigger_halo!(::EPD0, p_idx, pg, needs_recalc, orders) = nothing

@inline function trigger_halo!(::StrictEPD0, p_idx, pg, needs_recalc, orders)
    nb_slices = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices
    is_boundary = pg.core.is_boundary
    
    @inbounds for k in nb_slices[p_idx]
        j = nb_indices[k]
        if !is_boundary[j]
            needs_recalc[j] = true
            orders[j] = min(orders[j], orders[p_idx])
            for m in nb_slices[j]
                nj = nb_indices[m]
                if !is_boundary[nj]
                    needs_recalc[nj] = true
                end
            end
        end
    end
end

@inline function trigger_halo!(::EPD1, p_idx, pg, needs_recalc, orders)
    nb_indices = pg.neighbor.indices
    @inbounds for k in pg.neighbor.ranges[p_idx]
        j = nb_indices[k]
        if !pg.core.is_boundary[j]
            needs_recalc[j] = true
        end
    end
end

@inline function trigger_halo!(::EPD2, p_idx, pg, needs_recalc,orders)
    nb_slices = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices
    is_boundary = pg.core.is_boundary
    
    @inbounds for k in nb_slices[p_idx]
        j = nb_indices[k]
        if !is_boundary[j]
            needs_recalc[j] = true
            for m in nb_slices[j]
                nj = nb_indices[m]
                if !is_boundary[nj]
                    needs_recalc[nj] = true
                end
            end
        end
    end
end

# =========================================================================
# STATE{M, T} EXTREMA FINDERS
# =========================================================================

@inline function findLocalExtrema(rho_i::State{M, T}, nb_slice::UnitRange{Int}, neighbor_fs::AbstractVector{State{M, T}}) where {M, T}
    minU = rho_i
    maxU = rho_i
    
    @inbounds for k in nb_slice 
        rho_j = neighbor_fs[k]
        minU = math_min.(minU, rho_j)
        maxU = math_max.(maxU, rho_j)
    end
    
    return minU, maxU
end

@inline function findLocalExtremaAbs(
    c_i::State{M, T}, curve_idx::Int, nb_slice::UnitRange{Int}, 
    neighbor_indices::AbstractVector{Int}, grad_vec::AbstractVector
) where {M, T}
    mini = c_i
    maxi = c_i
    
    minAbs = abs.(c_i)
    maxAbs = minAbs
    
    @inbounds for k in nb_slice
        j = neighbor_indices[k]
        c_j = grad_vec[j][curve_idx]
        
        abs_cj = abs.(c_j)
        
        mini = math_min.(mini, c_j)
        maxi = math_max.(maxi, c_j)
        minAbs = math_min.(minAbs, abs_cj)
        maxAbs = math_max.(maxAbs, abs_cj)
    end
    
    return mini, maxi, minAbs, maxAbs
end
# =========================================================================
# MOOD CRITERIA FUNCTORS
# =========================================================================

(m::MOOD{<:MOODStrategy, NoMOOD})(args...) = false
(m::MOOD{<:MOODStrategy, OnlyMOOD})(args...) = true

# --- MOODu1 (Standard DMP) ---
function (m::MOOD{<:MOODStrategy, MOODu1{T}})(
    g::Any, p_idx::Int, rho_i::State{M, T}, nb_slice::UnitRange{Int}, 
    newRho::State{M, T}, pg::ParticleGrid{D, M, T}, int_buffer_f::AbstractVector{State{M, T}}
) where {D, M, T}
    
    minU, maxU = findLocalExtrema(rho_i, nb_slice, int_buffer_f)
    δ = m.criterion.d 
    
    for m_idx in 1:M
        if abs(maxU[m_idx] - minU[m_idx]) >= δ^3
            if newRho[m_idx] < minU[m_idx] - δ || newRho[m_idx] > maxU[m_idx] + δ
                return true
            end
        end
    end
    return false
end

# --- MOODu2 (N-Dimensional MUSCL Optimization) ---
function (m::MOOD{<:MOODStrategy, MOODu2{T}})(
    g::MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, MOOD_T, INTERPS, L, NF}, p_idx::Int, rho_i::State{M, T}, nb_slice::UnitRange{Int}, 
    newRho::State{M, T}, pg::ParticleGrid{D, M, T}, int_buffer_f::AbstractVector{State{M, T}}
) where {D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, MOOD_T, INTERPS, L, NF}
    
    minU, maxU = findLocalExtrema(rho_i, nb_slice, int_buffer_f)
    δ = m.criterion.d
    
    dmp_fail = false
    for m_idx in 1:M
        if abs(maxU[m_idx] - minU[m_idx]) >= δ^3
            if newRho[m_idx] < minU[m_idx] - δ || newRho[m_idx] > maxU[m_idx] + δ
                dmp_fail = true
                break
            end
        end
    end
    
    if !dmp_fail; return false; end

    if MAX_ORDER < 3 || g.particle_orders[p_idx] < 3
        return true 
    end
    
    grad_vec = g.gradients
    neighbors = pg.neighbor.indices
    
    u2_satisfied = true
    
    for d in 1:D
        curve_idx = D + d 
        c_i = grad_vec[p_idx][curve_idx]
        
        mini, maxi, minAbs, maxAbs = findLocalExtremaAbs(c_i, curve_idx, nb_slice, neighbors, grad_vec)
        
        for m in 1:M
            ratio = maxAbs[m] < T(1e-12) ? one(T) : minAbs[m] / maxAbs[m]
            valid = (mini[m] * maxi[m] > -δ) && (ratio >= T(0.5) || maxAbs[m] < δ)
            
            if !valid
                u2_satisfied = false
                break
            end
        end
        if !u2_satisfied; break; end
    end
    
    return !u2_satisfied 
end

function (m::MOOD{<:MOODStrategy, MOODu2{T}})(
    g::Any, p_idx::Int, rho_i::State{M, T}, nb_slice::UnitRange{Int}, 
    newRho::State{M, T}, pg::ParticleGrid{D, M, T}, int_buffer_f::AbstractVector{State{M, T}}
) where {D, M, T}
    return MOOD(m.strategy, MOODu1(m.criterion.d))(g, p_idx, rho_i, nb_slice, newRho, pg, int_buffer_f)
end