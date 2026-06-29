@inline function evaluate_mood_and_halo!(
    main_grad, pg, rk, stage, dt, rho_stage
)
    return false
end

# Dispatch 2: Specialization for MUSCL with an active MOOD Criterion
@inline function evaluate_mood_and_halo!(
    main_grad::MUSCL{D, M, B_LEN, MAX_ORDER, DIV_ORDER, MOOD}, 
    pg, rk, stage, dt, rho_stage
) where {D, M, B_LEN, MAX_ORDER, DIV_ORDER, MOOD <: RealMOOD}
    
    mood_fun = main_grad.mood
    N = pg.meta.N
    nb_slices = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices
    is_boundary = pg.core.is_boundary
    
    int_buffer = rk.int_buffer
    orders = main_grad.particle_orders
    mood_triggered = main_grad.mood_triggered
    needs_recalc = pg.shared.bit_buffer
    
    any_triggered = false
    fill!(mood_triggered, false)

    # Step 1: Predict candidate state and evaluate MOOD
    @batch for p_idx in 1:N
        # Only evaluate MOOD if the particle just computed a new divergence
        if is_boundary[p_idx] || !needs_recalc[p_idx]; continue; end
        
        fi = rho_stage[p_idx]
        nb_slice = nb_slices[p_idx]
        div_val = rk.K_stages[stage][p_idx]

        s = length(rk.K_stages)
        base_rho = rk.rho_n[p_idx]
        A_coef = stage < s ? rk.tableau.A[stage+1, stage] : rk.tableau.b[stage]
        
        if stage < s
            for j in 1:(stage-1)
                a_val = rk.tableau.A[stage+1, j]
                if a_val != 0.0; base_rho -= dt * a_val * rk.K_stages[j][p_idx]; end
            end
        else
            for j in 1:(s-1)
                b_val = rk.tableau.b[j]
                if b_val != 0.0; base_rho -= dt * b_val * rk.K_stages[j][p_idx]; end
            end
        end
        
        rho_candidate = base_rho - dt * A_coef * div_val
        
        if mood_fun(main_grad, p_idx, fi, nb_slice, rho_candidate, pg, int_buffer.f)
            mood_triggered[p_idx] = true
        end
    end
    
    fill!(needs_recalc, false)

    # Step 2: Drop order and trigger highly localized EPD_1 Halo
    for p_idx in 1:N
        if mood_triggered[p_idx] && orders[p_idx] > 1
            any_triggered = true
            orders[p_idx] -= 1
            needs_recalc[p_idx] = true # Recompute self
            
            # EPD_1 Halo: Immediate Neighbors ONLY
            for k in nb_slices[p_idx]
                j = nb_indices[k]
                if !is_boundary[j]
                    needs_recalc[j] = true
                end
            end
        end
    end
    
    return any_triggered
end

# =========================================================================
# STATE{M} EXTREMA FINDERS
# =========================================================================

@inline function findLocalExtrema(rho_i::State{M}, nb_slice::UnitRange{Int}, neighbor_fs::AbstractVector{State{M}}) where {M}
    minU = rho_i
    maxU = rho_i
    
    @inbounds for k in nb_slice 
        rho_j = neighbor_fs[k]
        
        # Use native SVector broadcasting! 
        # This completely eliminates the closure and unrolls automatically.
        minU = math_min.(minU, rho_j)
        maxU = math_max.(maxU, rho_j)
    end
    
    return minU, maxU
end

@inline function findLocalExtremaAbs(
    c_i::State{M}, curve_idx::Int, nb_slice::UnitRange{Int}, 
    neighbor_indices::AbstractVector{Int}, grad_vec::AbstractVector
) where {M}
    mini = c_i
    maxi = c_i
    
    # Native SVector broadcasting for absolute value
    minAbs = abs.(c_i)
    maxAbs = minAbs
    
    @inbounds for k in nb_slice
        j = neighbor_indices[k]
        c_j = grad_vec[j][curve_idx]
        
        abs_cj = abs.(c_j)
        
        # Native broadcasting eliminates all closures
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

(mood::NoMOOD)(args...) = false
(mood::OnlyMOOD)(args...) = true

# --- MOODu1 (Standard DMP) ---
function (mood::MOODu1)(
    g::Any, p_idx::Int, rho_i::State{M}, nb_slice::UnitRange{Int}, 
    newRho::State{M}, pg::ParticleGrid{D}, int_buffer_f::AbstractVector{State{M}}
) where {D, M}
    
    minU, maxU = findLocalExtrema(rho_i, nb_slice, int_buffer_f)
    δ = mood.d
    
    # Check DMP component-by-component
    for m in 1:M
        if abs(maxU[m] - minU[m]) >= δ^3 # Flatness check
            if newRho[m] < minU[m] - δ || newRho[m] > maxU[m] + δ
                return true # MOOD event triggered
            end
        end
    end
    return false
end

# --- MOODu2 (Generic Fallback for Non-MUSCL gradients like Upwind) ---
function (mood::MOODu2)(
    g::Any, p_idx::Int, rho_i::State{M}, nb_slice::UnitRange{Int}, 
    newRho::State{M}, pg::ParticleGrid{D}, int_buffer_f::AbstractVector{State{M}}
) where {D, M}
    # No curvature available, so just evaluate u1 (DMP)
    return MOODu1(mood.d)(g, p_idx, rho_i, nb_slice, newRho, pg, int_buffer_f)
end


# --- MOODu2 (N-Dimensional MUSCL Optimization) ---
function (mood::MOODu2)(
    g::MUSCL{D, M, B_LEN, MAX_ORDER}, p_idx::Int, rho_i::State{M}, nb_slice::UnitRange{Int}, 
    newRho::State{M}, pg::ParticleGrid{D}, int_buffer_f::AbstractVector{State{M}}
) where {D, M, B_LEN, MAX_ORDER}
    
    # 1. Base Extrema Check (DMP)
    minU, maxU = findLocalExtrema(rho_i, nb_slice, int_buffer_f)
    δ = mood.d
    
    dmp_fail = false
    for m in 1:M
        if abs(maxU[m] - minU[m]) >= δ^3
            if newRho[m] < minU[m] - δ || newRho[m] > maxU[m] + δ
                dmp_fail = true
                break
            end
        end
    end
    
    if !dmp_fail; return false; end
    
    # 2. Curvature (u2) Check
    # ✅ FIXED: If the scheme doesn't support curvature (MAX_ORDER < 3) 
    # OR the particle has dynamically dropped to linear or lower, we cannot rescue it!
    if MAX_ORDER < 3 || g.particle_orders[p_idx] < 3
        return true # DMP failed, and no curvature info exists to rescue it, drop order
    end
    
    grad_vec = g.gradients
    neighbors = pg.neighbor.indices
    
    u2_satisfied = true
    
    # Check curvature in every dimension (xx, yy, zz...)
    for d in 1:D
        # Because of how we built the basis, spatial curves are perfectly aligned!
        curve_idx = D + d 
        c_i = grad_vec[p_idx][curve_idx]
        
        mini, maxi, minAbs, maxAbs = findLocalExtremaAbs(c_i, curve_idx, nb_slice, neighbors, grad_vec)
        
        for m in 1:M
            ratio = maxAbs[m] < 1e-12 ? 1.0 : minAbs[m] / maxAbs[m]
            valid = (mini[m] * maxi[m] > -δ) && (ratio >= 0.5 || maxAbs[m] < δ)
            
            if !valid
                u2_satisfied = false
                break
            end
        end
        if !u2_satisfied; break; end
    end
    
    return !u2_satisfied # Return true (Drop Order) if u2 was not satisfied
end