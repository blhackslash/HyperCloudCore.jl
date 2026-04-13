# =========================================================================
# BRANCHLESS MATH HELPERS
# =========================================================================
@inline math_max(a::Float64, b::Float64) = 0.5 * (a + b + abs(a - b))
@inline math_min(a::Float64, b::Float64) = 0.5 * (a + b - abs(a - b))

# =========================================================================
# STATE{M} EXTREMA FINDERS
# =========================================================================

@inline function findLocalExtrema(rho_i::State{M}, nb_slice::UnitRange{Int}, neighbor_fs::AbstractVector{State{M}}) where M
    minU = rho_i
    maxU = rho_i
    @inbounds for k in nb_slice 
        rho_j = neighbor_fs[k]
        minU = State{M}(ntuple(m -> math_min(minU[m], rho_j[m]), Val(M)))
        maxU = State{M}(ntuple(m -> math_max(maxU[m], rho_j[m]), Val(M)))
    end
    return minU, maxU
end

@inline function findLocalExtremaAbs(
    c_i::State{M}, curve_idx::Int, nb_slice::UnitRange{Int}, 
    neighbor_indices::AbstractVector{Int}, grad_vec::AbstractVector
) where M
    mini = c_i
    maxi = c_i
    minAbs = State{M}(ntuple(m -> abs(c_i[m]), Val(M)))
    maxAbs = minAbs
    
    @inbounds for k in nb_slice
        j = neighbor_indices[k]
        c_j = grad_vec[j][curve_idx]
        abs_cj = State{M}(ntuple(m -> abs(c_j[m]), Val(M)))
        
        mini = State{M}(ntuple(m -> math_min(mini[m], c_j[m]), Val(M)))
        maxi = State{M}(ntuple(m -> math_max(maxi[m], c_j[m]), Val(M)))
        minAbs = State{M}(ntuple(m -> math_min(minAbs[m], abs_cj[m]), Val(M)))
        maxAbs = State{M}(ntuple(m -> math_max(maxAbs[m], abs_cj[m]), Val(M)))
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
    newRho::State{M}, pg::ParticleGrid{D}, neighbor_fs::AbstractVector{State{M}}
) where {D, M}
    
    minU, maxU = findLocalExtrema(rho_i, nb_slice, neighbor_fs)
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
    newRho::State{M}, pg::ParticleGrid{D}, neighbor_fs::AbstractVector{State{M}}
) where {D, M}
    # No curvature available, so just evaluate u1 (DMP)
    return MOODu1(mood.d)(g, p_idx, rho_i, nb_slice, newRho, pg, neighbor_fs)
end


# --- MOODu2 (N-Dimensional MUSCL Optimization) ---
function (mood::MOODu2)(
    g::MUSCL{D, M, B_LEN, ORDER}, p_idx::Int, rho_i::State{M}, nb_slice::UnitRange{Int}, 
    newRho::State{M}, pg::ParticleGrid{D}, neighbor_fs::AbstractVector{State{M}}
) where {D, M, B_LEN, ORDER}
    
    # 1. Base Extrema Check (DMP)
    minU, maxU = findLocalExtrema(rho_i, nb_slice, neighbor_fs)
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
    if ORDER < 2
        return true # DMP failed, and no curvature info exists to rescue it
    end
    
    # Access thread-local workspace natively
    ws = g.workspaces[mod1(Threads.threadid(), Threads.nthreads())]
    grad_vec = ws.gradients
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