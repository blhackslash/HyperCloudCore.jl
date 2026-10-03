export MOODu1, MOODu2, OnlyMOOD, EPD1, EPD2, EPD0, StrictEPD0, Halo
# =========================================================================
# THE HALO STRATEGIES (Force_Depth, Recalc_Depth)
# =========================================================================

"""
    Halo{Force_N, Recalc_N} <: MOODStrategy

A MOOD strategy parameterizing how local order degradation cascades.
- `Force_N`: The topological depth (hops) to which neighboring particles are aggressively forced to drop their structural order.
- `Recalc_N`: The topological depth to which particles are flagged for recalculation in the next phase.
"""
abstract type Halo{Force_N, Recalc_N} <: MOODStrategy end

"""
    EPD0 <: Halo{0, 0}

Strictly local Edge Polynomial Degree strategy. 
If a particle drops its order, it does not affect its neighbors' orders or trigger any recalculations.
"""
struct EPD0 <: Halo{0, 0} end

"""
    EPD1 <: Halo{0, 1}

Symmetric minimum Edge Polynomial Degree strategy.
If a particle drops its order, its immediate neighbors (1-hop) are flagged for recalculation to ensure shared symmetric interfaces are properly updated. No structural order forcing is applied.
"""
struct EPD1 <: Halo{0, 1} end

"""
    EPD2 <: Halo{0, 2}

Neighborhood minimum Edge Polynomial Degree strategy.
If a particle drops its order, recalculation flags cascade to immediate and next-nearest neighbors (2-hop) to accommodate the wider effective stencil evaluation. No structural order forcing is applied.
"""
struct EPD2 <: Halo{0, 2} end

"""
    StrictEPD0 <: Halo{1, 2}

Aggressive local Edge Polynomial Degree strategy.
If a particle drops its order, it violently forces all immediate neighbors (1-hop) to drop their structural order to match. This triggers recalculation flags up to next-nearest neighbors (2-hop).
"""
struct StrictEPD0 <: Halo{1, 2} end

# --- Depth-Based Trigger Dispatch ---

# Halo{0, 0}: Strictly local. No neighbors affected.
@inline trigger_halo!(::Halo{0, 0}, p_idx, pg, needs_recalc, orders) = nothing

# Halo{0, 1}: 1-Hop Recalculation (No Forcing). Used by EPD1.
@inline function trigger_halo!(::Halo{0, 1}, p_idx, pg, needs_recalc, orders)
    nb_slices = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices
    is_boundary = pg.core.is_boundary
    
    @inbounds for k in nb_slices[p_idx]
        j = nb_indices[k]
        if !is_boundary[j]
            needs_recalc[j] = true
        end
    end
    return nothing
end

# Halo{0, 2}: 2-Hop Recalculation (No Forcing). Used by EPD2.
@inline function trigger_halo!(::Halo{0, 2}, p_idx, pg, needs_recalc, orders)
    nb_slices = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices
    is_boundary = pg.core.is_boundary
    
    @inbounds for k in nb_slices[p_idx]
        j = nb_indices[k]
        if !is_boundary[j]
            needs_recalc[j] = true
            
            # Cascade flag to next-nearest neighbors
            for m in nb_slices[j]
                nj = nb_indices[m]
                if !is_boundary[nj]
                    needs_recalc[nj] = true
                end
            end
        end
    end
    return nothing
end

# Halo{1, 2}: 1-Hop Forcing + 2-Hop Recalculation. Used by StrictEPD0.
@inline function trigger_halo!(::Halo{1, 2}, p_idx, pg, needs_recalc, orders)
    nb_slices = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices
    is_boundary = pg.core.is_boundary
    
    @inbounds for k in nb_slices[p_idx]
        j = nb_indices[k]
        if !is_boundary[j]
            needs_recalc[j] = true
            # Force order drop on neighbor
            orders[j] = min(orders[j], orders[p_idx])
            
            # Cascade flag to next-nearest neighbors
            for m in nb_slices[j]
                nj = nb_indices[m]
                if !is_boundary[nj]
                    needs_recalc[nj] = true
                end
            end
        end
    end
    return nothing
end

# --- MUSCL-Exclusive Interface Evaluators ---
# Used strictly during the flux pass to resolve the interface polynomials
@inline _get_interface_orders(::EPD1, oi, oj) = (min(oi, oj), min(oi, oj)) 
@inline _get_interface_orders(::EPD2, oi, oj) = (min(oi, oj), min(oi, oj)) 
@inline _get_interface_orders(::MOODStrategy, oi, oj) = (oi, oj) # Default for EPD0 / StrictEPD0

# =========================================================================
# MOOD CRITERIA
# =========================================================================

"""
    MOODu1{T} <: MOODCriterion

The standard Discrete Maximum Principle (DMP) check. Flags a particle if its updated state falls outside the extrema of its spatial neighborhood, padded by a relaxation threshold `d`.
"""
struct MOODu1{T} <: MOODCriterion 
    d::T
end

"""
    MOODu2{T} <: MOODCriterion

A MUSCL-optimized physical admissibility check. Acts as `MOODu1`, but if the strict DMP fails, it analyzes the magnitudes and ratios of the local spatial gradients. If the gradients indicate a smooth physical extremum (rather than a numerical oscillation), it overrides the DMP and permits the state.
"""
struct MOODu2{T} <: MOODCriterion 
    d::T
end

struct OnlyMOOD <: MOODCriterion end

MOOD(criterion::MOODCriterion) = MOOD(EPD1(), criterion)

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
    g::DivergenceInterpolator, p_idx::Int, rho_i::State{M, T}, nb_slice::UnitRange{Int}, 
    newRho::State{M, T}, pg::ParticleGrid{D, M, T}, int_buffer_f::AbstractVector{State{M, T}}
) where {D, M, T}
    
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

    # g.particle_orders is universally accessible on the ParticleGridCore!
    if _extract_order(g) < 3 || pg.core.particle_orders[p_idx] < 3
        return true 
    end
    
    if !hasproperty(g, :gradients)
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

# Fallback for schemes without gradients
function (m::MOOD{<:MOODStrategy, MOODu2{T}})(
    g::Any, p_idx::Int, rho_i::State{M, T}, nb_slice::UnitRange{Int}, 
    newRho::State{M, T}, pg::ParticleGrid{D, M, T}, int_buffer_f::AbstractVector{State{M, T}}
) where {D, M, T}
    return MOOD(m.strategy, MOODu1(m.criterion.d))(g, p_idx, rho_i, nb_slice, newRho, pg, int_buffer_f)
end