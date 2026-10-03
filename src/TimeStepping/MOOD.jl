export MOODu1, MOODu2, NoMOOD, OnlyMOOD, EPD1, EPD2, EPD0, StrictEPD0, NoStrategy

# =========================================================================
# HALO STRATEGIES (Depth-based Order Reduction)
# =========================================================================

# Halo{0}: Completely local order reduction (no neighbor effects)
struct EPD0 <: Halo{0} end
struct EPD1 <: Halo{0} end

# Halo{1}: Order reduction forces neighbors to fall down and recalculate
struct StrictEPD0 <: Halo{1} end
struct EPD2 <: Halo{1} end

# --- Generic Node Recalculation Dispatch ---
# Dispatches purely on the depth parameter N, ensuring absolute type stability

@inline trigger_halo!(::Halo{0}, p_idx, pg, needs_recalc, orders) = nothing

@inline function trigger_halo!(::Halo{1}, p_idx, pg, needs_recalc, orders)
    nb_slices = pg.neighbor.ranges
    nb_indices = pg.neighbor.indices
    is_boundary = pg.core.is_boundary
    
    @inbounds for k in nb_slices[p_idx]
        j = nb_indices[k]
        if !is_boundary[j]
            needs_recalc[j] = true
            orders[j] = min(orders[j], orders[p_idx])
            
            # Cascade recalculation flag to neighbors of j
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
# These specific fallbacks only ever execute inside the MUSCL flux functor
@inline _get_interface_orders(::EPD1, oi, oj) = (min(oi, oj), min(oi, oj)) 
@inline _get_interface_orders(::EPD2, oi, oj) = (min(oi, oj), min(oi, oj)) 
@inline _get_interface_orders(::MOODStrategy, oi, oj) = (oi, oj) # Default for EPD0 / StrictEPD0

# =========================================================================
# MOOD CRITERIA
# =========================================================================

struct MOODu1{T} <: MOODCriterion 
    d::T
end

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

    # g.particle_orders is now universally accessible on the ParticleGridCore!
    if _extract_order(g) < 3 || pg.core.particle_orders[p_idx] < 3
        return true 
    end
    
    # Check if the scheme has stored gradients for MUSCL-specific u2 checks
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