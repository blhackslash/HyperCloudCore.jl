function initMOOD!(mood::MOODCriterion, d)
    return
end



# MOODu1 functor signature now includes pg
function (mood::MOODu1)(
    g,        # The primary gradient interpolator
    i::Int,                         # Current particle index
    rho_i::Float64,                 # Value of rho at particle i
    nb_slice::UnitRange{Int},
    newRho::Float64,                # Proposed new value
    pg::ParticleGrid,     # Grid to access neighbor info
    neighbor_fs::AbstractVector{Float64} # Full neighbor rho vector
)::Bool
    
    # Calculate local extrema using the helper with direct indexing
    minU, maxU = findLocalExtrema(rho_i, nb_slice, neighbor_fs)
    
    δ = mood.d # Relaxation parameter

    # Basic DMP check with relaxation delta
    moodEvent = (newRho < minU - δ) || (newRho > maxU + δ)
    
    # Flatness check
    if abs(maxU - minU) < δ^3 
        moodEvent = false
    end
    
    return moodEvent
end

# Helper to check for curvature remains the same
function _has_curvature(g)
    if hasproperty(g, :workspace)
        return hasproperty(g.workspace, :curves_xx) && hasproperty(g.workspace, :curves_yy)
    else 
        return false
    end 
end

# --- MOODu2 Functor (1D) ---
function (mood::MOODu2)(
    g,     # The primary gradient interpolator (1D)
    i::Int,                         # Current particle index
    rho_i::Float64,                 # Value of rho at particle i
    nb_slice::UnitRange{Int},
    newRho::Float64,                # Proposed new value
    pg::ParticleGrid1D,  # Grid to access neighbor info (1D)
    neighbor_fs::AbstractVector{Float64} # Full neighbor rho vector
)::Bool

    minU, maxU = findLocalExtrema(rho_i, nb_slice, neighbor_fs)
    δ = mood.d

    # Basic DMP check
    DMPFail = (newRho < minU - δ) || (newRho > maxU + δ)
    if abs(maxU - minU) < δ^3 # Flatness check
        DMPFail = false
    end
    
    # --- Conditional u2 check for 1D ---
    u2_satisfied = false 
    if _has_curvature(g)
        curve_vec = g.workspace.curves_xx # Assumes 1D curve stored here
        curve_i   = curve_vec[i]
        
        mini, maxi, minAbs, maxAbs = findLocalExtremaAbs( # Dispatches to 1D version
            curve_i, nb_slice, pg.neighbor.indices, curve_vec
        )
        
        ratio = (maxAbs < 1e-12) ? 1.0 : minAbs / maxAbs 
        u2_satisfied = (mini * maxi > -δ) && ((ratio >= 0.5) || (maxAbs < δ))
    end
    # --- End Conditional u2 check ---
    
    moodEvent = DMPFail ? !u2_satisfied : false
    return moodEvent
end

# --- MOODu2 Functor (2D) ---
function (mood::MOODu2)(
    g,     # The primary gradient interpolator (2D)
    i::Int,                         # Current particle index
    rho_i::Float64,                 # Value of rho at particle i
    nb_slice::UnitRange{Int},
    newRho::Float64,                # Proposed new value
    pg::ParticleGrid2D,  # Grid to access neighbor info (2D)
    neighbor_fs::AbstractVector{Float64} # Full neighbor rho vector
)::Bool
    
    num_nb = pg.neighbor.amount[i]
    if num_nb == 0; return false; end

    minU, maxU = findLocalExtrema(rho_i, nb_slice, neighbor_fs)
    δ = mood.d

    # Basic DMP check
    DMPFail = (newRho < minU - δ) || (newRho > maxU + δ)
    if abs(maxU - minU) < δ^3 # Flatness check
        DMPFail = false
    end
    
    # --- Conditional u2 check for 2D ---
    u2_satisfied = false 
    if _has_curvature(g)
        curve_xx_vec = g.workspace.curves_xx
        curve_yy_vec = g.workspace.curves_yy
        curve_xx_i   = curve_xx_vec[i]
        curve_yy_i   = curve_yy_vec[i]
        
        extrema_vals = findLocalExtremaAbs( # Dispatches to 2D version
            curve_xx_i, curve_yy_i, nb_slice, pg.neighbor.indices, 
            curve_xx_vec, curve_yy_vec
        )
        mini1, maxi1, minAbs1, maxAbs1 = extrema_vals[1:4]
        mini2, maxi2, minAbs2, maxAbs2 = extrema_vals[5:8]
        
        ratio1 = (maxAbs1 < 1e-12) ? 1.0 : minAbs1 / maxAbs1 
        ratio2 = (maxAbs2 < 1e-12) ? 1.0 : minAbs2 / maxAbs2 
        
        u2x = (mini1 * maxi1 > -δ) && ((ratio1 >= 0.5) || (maxAbs1 < δ))
        u2y = (mini2 * maxi2 > -δ) && ((ratio2 >= 0.5) || (maxAbs2 < δ))
        u2_satisfied = u2x && u2y
    end
    # --- End Conditional u2 check ---
    
    moodEvent = DMPFail ? !u2_satisfied : false
    return moodEvent
end

function (mood::NoMOOD)(kwargs...)::Bool
    return false
end
function (mood::OnlyMOOD)(kwargs...)::Bool
    return true
end