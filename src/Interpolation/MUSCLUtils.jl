"""
(1D Order 1-4) Calculate only slope_x using alfaij_bar.
Returns (slope_x, 0.0) to match the 2D signature.
"""
@inline function _calculate_slopes(
    nb_slice::UnitRange{Int},
    df_neighbors::AbstractVector,
    ws::MUSCLWorkspace1D
)
    slope_x = 0.0
    @inbounds for k in nb_slice
        # 1D slope always comes from alfaij_bar (the c1 coefficient)
        slope_x += ws.alfaij_bars[k] * df_neighbors[k]
    end
    return slope_x
end

# --- _calculate_higher_derivatives (1D) ---

"""
(1D Order 1) No higher derivatives. Returns empty tuple.
"""
@inline function _calculate_higher_derivatives(
    ::MUSCLORDER1,
    nb_slice::UnitRange{Int},
    df_neighbors::AbstractVector,
    ws::MUSCLWorkspace1D1O
)
    return () # Return empty tuple
end

"""
(1D Order 2) Calculate curve_xx (betaij).
"""
@inline function _calculate_higher_derivatives(
    ::MUSCLORDER2,
    nb_slice::UnitRange{Int},
    df_neighbors::AbstractVector,
    ws::MUSCLWorkspace1D2O
)
    curve_xx = 0.0
    @inbounds for k in nb_slice
        curve_xx += ws.betaijs[k] * df_neighbors[k]
    end
    return (curve_xx,) # Return tuple (curve_xx,)
end

"""
(1D Order 3) Calculate curve_xx (betaij) and d3f/dx3 (alfaij).
"""
@inline function _calculate_higher_derivatives(
    ::MUSCLORDER3,
    nb_slice::UnitRange{Int},
    df_neighbors::AbstractVector,
    ws::MUSCLWorkspace1D3O
)
    curve_xx = 0.0
    d3fdx3 = 0.0
    for k in nb_slice
        df = df_neighbors[k]
        curve_xx += ws.betaijs[k] * df # Curve uses betaij
        d3fdx3   += ws.alfaijs[k] * df # 3rd deriv uses alfaij
        if isnan(curve_xx); error("beta ",ws.betaijs[k],"df ",df) end
        if isnan(d3fdx3); error("alfa ",ws.alfaijs[k], "df ",df) end
    end
    return (curve_xx, d3fdx3)
end

"""
(1D Order 4) Calculate curve_xx, d3f/dx3, and d4f/dx4.
"""
@inline function _calculate_higher_derivatives(
    ::MUSCLORDER4,
    nb_slice::UnitRange{Int},
    df_neighbors::AbstractVector,
    ws::MUSCLWorkspace1D4O
)
    curve_xx = 0.0
    d3fdx3 = 0.0
    d4fdx4 = 0.0
    @inbounds for k in nb_slice
        df = df_neighbors[k]
        curve_xx += ws.betaijs[k] * df # Curve uses betaij
        d3fdx3   += ws.alfaijs[k] * df # 3rd deriv uses alfaij
        d4fdx4   += ws.gammaijs[k]* df # 4th deriv uses gammaij
    end
    return (curve_xx, d3fdx3, d4fdx4)
end


# --- _save_derivatives! (1D) ---
@inline function _save_derivatives!(
    ws::MUSCLWorkspace1D1O, i::Int, slope_x::Real, higher_derivatives::Tuple{}
)
    ws.slopes[i] = slope_x
    ws.curves_xx[i] = 0.
end
# --- _save_derivatives! (1D) ---
@inline function _save_derivatives!(
    ws::MUSCLWorkspace1D2O, i::Int, slope_x::Real, higher_derivatives
)
    curve_xx = higher_derivatives[1]
    ws.slopes[i] = slope_x
    ws.curves_xx[i] = curve_xx # Save curve_xx (even if 0.0 for O1)
end

@inline function _save_derivatives!(
    ws::MUSCLWorkspace1D3O, i::Int, slope_x::Real, higher_derivatives
)
    ws.slopes[i] = slope_x
    ws.curves_xx[i] = higher_derivatives[1]
    ws.d3fdx3[i] = higher_derivatives[2]
end

@inline function _save_derivatives!(
    ws::MUSCLWorkspace1D4O, i::Int, slope_x::Real, higher_derivatives
)
    ws.slopes[i] = slope_x
    ws.curves_xx[i] = higher_derivatives[1]
    ws.d3fdx3[i] = higher_derivatives[2]
    ws.d4fdx4[i] = higher_derivatives[3]
end

"""
Calculates the unlimited slopes for a single particle.
This is the internal logic from the old `calculate_slopes!(::MUSCLORDER1, ...)`
"""
function _calculate_slopes(
    nb_slice::UnitRange{Int64},
    df_neighbors::AbstractVector,   # View of neighbor df-values
    ws::MUSCLWorkspace2D
)
    alfaij = ws.alfaijs
    betaij = ws.betaijs
    slope_x = 0.0
    slope_y = 0.0
    @inbounds for k in nb_slice
        df = df_neighbors[k]
        slope_x += alfaij[k] * df
        slope_y += betaij[k] * df
    end
    return slope_x, slope_y
end

"""
(Order 1) No higher derivatives to calculate.
"""
@inline function _calculate_higher_derivatives(::MUSCLORDER1, nb_slice, df_neighbors, ws::MUSCLWorkspace2D1O)
    return () # Return empty tuple
end

"""
(Order 2) Calculate curve_xx, curve_yy, curve_xy.
"""
@inline function _calculate_higher_derivatives(
    ::MUSCLORDER2,
    nb_slice::UnitRange{Int},
    df_neighbors::AbstractVector,
    ws::MUSCLWorkspace2D2O # Requires the O2 workspace
)
    curve_xx = 0.0
    curve_yy = 0.0
    curve_xy = 0.0

    # O2 higher derivatives use alfaij_bar, betaij_bar, gammaij
    @inbounds for k in nb_slice
        df = df_neighbors[k]
        curve_xx += ws.alfaij_bars[k] * df
        curve_yy += ws.betaij_bars[k] * df
        curve_xy += ws.gammaijs[k] * df
    end
    return curve_xx, curve_yy, curve_xy
end

# --- In MUSCL.jl ---

"""
(Order 1) Save only slopes.
"""
@inline function _save_derivatives!(
    ws::MUSCLWorkspace2D1O, # Dispatches on O1 workspace
    i::Int,
    slopes::NTuple{2,Real},
    higher_derivatives::Tuple{} # Expects empty tuple
)
    ws.slopes_x[i] = slopes[1]
    ws.slopes_y[i] = slopes[2]
end

"""
(Order 2) Save slopes and curves.
"""
@inline function _save_derivatives!(
    ws::MUSCLWorkspace2D2O, # Dispatches on O2 workspace
    i::Int,
    slopes::NTuple{2,Real},
    higher_derivatives::NTuple{3, Real} # Expects (cxx, cyy, cxy)
)
    ws.slopes_x[i]  = slopes[1]
    ws.slopes_y[i]  = slopes[2]
    ws.curves_xx[i] = higher_derivatives[1]
    ws.curves_yy[i] = higher_derivatives[2]
    ws.curves_xy[i] = higher_derivatives[3]
end

# --- In MUSCLUtils.jl ---

# 1D Slope Calculation for Order 0 (Returns 0)
@inline function _calculate_slopes(
    nb_slice::UnitRange{Int},
    df_neighbors::AbstractVector,
    ws::MUSCLWorkspace1D0O
)
    return 0.0
end

# 2D Slope Calculation for Order 0 (Returns 0,0)
@inline function _calculate_slopes(
    nb_slice::UnitRange{Int64},
    df_neighbors::AbstractVector, 
    ws::MUSCLWorkspace2D0O
)
    return 0.0, 0.0
end

# 1D Higher Derivs for Order 0 (Returns empty)
@inline function _calculate_higher_derivatives(
    ::MUSCLORDER0, nb_slice, df_neighbors, ws::MUSCLWorkspace1D0O
)
    return ()
end

# 2D Higher Derivs for Order 0 (Returns empty)
@inline function _calculate_higher_derivatives(
    ::MUSCLORDER0, nb_slice, df_neighbors, ws::MUSCLWorkspace2D0O
)
    return ()
end

# Save Derivatives (No-op for Order 0 as we don't store slopes)
@inline function _save_derivatives!(
    ws::Union{MUSCLWorkspace1D0O, MUSCLWorkspace2D0O}, i::Int, slopes, higher_derivatives
)
    return
end

# Reconstruction (1D) - Already exists in your code, but ensuring coverage
function reconstruct_interface_states(::MUSCLORDER0, ws, fi, fj, p_idx, nb_idx, deltaPos)
    return fi, fj
end

# Reconstruction (2D)
function reconstruct_interface_states(::MUSCLORDER0, ws, fi, fj, p_idx, nb_idx, deltaX, deltaY)
    return fi, fj
end

# --- Reconstruction Helpers for fij and fji (2D) ---
# These are kept as they are called by the new functor
# It uses pre-calculated slopes.
function reconstruct_interface_states(::MUSCLORDER1, ws::MUSCLWorkspace2D1O, fi, fj, p_idx, nb_idx, deltaX, deltaY)
    fij = fi  + 0.5 * (deltaX * ws.slopes_x[p_idx]  + deltaY * ws.slopes_y[p_idx])
    fji = fj - 0.5 * (deltaX * ws.slopes_x[nb_idx] + deltaY * ws.slopes_y[nb_idx])
    return fij, fji
end


# --- REFACTORED: Order 2 now just fetches pre-calculated values ---
function reconstruct_interface_states(::MUSCLORDER2, ws::MUSCLWorkspace2D2O, f_i, f_j, p_idx, nb_idx, deltaX, deltaY)
    # Fetch derivatives for particle i
    slope_ix   = ws.slopes_x[p_idx]
    slope_iy   = ws.slopes_y[p_idx]
    curve_xx_i = ws.curves_xx[p_idx]
    curve_yy_i = ws.curves_yy[p_idx]
    curve_xy_i = ws.curves_xy[p_idx]

    # Fetch derivatives for neighbor j
    slope_jx   = ws.slopes_x[nb_idx]
    slope_jy   = ws.slopes_y[nb_idx]
    curve_xx_j = ws.curves_xx[nb_idx]
    curve_yy_j = ws.curves_yy[nb_idx]
    curve_xy_j = ws.curves_xy[nb_idx]

    # Taylor expansion
    h = 0.5 * deltaX
    k = 0.5 * deltaY
    fij = f_i + h*slope_ix + k*slope_iy + 0.5*(h*h*curve_xx_i + 2. *h*k*curve_xy_i + k*k*curve_yy_i)
    fji = f_j - h*slope_jx - k*slope_jy + 0.5*(h*h*curve_xx_j + 2. *h*k*curve_xy_j + k*k*curve_yy_j)
    
    return fij, fji
end
# --- In MUSCLUtils.jl, replace all 1D reconstruct_interface_states ---

# --- Reconstruction Helpers for fij and fji (1D) ---

function reconstruct_interface_states(::MUSCLORDER1, ws::MUSCLWorkspace1D1O, f_i, f_j, p_idx, nb_idx, deltaPos)
    fij = f_i + 0.5 * deltaPos * ws.slopes[p_idx]
    fji = f_j - 0.5 * deltaPos * ws.slopes[nb_idx]
    return fij, fji
end

function reconstruct_interface_states(::MUSCLORDER2, ws::MUSCLWorkspace1D2O, f_i, f_j, p_idx, nb_idx, deltaPos)
    # Read pre-calculated derivatives
    slope_i = ws.slopes[p_idx]
    curve_i = ws.curves_xx[p_idx]
    slope_j = ws.slopes[nb_idx]
    curve_j = ws.curves_xx[nb_idx]

    # Taylor expansion
    h = 0.5 * deltaPos
    h2 = h*h
    fij = f_i + h * slope_i + 0.5 * h2 * curve_i
    fji = f_j - h * slope_j + 0.5 * h2 * curve_j
    
    return fij, fji
end

function reconstruct_interface_states(::MUSCLORDER3, ws::MUSCLWorkspace1D3O, f_i, f_j, p_idx, nb_idx, deltaPos)
    # Read pre-calculated derivatives
    slope_i = ws.slopes[p_idx]
    curve_i = ws.curves_xx[p_idx]
    d3_i    = ws.d3fdx3[p_idx]
    slope_j = ws.slopes[nb_idx]
    curve_j = ws.curves_xx[nb_idx]
    d3_j    = ws.d3fdx3[nb_idx]

    h = 0.5 * deltaPos
    h2 = h*h
    h3 = h2*h
    
    # 3rd order Taylor expansion
    fij = f_i + h * slope_i + 0.5 * h2 * curve_i + (1.0/6.0) * h3 * d3_i
    fji = f_j - h * slope_j + 0.5 * h2 * curve_j - (1.0/6.0) * h3 * d3_j
    
    return fij, fji
end

function reconstruct_interface_states(::MUSCLORDER4, ws::MUSCLWorkspace1D4O, f_i, f_j, p_idx, nb_idx, deltaPos)
    # Read pre-calculated derivatives
    slope_i = ws.slopes[p_idx]
    curve_i = ws.curves_xx[p_idx]
    d3_i    = ws.d3fdx3[p_idx]
    d4_i    = ws.d4fdx4[p_idx]
    slope_j = ws.slopes[nb_idx]
    curve_j = ws.curves_xx[nb_idx]
    d3_j    = ws.d3fdx3[nb_idx]
    d4_j    = ws.d4fdx4[nb_idx]

    h = 0.5 * deltaPos
    h2 = h*h
    h3 = h2*h
    h4 = h3*h
    
    # 4th order Taylor expansion
    fij = f_i + h * slope_i + 0.5 * h2 * curve_i + (1.0/6.0) * h3 * d3_i + (1.0/24.0) * h4 * d4_i
    fji = f_j - h * slope_j + 0.5 * h2 * curve_j - (1.0/6.0) * h3 * d3_j + (1.0/24.0) * h4 * d4_j
    
    return fij, fji
end