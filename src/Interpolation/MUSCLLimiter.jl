function _limit_slopes(::NoLimiter, slopes, kwargs...)
    return slopes
end

function _limit_slopes(::RealSlopeLimiter, kwargs...)
    error("Slope limiting not implemented for the requested order, dimension or method!")
end

function minmod_phi(r::Real)::Float64
    if r <= 0.0
        return 0.0
    else
        return min(1.0, r)
    end
end

function superbee_phi(r::Real)::Float64
    if r <= 0.0
        return 0.0
 
   else
        return max(min(1.0, 2.0 * r), min(2.0, r))
    end
end

function venkatakrishnan_psi(r::Real)::Float64
    if r <= 0.0
        return 0.0
    end
    return (r^2 + 2.0 * r) / (r^2 + r + 2.0)
end

"""
    find_closest_lr_neighbors_1D(nb_slice, dx_global, f_neighbors_global)

Finds the closest neighbor to the left and right of a particle,
using the pre-calculated flat neighbor arrays.
"""
function find_closest_lr_neighbors_1D(
    nb_slice::UnitRange{Int},
    dx_global::AbstractVector,
    f_neighbors_global::AbstractVector
)
    # Initialize return values
    val_L, dist_L = 0.0, 0.0
    val_R, dist_R = 0.0, 0.0
    
    # Initialize minimum distances found so far
    min_abs_dist_L = Inf
    min_dist_R = Inf

    # Access the neighbor data directly from the flat global arrays
    @inbounds for k in nb_slice
        dx_ij = dx_global[k]

        if dx_ij > 1e-9 # Potential right neighbor
            if dx_ij < min_dist_R
                min_dist_R = dx_ij
                val_R = f_neighbors_global[k]
                dist_R = dx_ij
            end
        elseif dx_ij < -1e-9 # Potential left neighbor
            abs_dx_ij = abs(dx_ij)
            if abs_dx_ij < min_abs_dist_L
                min_abs_dist_L = abs_dx_ij
                val_L = f_neighbors_global[k]
                dist_L = dx_ij # Keep its negative sign
            end
        end
    end
    
    return val_L, dist_L, val_R, dist_R
end

# --- 1D Limiters (matching 2D signature) ---

"""
(1D Dispatch) Minmod/Superbee. Limits slope_x.
"""
function _limit_slopes(
    strategy::Union{SuperbeeLimiter, MinmodLimiter},
    slope_x::Real,
    nb_slice::UnitRange{Int},
    f_i::Real,
    f_neighbors::AbstractVector,
    pg::ParticleGrid1D # Dispatches on 1D grid
)
    dx = get_xdistance(pg)

    # Find neighbors (using the existing 1D helper)
    val_L, dist_L, val_R, dist_R = find_closest_lr_neighbors_1D(nb_slice, dx, f_neighbors) # Needs fix

    # --- This logic is simplified from your helper ---
    # We need to find the closest L/R *from the neighbor list*
    val_L, dist_L = 0.0, 0.0
    val_R, dist_R = 0.0, 0.0
    min_dist_L, min_dist_R = Inf, Inf

    @inbounds for k in nb_slice
        dx_k = dx[k]
        if dx_k > 1e-9 && dx_k < min_dist_R # Right neighbor
            min_dist_R = dx_k
            val_R = f_neighbors[k]
            dist_R = dx_k
        elseif dx_k < -1e-9 && -dx_k < min_dist_L # Left neighbor
            min_dist_L = -dx_k
            val_L = f_neighbors[k]
            dist_L = dx_k
        end
    end
    # --- End find neighbors ---

    slope_L = abs(dist_L) > 1e-12 ? (f_i - val_L) / (-dist_L) : 0.0
    slope_R = abs(dist_R) > 1e-12 ? (val_R - f_i) / dist_R  : 0.0

    local limited_slope_x
    if slope_L * slope_R <= 0.0
        limited_slope_x = 0.0
    else
        r = slope_R ≈ 0.0 ? 1.0 : slope_L / slope_R
        phi = strategy isa SuperbeeLimiter ? superbee_phi(r) : minmod_phi(r)
        limited_slope_x = phi * slope_R
    end
    
    return limited_slope_x
end

"""
(1D Dispatch) Barth-Jespersen/Venkatakrishnan. Limits slope_x.
"""
function _limit_slopes(
    strategy::Union{BarthJespersenLimiter, VenkatakrishnanLimiter},
    slope_x::Real,
    nb_slice::UnitRange{Int},
    f_i::Real,
    f_neighbors::AbstractVector,
    pg::ParticleGrid1D # Dispatches on 1D grid
)
    dx = get_xdistance(pg)
    if isempty(nb_slice) || abs(slope_x) < 1e-12
        return 0.0
    end

    u_max_stencil = f_i
    u_min_stencil = f_i
    @inbounds for k in nb_slice
        u_max_stencil = max(u_max_stencil, f_neighbors[k])
        u_min_stencil = min(u_min_stencil, f_neighbors[k])
    end

    phi_i = 1.0
    @inbounds for k in nb_slice
        dx_ij = dx[k] # 1D uses only dx
        delta_recon = slope_x * dx_ij 
        
        if abs(delta_recon) < 1e-12; continue; end
        
        r = if delta_recon > 0.0 # Overshoot
            (u_max_stencil - f_i) / delta_recon
        else # Undershoot
            (u_min_stencil - f_i) / delta_recon
        end
        
        phi_j = strategy isa BarthJespersenLimiter ? min(1.0, r) : venkatakrishnan_psi(r)
        phi_i = min(phi_i, phi_j)
    end
    
    limited_slope_x = slope_x * clamp(phi_i, 0.0, 1.0)
    return limited_slope_x
end

"""
Local slope limiting function (RealSlopeLimiter dispatch).
Applies geometric limiting logic from the old `limit_slopes!(...)`
"""
function _limit_slopes(
    strategy::Union{BarthJespersenLimiter, VenkatakrishnanLimiter},
    slopes::NTuple{2,<:Real},
    nb_slice::UnitRange{Int},
    f_i::Real,
    f_neighbors::AbstractVector, # View of neighbor f-values
    pg::ParticleGrid2D
)
    dx = get_xdistance(pg)
    dy = get_ydistance(pg)
    slope_x = slopes[1]
    slope_y = slopes[2]

    ui = f_i
    if slope_x^2 + slope_y^2 < 1e-12
        return 0.0, 0.0
    end

    # Find min/max among neighbors
    u_max = ui
    u_min = ui
    @inbounds for k in nb_slice
        f_neighbor = f_neighbors[k]
     
        u_max = max(u_max, f_neighbor)
        u_min = min(u_min, f_neighbor)
    end

    phi_i = 1.0
    
    @inbounds for k in nb_slice
     
        delta_recon = slope_x * dx[k] + slope_y * dy[k]
        
        if abs(delta_recon) < 1e-12; continue; end
        
        r = delta_recon > 0.0 ? (u_max - ui) / delta_recon : (u_min - ui) / delta_recon
        phi_j = strategy isa BarthJespersenLimiter ? min(1.0, r) : venkatakrishnan_psi(r)
        phi_i = min(phi_i, phi_j)
    end

    phi_i = clamp(phi_i, 0.0, 1.0)
    
    limited_slope_x = slope_x * phi_i
    limited_slope_y = slope_y * phi_i
    
    return limited_slope_x, limited_slope_y
end