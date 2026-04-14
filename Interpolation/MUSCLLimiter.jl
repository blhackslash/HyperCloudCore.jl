
# =========================================================================
# BASE LIMITER FUNCTIONS (Dispatched on Strategy!)
# =========================================================================

@inline _limit_slopes(::NoLimiter, raw_grad, args...) = raw_grad

@inline limiter_phi(::BarthJespersenLimiter, r::Real) = math_min(1.0, Float64(r))

@inline function limiter_phi(::VenkatakrishnanLimiter, r::Real)
    # math_max cleanly replaces the `if r <= 0.0 return 0.0` branch!
    r_pos = math_max(0.0, Float64(r)) 
    return (r_pos^2 + 2.0 * r_pos) / (r_pos^2 + r_pos + 2.0)
end

@inline function limiter_phi(::SuperbeeLimiter, r::Real)
    r_f = Float64(r)
    min1 = math_min(1.0, 2.0 * r_f)
    min2 = math_min(2.0, r_f)
    return math_max(0.0, math_max(min1, min2))
end

@inline function limiter_phi(::MinmodLimiter, r::Real)
    min1 = math_min(1.0, Float64(r))
    return math_max(0.0, min1)
end

"""
Generalized local extrema limiting. Works flawlessly for 1D, 2D, and 3D, 
and applies component-wise limiting for Systems of Equations (State{M}).
"""
function _limit_slopes(
    strategy::Union{BarthJespersenLimiter, VenkatakrishnanLimiter},
    raw_grad::SVector{B_LEN, State{M}},
    nb_slice::UnitRange{Int},
    f_i::State{M},
    f_neighbors::AbstractVector{State{M}}, 
    pg::ParticleGrid{D},
    distVec::AbstractVector{Space{D}}
) where {B_LEN, M, D}
    
    # 1. Stack-allocated Mutable Vectors! ZERO GC allocations.
    u_max = MVector{M, Float64}(f_i)
    u_min = MVector{M, Float64}(f_i)
    
    @inbounds for local_idx in 1:length(nb_slice)
        global_idx = nb_slice[local_idx]
        f_j = f_neighbors[global_idx]
        
        # Simple, native loops. M is a type parameter, so this unrolls perfectly.
        for m in 1:M
            u_max[m] = max(u_max[m], f_j[m])
            u_min[m] = min(u_min[m], f_j[m])
        end
    end

    # 2. Calculate limiting factor phi
    phi_i = MVector{M, Float64}(undef)
    fill!(phi_i, 1.0)
    
    # Pre-allocate a stack buffer for the Taylor reconstruction
    delta_recon = MVector{M, Float64}(undef)
    
    @inbounds for local_idx in 1:length(nb_slice)
        dist_k = distVec[local_idx]
        
        # Reconstruct difference at neighbor using ONLY the linear slopes (first D elements)
        fill!(delta_recon, 0.0)
        for d in 1:D
            grad_d = raw_grad[d]
            dist_d = dist_k[d]
            for m in 1:M
                delta_recon[m] += grad_d[m] * dist_d
            end
        end
        
        # Calculate phi component-by-component in-place
        for m in 1:M
            recon_m = delta_recon[m]
            if abs(recon_m) > 1e-12
                r = recon_m > 0.0 ? (u_max[m] - f_i[m]) / recon_m : (u_min[m] - f_i[m]) / recon_m
                phi_j = limiter_phi(strategy, r)
                phi_i[m] = min(phi_i[m], phi_j)
            end
        end
    end

    # 3. Apply limiting factor ONLY to the linear slopes
    # Repackages everything safely back into your strictly typed SVector
    limited_grad = SVector{B_LEN, State{M}}(ntuple(Val(B_LEN)) do k
        if k <= D
            State{M}(ntuple(m -> raw_grad[k][m] * phi_i[m], Val(M)))
        else
            raw_grad[k]
        end
    end)
    
    return limited_grad
end

# =========================================================================
# 1D DIRECTIONAL LIMITERS (Minmod & Superbee)
# =========================================================================
"""
(1D Exclusive) Applies Left/Right sweeping limiters component-wise.
"""
function _limit_slopes(
    strategy::Union{SuperbeeLimiter, MinmodLimiter},
    raw_grad::SVector{B_LEN, State{M}},
    nb_slice::UnitRange{Int},
    f_i::State{M},
    f_neighbors::AbstractVector{State{M}},
    pg::ParticleGrid{1},
    distVec::AbstractVector{Space{1}}
) where {B_LEN, M}
    
    val_L, val_R = f_i, f_i
    dist_L, dist_R = 0.0, 0.0
    min_dist_L, min_dist_R = Inf, Inf

    # Find the closest Left and Right neighbors
    @inbounds for local_idx in 1:length(nb_slice)
        global_idx = nb_slice[local_idx]
        dx_k = distVec[local_idx][1]
        
        if dx_k > 1e-9 && dx_k < min_dist_R # Right neighbor
            min_dist_R = dx_k
            val_R = f_neighbors[global_idx]
            dist_R = dx_k
        elseif dx_k < -1e-9 && -dx_k < min_dist_L # Left neighbor
            min_dist_L = -dx_k
            val_L = f_neighbors[global_idx]
            dist_L = dx_k
        end
    end

    # Calculate backward and forward differences
    slope_L = abs(dist_L) > 1e-12 ? (f_i - val_L) / (-dist_L) : zero(State{M})
    slope_R = abs(dist_R) > 1e-12 ? (val_R - f_i) / dist_R    : zero(State{M})

    # Apply limiter component-by-component
    limited_slope = State{M}(ntuple(Val(M)) do m
        sL = slope_L[m]
        sR = slope_R[m]
        
        if sL * sR <= 0.0
            return 0.0
        else
            r = abs(sR) < 1e-12 ? 1.0 : sL / sR
            # Automatically dispatches to Minmod or Superbee!
            phi = limiter_phi(strategy, r)
            return phi * sR
        end
    end)
    
    # Re-pack into the unified SVector. (1D Linear slope is always index 1)
    limited_grad = SVector{B_LEN, State{M}}(ntuple(Val(B_LEN)) do k
        k == 1 ? limited_slope : raw_grad[k]
    end)
    
    return limited_grad
end