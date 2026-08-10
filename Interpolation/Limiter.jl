# =========================================================================
# BASE LIMITER FUNCTIONS (Dispatched on Strategy!)
# =========================================================================

@inline _limit_slopes(::NoLimiter, raw_grad, args...) = raw_grad

@inline limiter_phi(::BarthJespersenLimiter, r::T) where {T} = math_min(one(T), r)

@inline function limiter_phi(::VenkatakrishnanLimiter, r::T) where {T}
    r_pos = math_max(zero(T), r) 
    return (r_pos^2 + T(2.0) * r_pos) / (r_pos^2 + r_pos + T(2.0))
end

@inline function limiter_phi(::SuperbeeLimiter, r::T) where {T}
    min1 = math_min(one(T), T(2.0) * r)
    min2 = math_min(T(2.0), r)
    return math_max(zero(T), math_max(min1, min2))
end

@inline function limiter_phi(::MinmodLimiter, r::T) where {T}
    min1 = math_min(one(T), r)
    return math_max(zero(T), min1)
end

function _limit_slopes(
    strategy::Union{BarthJespersenLimiter{Mode}, VenkatakrishnanLimiter{Mode}},
    raw_grad::SVector{B_LEN, State{M, T}},
    nb_slice::UnitRange{Int},
    f_i::State{M, T},
    f_neighbors::AbstractVector{State{M, T}}, 
    pg::ParticleGrid{D, M, T},
    distVec::AbstractVector{Space{D, T}},
    ::Val{DEGREE} 
) where {B_LEN, M, D, T, DEGREE, Mode}
    
    u_max = MVector{M, T}(f_i)
    u_min = MVector{M, T}(f_i)
    
    @inbounds for global_idx in nb_slice
        f_j = f_neighbors[global_idx]
        for m in 1:M
            u_max[m] = max(u_max[m], f_j[m])
            u_min[m] = min(u_min[m], f_j[m])
        end
    end

    phi_i = MVector{M, T}(undef)
    fill!(phi_i, one(T))
    
    @inbounds for global_idx in nb_slice
        dist_k = distVec[global_idx] 
        
        p_interface = build_basis(Val(DEGREE), T(0.5) * dist_k)
        
        delta_recon = zero(State{M, T})
        for k in 1:B_LEN
            delta_recon += raw_grad[k] * p_interface[k]
        end
        
        for m in 1:M
            recon_m = delta_recon[m]
            if abs(recon_m) > T(1e-12)
                r = recon_m > zero(T) ? (u_max[m] - f_i[m]) / recon_m : (u_min[m] - f_i[m]) / recon_m
                phi_j = limiter_phi(strategy, r)
                phi_i[m] = min(phi_i[m], phi_j)
            end
        end
    end

    limited_grad = SVector{B_LEN, State{M, T}}(ntuple(Val(B_LEN)) do k
        if Mode === :hard || k <= D
            State{M, T}(ntuple(m -> raw_grad[k][m] * phi_i[m], Val(M)))
        else
            raw_grad[k] 
        end
    end)
    
    return limited_grad
end

# =========================================================================
# 1D DIRECTIONAL LIMITERS (Minmod & Superbee)
# =========================================================================
function _limit_slopes(
    strategy::Union{SuperbeeLimiter{Mode}, MinmodLimiter{Mode}},
    raw_grad::SVector{B_LEN, State{M, T}},
    nb_slice::UnitRange{Int},
    f_i::State{M, T},
    f_neighbors::AbstractVector{State{M, T}},
    pg::ParticleGrid{1, M, T},
    distVec::AbstractVector{Space{1, T}},
    ::Val{DEGREE}
) where {B_LEN, M, T, DEGREE, Mode}
    
    val_L, val_R = f_i, f_i
    dist_L, dist_R = zero(T), zero(T)
    min_dist_L, min_dist_R = T(Inf), T(Inf)

    @inbounds for global_idx in nb_slice
        dx_k = distVec[global_idx][1] 
        
        if dx_k > T(1e-9) && dx_k < min_dist_R
            min_dist_R = dx_k
            val_R = f_neighbors[global_idx]
            dist_R = dx_k
        elseif dx_k < -T(1e-9) && -dx_k < min_dist_L
            min_dist_L = -dx_k
            val_L = f_neighbors[global_idx]
            dist_L = dx_k
        end
    end

    slope_L = abs(dist_L) > T(1e-12) ? (f_i - val_L) / (-dist_L) : zero(State{M, T})
    slope_R = abs(dist_R) > T(1e-12) ? (val_R - f_i) / dist_R    : zero(State{M, T})

    phi_scale = MVector{M, T}(undef)
    
    for m in 1:M
        sL = slope_L[m]
        sR = slope_R[m]
        orig_slope = raw_grad[1][m] 
        
        if sL * sR <= zero(T)
            phi_scale[m] = zero(T)
        else
            r = abs(sR) < T(1e-12) ? one(T) : sL / sR
            phi = limiter_phi(strategy, r)
            new_slope = phi * sR
            
            if orig_slope * new_slope <= zero(T)
                phi_scale[m] = zero(T)
            else
                phi_scale[m] = clamp(new_slope / orig_slope, zero(T), one(T))
            end
        end
    end
    
    limited_grad = SVector{B_LEN, State{M, T}}(ntuple(Val(B_LEN)) do k
        if Mode === :hard || k <= 1 
            State{M, T}(ntuple(m -> raw_grad[k][m] * phi_scale[m], Val(M))) 
        else
            raw_grad[k] 
        end
    end)
    
    return limited_grad
end