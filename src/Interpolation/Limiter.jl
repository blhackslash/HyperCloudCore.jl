export BarthJespersenLimiter, VenkatakrishnanLimiter, SuperbeeLimiter, MinmodLimiter, NoLimiter
"""
    BarthJespersenLimiter{Mode}
    VenkatakrishnanLimiter{Mode}
    SuperbeeLimiter{Mode}
    MinmodLimiter{Mode}

Slope limiters used to enforce monotonicity during MUSCL interface reconstruction.
- `BarthJespersenLimiter`: Applies a standard strict minimum/maximum bounding.
- `VenkatakrishnanLimiter`: Applies a differentiable, smooth relaxation of the Barth-Jespersen limiter bounds.
- `SuperbeeLimiter`: A 1D-focused limiter applying the Superbee bounding function.
- `MinmodLimiter`: A 1D-focused limiter returning the minimum modulus of the local slopes.
- All limiters can be instantiated with a specific `Mode` (such as `:soft` or `:hard`).
"""
struct BarthJespersenLimiter{Mode} <: AbstractSlopeLimiter 
    BarthJespersenLimiter(mode::Symbol=:soft) = new{mode}()
end
struct VenkatakrishnanLimiter{Mode} <: AbstractSlopeLimiter 
    VenkatakrishnanLimiter(mode::Symbol=:soft) = new{mode}()
end
struct SuperbeeLimiter{Mode} <: AbstractSlopeLimiter 
    SuperbeeLimiter(mode::Symbol=:soft) = new{mode}()
end
struct MinmodLimiter{Mode} <: AbstractSlopeLimiter 
    MinmodLimiter(mode::Symbol=:soft) = new{mode}()
end
struct NoLimiter <: AbstractSlopeLimiter end

# =========================================================================
# BASE LIMITER FUNCTIONS (Dispatched on Strategy!)
# =========================================================================

@inline _limit_slopes(::NoLimiter, raw_grad, args...) = raw_grad

@inline limiter_phi(::BarthJespersenLimiter, r::T) where {T} = math_min(one(T), r)

# (r² + 2r) / (r² + r + 2), divided by r² for r > 1: r² overflows Float32 from r ≈ 1.8e19 (Inf/Inf = NaN, which
# `min` then spreads over the whole limited gradient).
@inline function limiter_phi(::VenkatakrishnanLimiter, r::T) where {T}
    r_pos = math_max(zero(T), r)
    s     = one(T) / math_max(one(T), r_pos)
    small = (r_pos^2 + T(2) * r_pos) / (r_pos^2 + r_pos + T(2))
    large = (one(T) + T(2) * s) / (one(T) + s + T(2) * s^2)
    return ifelse(r_pos > one(T), large, small)
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

# Component-wise extrema of the target state and its neighbours. Static `SVector`s only:
# this runs inside the MUSCL pre-gather kernel, where an `MVector` would allocate on the
# host and does not compile on device backends.
@inline function _local_extrema(f_i::State{M, T}, f_neighbors, nb_slice) where {M, T}
    u_max = f_i
    u_min = f_i
    @inbounds for global_idx in nb_slice
        f_j = f_neighbors[global_idx]
        u_max = max.(u_max, f_j)
        u_min = min.(u_min, f_j)
    end
    return u_max, u_min
end

# Tighten the bounding coefficient of every equation with one reconstructed interface increment. Increments at
# round-off level of the local data (`|u_i|` and the neighbourhood range) are left alone, relative to that scale
# rather than absolute, so a profile of height 1e-13 is limited like one of height 1.
@inline function _update_phi(strategy, phi::State{M, T}, delta_recon::State{M, T}, f_i::State{M, T},
                             u_max::State{M, T}, u_min::State{M, T}) where {M, T}
    return State{M, T}(ntuple(Val(M)) do m
        recon_m = delta_recon[m]
        tol_m   = 16 * eps(T) * max(abs(f_i[m]), u_max[m] - u_min[m])
        if abs(recon_m) > tol_m
            r = recon_m > zero(T) ? (u_max[m] - f_i[m]) / recon_m : (u_min[m] - f_i[m]) / recon_m
            min(phi[m], limiter_phi(strategy, r))
        else
            phi[m]
        end
    end)
end

# Bounding coefficient per equation over all interfaces of the neighbourhood.
@inline function _bounding_phi(strategy, raw_grad::SVector{B_LEN, State{M, T}}, nb_slice, f_i::State{M, T},
                               u_max::State{M, T}, u_min::State{M, T}, distVec, ::Val{DEGREE}) where {B_LEN, M, T, DEGREE}
    phi = ones(State{M, T})
    @inbounds for global_idx in nb_slice
        p_interface = build_basis(Val(DEGREE), T(0.5) * distVec[global_idx])

        delta_recon = zero(State{M, T})
        for k in 1:B_LEN
            delta_recon += raw_grad[k] * p_interface[k]
        end

        phi = _update_phi(strategy, phi, delta_recon, f_i, u_max, u_min)
    end
    return phi
end

"""
    _limit_slopes(strategy, raw_grad, nb_slice, f_i, f_neighbors, pg, distVec, ::Val{DEGREE})

Applies the specified slope limiter strategy to the raw gradient of a target particle.

# Details
- For `BarthJespersen` and `Venkatakrishnan`, it evaluates local maxima and minima, reconstructs the interface state using the raw gradient, and computes a limiting coefficient `phi` for each target equation.
- For `Minmod` and `Superbee`, it computes left and right finite difference slopes based on neighbor distances and limits the gradient strictly along a 1D stencil.
- Returns a limited gradient vector where specific polynomial basis terms are scaled by the bounding coefficient `phi`.
- Runs inside the MUSCL pre-gather kernel: all per-equation scratch is static (`SVector`), nothing is allocated.
"""
function _limit_slopes(
    strategy::Union{BarthJespersenLimiter{Mode}, VenkatakrishnanLimiter{Mode}},
    raw_grad::SVector{B_LEN, State{M, T}},
    nb_slice::OrdinalRange{Int, Int},
    f_i::State{M, T},
    f_neighbors::AbstractVector{State{M, T}}, 
    pg::ParticleGrid{D, M, T},
    distVec::AbstractVector{Space{D, T}},
    ::Val{DEGREE} 
) where {B_LEN, M, D, T, DEGREE, Mode}
    
    u_max, u_min = _local_extrema(f_i, f_neighbors, nb_slice)
    phi_i = _bounding_phi(strategy, raw_grad, nb_slice, f_i, u_max, u_min, distVec, Val(DEGREE))

    limited_grad = SVector{B_LEN, State{M, T}}(ntuple(Val(B_LEN)) do k
        if Mode === :hard || k <= D
            raw_grad[k] .* phi_i
        else
            raw_grad[k] 
        end
    end)
    
    return limited_grad
end

# =========================================================================
# 1D DIRECTIONAL LIMITERS (Minmod & Superbee)
# =========================================================================
# Sign tests instead of products: `sL * sR` underflows to zero for slopes below ~1e-154 (1e-19 in Float32)
@inline _same_sign(a, b) = !iszero(a) && !iszero(b) && signbit(a) == signbit(b)

# Minmod and Superbee limit along the 1-D stencil only: rejected when the scheme is built (the kernel-side error
# below stays as a guard for a scheme used on a grid of another dimension)
_check_limiter_dimension(limiter, D::Int) = nothing
function _check_limiter_dimension(limiter::Union{SuperbeeLimiter, MinmodLimiter}, D::Int)
    D == 1 || throw(ArgumentError("$(nameof(typeof(limiter))) is a one-dimensional limiter and cannot be used in $(D)-D; " *
                                  "use BarthJespersenLimiter or VenkatakrishnanLimiter instead."))
    return nothing
end

function _limit_slopes(strategy::Union{SuperbeeLimiter, MinmodLimiter}, raw_grad, nb_slice, f_i, f_neighbors,
                       pg::ParticleGrid{D}, distVec, ::Val) where {D}
    error("$(nameof(typeof(strategy))) is a one-dimensional limiter and cannot be used on a $(D)-D grid; " *
          "use BarthJespersenLimiter or VenkatakrishnanLimiter instead.")
end

function _limit_slopes(
    strategy::Union{SuperbeeLimiter{Mode}, MinmodLimiter{Mode}},
    raw_grad::SVector{B_LEN, State{M, T}},
    nb_slice::OrdinalRange{Int, Int},
    f_i::State{M, T},
    f_neighbors::AbstractVector{State{M, T}},
    pg::ParticleGrid{1, M, T},
    distVec::AbstractVector{Space{1, T}},
    ::Val{DEGREE}
) where {B_LEN, M, T, DEGREE, Mode}
    
    val_L, val_R = f_i, f_i
    dist_L, dist_R = zero(T), zero(T)
    # Nearest neighbour on each side; (nearly) coincident particles are skipped, relative to the spacing
    min_dist = T(1e-6) * pg.meta.dx[1]

    @inbounds for global_idx in nb_slice
        dx_k = distVec[global_idx][1]

        if dx_k > min_dist && (iszero(dist_R) || dx_k < dist_R)
            val_R = f_neighbors[global_idx]
            dist_R = dx_k
        elseif dx_k < -min_dist && (iszero(dist_L) || dx_k > dist_L)
            val_L = f_neighbors[global_idx]
            dist_L = dx_k
        end
    end

    slope_L = iszero(dist_L) ? zero(State{M, T}) : (f_i - val_L) / (-dist_L)
    slope_R = iszero(dist_R) ? zero(State{M, T}) : (val_R - f_i) / dist_R

    phi_scale = State{M, T}(ntuple(Val(M)) do m
        sL = slope_L[m]
        sR = slope_R[m]
        orig_slope = raw_grad[1][m]

        if !_same_sign(sL, sR)
            zero(T)
        else
            new_slope = limiter_phi(strategy, sL / sR) * sR

            _same_sign(orig_slope, new_slope) ? clamp(new_slope / orig_slope, zero(T), one(T)) : zero(T)
        end
    end)
    
    limited_grad = SVector{B_LEN, State{M, T}}(ntuple(Val(B_LEN)) do k
        if Mode === :hard || k <= 1 
            raw_grad[k] .* phi_scale
        else
            raw_grad[k] 
        end
    end)
    
    return limited_grad
end