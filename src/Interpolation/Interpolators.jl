
"""
    Interpolator{D, IO, DO}()

A generalized moving least squares (MLS) interpolator struct parameterized for arbitrary dimensions and orders. 

# Type Parameters
- `D`: The spatial dimension of the problem.
- `IO`: The interpolation order.
- `DO`: The derivative order.
"""
struct Interpolator{D, IO, DO}
    function Interpolator{D, IO, DO}() where {D, IO, DO}
        new{D, IO, DO}()
    end
end

struct ConstantReconstruction end

"""
    _mls_reg(T)

Relative Tikhonov term of the *retry* factorisation in [`_cholesky_moment`](@ref): `ε tr(N) / B` is added to
the diagonal only after the plain Cholesky factorisation of a moment matrix has failed or produced a pivot at
round-off level. `ε = sqrt(eps(T))` (1.5e-8 in Float64, 3.5e-4 in Float32): the normal equations square the
conditioning, and `sqrt(eps)` balances the two errors of the retry, the spurious coefficients of the directions
the stencil cannot determine (∝ `eps / ε`) against the distortion of the directions it can (∝ `ε`). Healthy
stencils are never regularised: a fixed `1e-10` capped high-order Float64 fits at errors of ~1e-8, and any term
that is representable in Float32 costs an order of magnitude on degree-5 fits.
"""
@inline _mls_reg(::Type{T}) where {T} = sqrt(eps(T))

# Smallest acceptable Cholesky pivot of a moment matrix, relative to its diagonal entry. A smaller one is the
# round-off of a (numerically) singular stencil, which `cholesky` alone does not report.
@inline _pivot_tol(::Type{T}) where {T} = T(10 * eps(T))

"""
    _cholesky_moment(N_s::SMatrix{B, B, T}) -> (C, ok)

Cholesky factorisation of an MLS moment matrix with conditional regularisation. The matrix is factorised as it
is; only if that fails, or leaves a pivot below `_pivot_tol(T)` times its diagonal entry (too few or collinear
neighbours), is the Tikhonov term `_mls_reg(T) tr(N_s) / B` added to the diagonal and the factorisation
repeated. A healthy stencil therefore keeps the exact fit for the price of one factorisation; a degenerate one
gets a damped fit instead of none. `ok` is `false` if the retry fails as well.
"""
@inline function _cholesky_moment(N_s::SMatrix{B, B, T}) where {B, T}
    C = cholesky(Symmetric(N_s), check = false)
    _pivots_ok(C, N_s) && return C, true
    eps_reg = _mls_reg(T) * tr(N_s) / B
    C = cholesky(Symmetric(N_s + eps_reg * I), check = false)
    return C, LinearAlgebra.issuccess(C)
end

# `true` if the factorisation succeeded with every pivot `U[k, k]^2` above `_pivot_tol(T) * N_s[k, k]`
# (strict, so zero and NaN pivots fail too).
@inline function _pivots_ok(C, N_s::SMatrix{B, B, T}) where {B, T}
    LinearAlgebra.issuccess(C) || return false
    U   = C.factors
    tol = _pivot_tol(T)
    @inbounds for k in 1:B
        U[k, k]^2 > tol * N_s[k, k] || return false
    end
    return true
end

# =========================================================================
# UNIVERSAL DYNAMIC DISPATCH ROUTERS (Zero-Allocation)
# =========================================================================

@inline _pad_grad(g::SVector{L, T}, ::Val{MAX_L}) where {L, MAX_L, T} = SVector{MAX_L, T}(ntuple(i -> i <= L ? g[i] : zero(T), Val(MAX_L)))

# --- 1. Standard (Unmasked) Evaluators used by MUSCL & Central ---
@inline _compute_raw_grad(::ConstantReconstruction, nb_slice, dist_all, w_all, df, scale, ::Val{MAX_B_LEN}, ::Type{State{M, T}}) where {MAX_B_LEN, M, T} = zero(SVector{MAX_B_LEN, State{M, T}})

@inline function _compute_raw_grad(interp::Interpolator, nb_slice, dist_all, w_all, df, scale, ::Val{MAX_B_LEN}, ::Type{State{M, T}}) where {MAX_B_LEN, M, T}
    raw = interp(nb_slice, dist_all, w_all, df; scale=scale)
    return _pad_grad(raw, Val(MAX_B_LEN))
end

# --- 2. Masked Evaluators used by WENO & Upwind(Tiwari) ---
@inline _compute_raw_grad(::ConstantReconstruction, nb_slice, dist_all, w_all, df, mask::AbstractVector{Bool}, scale, ::Val{MAX_B_LEN}, ::Type{State{M, T}}) where {MAX_B_LEN, M, T} = zero(SVector{MAX_B_LEN, State{M, T}})

@inline function _compute_raw_grad(interp::Interpolator, nb_slice, dist_all, w_all, df, mask::AbstractVector{Bool}, scale, ::Val{MAX_B_LEN}, ::Type{State{M, T}}) where {MAX_B_LEN, M, T}
    raw = interp(nb_slice, dist_all, w_all, df, mask; scale=scale)
    return _pad_grad(raw, Val(MAX_B_LEN))
end

# --- 3. The Generated Dispatchers ---
@generated function dispatch_interpolator(interps::Tuple, order::Int, args...)
    N = length(interps.parameters)
    expr = :(error("Order out of bounds"))
    for i in N:-1:1
        expr = :(order == $i ? _compute_raw_grad(interps[$i], args...) : $expr)
    end
    return expr
end

@generated function compute_dynamic_divergence(interps::Tuple, div_idx::Int, nb_slice, dist_all, w_all, df_flux, df_scratch, scale)
    N = length(interps.parameters)
    expr = :(interps[$N](nb_slice, dist_all, w_all, df_flux, df_scratch; scale=scale))
    
    for i in (N-1):-1:1
        expr = :(div_idx == $i ? interps[$i](nb_slice, dist_all, w_all, df_flux, df_scratch; scale=scale) : $expr)
    end
    return expr
end

# --- 4. Constant Reconstruction Functor ---
# CRITICAL FIX: Extract M and T securely from the strongly-typed flux array 
# to guarantee a valid 0-state return, regardless of the dimension of `scale`.
@inline function (::ConstantReconstruction)(
    nb_slice, dist_all, w_all, df_flux::AbstractVector{Flux{D, M, T}}, df_scratch; scale
) where {D, M, T}
    return zero(State{M, T})
end

# =========================================================================
# COMPILE-TIME METADATA HELPERS
# =========================================================================

# 1. Compile-time factorial calculators
_ct_factorial(n::Int) = n <= 1 ? 1 : n * _ct_factorial(n - 1)
_ct_multi_factorial(t::Tuple) = prod(_ct_factorial.(t))

# 2. Generates exponents (a_1, a_2, ..., a_D) summing to the required orders
"""
    _generate_exponents(D::Int, max_order::Int)

Generates combinations of exponents for polynomial basis terms up to a specified maximum order.

# Arguments
- `D::Int`: The spatial dimension.
- `max_order::Int`: The maximum polynomial order.

# Returns
- Returns a sorted array of `NTuple{D, Int}` where the sum of each tuple equals the respective polynomial order. The sorting guarantees that index 1 corresponds to X, index 2 to Y, and index 3 to Z.
"""
function _generate_exponents(D::Int, max_order::Int)
    res = NTuple{D, Int}[]
    for order in 1:max_order
        current_order_tuples = NTuple{D, Int}[]
        
        for t in Iterators.product(ntuple(_ -> 0:order, D)...)
            if sum(t) == order
                push!(current_order_tuples, t)
            end
        end
        
        # FIX: Using `reverse(x)` ensures (1, 0) comes before (0, 1).
        # This guarantees that index 1 is X, index 2 is Y, index 3 is Z
        sort!(current_order_tuples, by = x -> (-maximum(x), reverse(x)))
        append!(res, current_order_tuples)
    end
    return res
end

# 3. Dynamic B_LEN Resolver
@generated function basis_length(::Val{D}, ::Val{IO}) where {D, IO}
    len = length(_generate_exponents(D, IO))
    return :(Val($len))
end

"""
    basis_count(D::Int, degree::Int) -> Int

Number of basis terms of degree 1..`degree` in `D` variables, `binomial(D + degree, degree) - 1`, i.e. the
`B_LEN` of `basis_length(Val(D), Val(degree))` for a *runtime* degree. Kernels use it to check neighbour
counts against the particle's current order without a dynamic `Val` dispatch.
"""
@inline function basis_count(D::Int, degree::Int)
    degree <= 0 && return 0
    n = 1
    for k in 1:degree           # binomial(D + degree, degree) without allocation or dispatch
        n = (n * (D + k)) ÷ k
    end
    return n - 1
end

# =========================================================================
# BASIS VECTOR EVALUATORS (Fully Generalized & Type-Stable)
# =========================================================================
"""
    build_basis(::Val{IO}, d::SVector{D, T})

Generates a fully generalized, type-stable polynomial basis vector for moving least squares interpolation.

# Arguments
- `::Val{IO}`: A value type specifying the interpolation order.
- `d::SVector{D, T}`: The scaled distance vector.

# Returns
- An `SVector` containing the computed basis terms, including precomputed Taylor prefactors.
"""
@generated function build_basis(::Val{IO}, d::SVector{D, T}) where {IO, D, T}
    exps = _generate_exponents(D, IO)
    B_LEN = length(exps)
    
    exprs = Any[] # Changed from Expr[] to Any[]
    for t in exps
        denom = _ct_multi_factorial(t)
        
        # Precompute the Taylor prefactor if it is not 1
        term = denom == 1 ? nothing : :(T($(1.0 / denom)))
        
        # Append the polynomial terms
        for i in 1:D
            if t[i] == 1
                term = isnothing(term) ? :(d[$i]) : :($term * d[$i])
            elseif t[i] > 1
                term = isnothing(term) ? :(d[$i]^$(t[i])) : :($term * d[$i]^$(t[i]))
            end
        end
        
        push!(exprs, term)
    end
    
    return :(SVector{$B_LEN, T}($(exprs...)))
end

# =========================================================================
# PHYSICAL SCALE FACTORS
# =========================================================================
"""
    build_scale_factors(::Val{D}, ::Val{IO}, invL::T)

Constructs the physical scale factors for the interpolator based on the inverse length scale.

# Arguments
- `::Val{D}`: The spatial dimension.
- `::Val{IO}`: The interpolation order.
- `invL::T`: The inverse of the characteristic length scale.

# Returns
- An `SVector` of scale factors corresponding to each polynomial degree in the basis.
"""
@generated function build_scale_factors(::Val{D}, ::Val{IO}, invL::T) where {D, IO, T}
    exps = _generate_exponents(D, IO)
    B_LEN = length(exps)
    
    exprs = Any[] # Changed from Expr[] to Any[]
    for t in exps
        deg = sum(t)
        push!(exprs, deg == 1 ? :(invL) : :(invL^$deg))
    end
    
    return :(SVector{$B_LEN, T}($(exprs...)))
end

# =========================================================================
# EPD_1 BASIS TRUNCATION
# =========================================================================
"""
    mask_basis(basis::SVector{B_LEN, T}, order::Int, ::Val{D})

Truncates the basis vector down to a specific spatial order at compile-time.

# Arguments
- `basis::SVector{B_LEN, T}`: The full evaluated basis vector.
- `order::Int`: The spatial order to which the basis should be truncated.
- `::Val{D}`: The spatial dimension.

# Returns
- A masked `SVector` where basis terms exceeding the requested spatial order are strictly zeroed out.
"""
@generated function mask_basis(basis::SVector{B_LEN, T}, order::Int, ::Val{D}) where {B_LEN, D, T}
    max_o = 1
    while length(_generate_exponents(D, max_o)) < B_LEN
        max_o += 1
    end
    
    expr = :(zero(SVector{B_LEN, T}))
    
    # FIX: Shift max_o up by 1 to match the spatial `order` numbering
    max_spatial_order = max_o + 1
    
    for o in max_spatial_order:-1:2 
        len = length(_generate_exponents(D, o - 1))
        
        mask_tuple = ntuple(k -> k <= len ? :(basis[$k]) : :(zero(T)), B_LEN)
        expr = :(order == $o ? SVector{B_LEN, T}($(mask_tuple...)) : $expr)
    end
    
    return expr
end

# =========================================================================
# UPWIND MATRIX-VECTORIZED DISPATCH (Stateless)
# =========================================================================

function (interp::Interpolator{D, IO, DO})(
    nb_slice::OrdinalRange{Int, Int}, dists::AbstractVector{Space{D, T}}, weights::AbstractVector{T},
    dfFluxVec::AbstractVector{Flux{D, M, T}}, dfVec_workspace::AbstractVector{State{M, T}};
    scale::Space{D, T}
) where {D, IO, DO, M, T}
    # The MLS fit is invariant to the (isotropic) coordinate scale up to the relative regularisation,
    # so one scale serves all directions and the workspace argument is kept only for the interface.
    return _mls_divergence(nb_slice, dists, weights, dfFluxVec, minimum(scale),
                           basis_length(Val(D), Val(IO)), Val(IO), Val(D))
end

# One moment matrix and one factorisation for all D flux components (instead of one per direction):
# the right-hand side holds the D·M flux differences side by side and the divergence is Σ_d ∂_d F_d.
#
# The divergence fit has a constant term (an intercept) and runs over the neighbours only: its derivative
# weights then sum to zero exactly, Σ_j c_ij = 0, which removes the row-sum part of the conservation defect
# of `2 Σ_j c_ij·(F*_ij - F_i)` on irregular point clouds (the reconstruction fits keep the Taylor form
# through the particle). See `_intercept_basis`.
@inline function _mls_divergence(
    nb_slice::OrdinalRange{Int, Int}, distVec::AbstractVector{Space{D, T}}, wVec::AbstractVector{T},
    dfFluxVec::AbstractVector{Flux{D, M, T}}, scale::T, ::Val{B0}, ::Val{IO}, ::Val{D}
) where {B0, IO, D, M, T}
    B_LEN = B0 + 1
    invL = one(T) / scale
    N_s = zero(SMatrix{B_LEN, B_LEN, T, B_LEN * B_LEN})
    b_s = zero(SMatrix{B_LEN, D * M, T, B_LEN * D * M})

    @inbounds for i in nb_slice
        w   = wVec[i]
        p_s = _intercept_basis(Val(IO), distVec[i] * invL)
        F   = dfFluxVec[i]
        row = SVector{D * M, T}(ntuple(k -> F[(k - 1) ÷ M + 1][(k - 1) % M + 1], Val(D * M)))
        N_s += w * (p_s * p_s')
        b_s += w * (p_s * row')
    end

    C, ok = _cholesky_moment(N_s)
    ok || return zero(State{M, T})
    c_s = C \ b_s

    # Basis term 1 is the constant, terms 2..D+1 the first derivatives in axis order (scaled by invL).
    return State{M, T}(ntuple(Val(M)) do m
        acc = zero(T)
        for d in 1:D
            acc += c_s[d + 1, (d - 1) * M + m]
        end
        acc * invL
    end)
end

"`[1; build_basis(Val(IO), d)]`: the Taylor basis with a constant term (divergence fits)."
@inline _intercept_basis(::Val{IO}, d::SVector{D, T}) where {IO, D, T} = vcat(SVector{1, T}(one(T)), build_basis(Val(IO), d))

# =========================================================================
# THE UNIVERSAL MLS INTERPOLATOR
# =========================================================================

function (interp::Interpolator{D, IO, 1})(
    nb_slice::OrdinalRange{Int, Int},
    distVec::AbstractVector{Space{D, T}}, 
    wVec::AbstractVector{T},
    dfVec::AbstractVector{State{M, T}};
    scale::T=one(T)
) where {D, IO, M, T}
    
    B_LEN_VAL = basis_length(Val(D), Val(IO))
    return _mls_solve(nb_slice, distVec, wVec, dfVec, scale, B_LEN_VAL, Val(IO), Val(D))
end

@inline function _mls_solve(
    nb_slice::OrdinalRange{Int, Int}, distVec::AbstractVector, wVec::AbstractVector,
    dfVec::AbstractVector{State{M, T}}, scale::T,
    ::Val{B_LEN}, ::Val{IO}, ::Val{D}
) where {B_LEN, IO, D, M, T}
    
    invL = one(T) / scale
    N_s = zero(SMatrix{B_LEN, B_LEN, T, B_LEN * B_LEN})
    b_s = zero(SMatrix{B_LEN, M, T, B_LEN * M})

    @inbounds for i in nb_slice
        w = wVec[i]
        p_s = build_basis(Val(IO), distVec[i] * invL) 
        
        N_s += w * (p_s * p_s') 
        b_s += w * (p_s * dfVec[i]') 
    end
    
    C, ok = _cholesky_moment(N_s)                 # regularised only if the plain factorisation fails
    ok || return zero(SVector{B_LEN, State{M, T}})

    c_s = C \ b_s
    scales = build_scale_factors(Val(D), Val(IO), invL)
    
    return SVector{B_LEN, State{M, T}}(ntuple(Val(B_LEN)) do k
        row_val = State{M, T}(ntuple(m -> c_s[k, m], Val(M)))
        row_val * scales[k]
    end)
end

function (interp::Interpolator{D, IO, 1})(
    nb_slice::OrdinalRange{Int, Int}, distVec::AbstractVector{Space{D, T}}, 
    wVec::AbstractVector{T}, dfVec::AbstractVector{State{M, T}},
    mask::AbstractVector{Bool}; scale::T=one(T)
) where {D, IO, M, T}
    
    B_LEN_VAL = basis_length(Val(D), Val(IO))
    return _mls_solve_masked(nb_slice, distVec, wVec, dfVec, mask, scale, B_LEN_VAL, Val(IO), Val(D))
end

@inline function _mls_solve_masked(
    nb_slice::OrdinalRange{Int, Int}, distVec::AbstractVector{Space{D, T}}, 
    wVec::AbstractVector{T}, dfVec::AbstractVector{State{M, T}},
    mask::AbstractVector{Bool}, scale::T,
    ::Val{B_LEN}, ::Val{IO}, ::Val{D}
) where {B_LEN, IO, D, M, T}
    
    invL = one(T) / scale
    N_s = zero(SMatrix{B_LEN, B_LEN, T, B_LEN * B_LEN})
    b_s = zero(SMatrix{B_LEN, M, T, B_LEN * M})

    @inbounds for i in nb_slice
        if !mask[i]; continue; end
        
        w = wVec[i]
        p_s = build_basis(Val(IO), distVec[i] * invL)
        N_s += w * (p_s * p_s')
        b_s += w * (p_s * dfVec[i]')
    end

    # Same conditional regularisation as the unmasked solve (one-sided stencils are degenerate more often)
    C, ok = _cholesky_moment(N_s)
    ok || return zero(SVector{B_LEN, State{M, T}})

    c_s = C \ b_s
    scales = build_scale_factors(Val(D), Val(IO), invL)
    
    return SVector{B_LEN, State{M, T}}(ntuple(Val(B_LEN)) do k
        row_val = State{M, T}(ntuple(m -> c_s[k, m], Val(M)))
        row_val * scales[k]
    end)
end

# =========================================================================
# CACHED MLS MOMENT MATRICES (frozen geometry)
# =========================================================================
"""
    MLSWeights{B_LEN, IO, T}(storage = Array)

Per-particle inverse moment matrices of the degree-`IO` MLS fit for a frozen geometry (`Growable` buffers
with arrays of type `storage`). With them a fit over
the neighbour differences `df_k` is `inv_N * Σ_k w_k p_k df_kᵀ` (basis `p_k` of the scaled distance), which
reproduces `Interpolator{D, IO, 1}` without assembling and factorising the moment matrix for every solve.
Only `B_LEN²` numbers per particle are stored, so the neighbour-slot traffic of the hot loops does not grow
(a per-slot weight cache was slower once the working set exceeded the caches).

`ok[i]` is `false` where the moment matrix was not positive definite (callers use the direct solve there,
which returns zero). `version` is the `geometry_version` the matrices belong to; see [`update_weights!`](@ref).
"""
abstract type AbstractMLSWeights{B_LEN, IO, T} end

# `C`: the fit has a constant term (`_intercept_basis`, divergence fits); `B_LEN` then counts it.
struct MLSWeights{B_LEN, IO, T, L, GM <: Growable{SMatrix{B_LEN, B_LEN, T, L}}, GB <: Growable{Bool}, C} <: AbstractMLSWeights{B_LEN, IO, T}   # L = B_LEN^2 (StaticArrays needs it)
    inv_moment::GM
    ok::GB
    version::Base.RefValue{Int}
    invL::Base.RefValue{T}
    scales::Base.RefValue{SVector{B_LEN, T}}
end
function MLSWeights{B_LEN, IO, T}(::Type{AT} = Array; intercept::Bool = false) where {B_LEN, IO, T, AT <: AbstractArray}
    inv_moment = Growable{SMatrix{B_LEN, B_LEN, T, B_LEN * B_LEN}}(AT)
    ok         = Growable{Bool}(AT)
    return MLSWeights{B_LEN, IO, T, B_LEN * B_LEN, typeof(inv_moment), typeof(ok), intercept}(inv_moment, ok, Ref(-1),
                                                                                              Ref(one(T)), Ref(zero(SVector{B_LEN, T})))
end

"""
    MLSWeightsView{B_LEN, IO, T, L, VM, VB}

The bits-type form of `MLSWeights` that enters kernels (see `DeviceAdapt.jl`): device arrays and plain values
instead of `Ref`s.
"""
struct MLSWeightsView{B_LEN, IO, T, L, VM <: AbstractVector{SMatrix{B_LEN, B_LEN, T, L}}, VB <: AbstractVector{Bool}, C} <: AbstractMLSWeights{B_LEN, IO, T}
    inv_moment::VM
    ok::VB
    version::Int
    invL::T
    scales::SVector{B_LEN, T}
end

"`true` if the cached matrices belong to the current geometry of `pg`."
@inline is_current(w::MLSWeights, pg::ParticleGrid) = w.version[] == pg.meta.geometry_version
@inline is_current(w::MLSWeightsView, pg::ParticleGrid) = w.version == pg.meta.geometry_version
@inline _cache_invL(w::MLSWeights) = w.invL[]
@inline _cache_invL(w::MLSWeightsView) = w.invL
@inline _cache_scales(w::MLSWeights) = w.scales[]
@inline _cache_scales(w::MLSWeightsView) = w.scales
@inline _has_intercept(::MLSWeights{B, IO, T, L, GM, GB, C}) where {B, IO, T, L, GM, GB, C} = C
@inline _has_intercept(::MLSWeightsView{B, IO, T, L, VM, VB, C}) where {B, IO, T, L, VM, VB, C} = C
@inline _fit_basis(::Val{false}, ::Val{IO}, d) where {IO} = build_basis(Val(IO), d)
@inline _fit_basis(::Val{true}, ::Val{IO}, d) where {IO} = _intercept_basis(Val(IO), d)

"""
    update_weights!(w::MLSWeights, pg::ParticleGrid)

Recomputes the inverse moment matrices for the current neighbour lists, distances and weights of `pg` unless
they are current already. One MLS factorisation per particle (parallel).
"""
function update_weights!(w::MLSWeights{B_LEN, IO, T}, pg::ParticleGrid{D, M, T}) where {B_LEN, IO, T, D, M}
    is_current(w, pg) && return w
    N  = pg.meta.N
    nb = pg.neighbor
    ensure_capacity!(w.inv_moment, N)
    ensure_capacity!(w.ok, N)
    invL = one(T) / minimum(pg.meta.dx)
    C = Val(_has_intercept(w))
    foreach_particle(_moment_body!, get_backend(), N, (w.inv_moment, w.ok, nb.ranges, nb.distances, nb.weights, invL, Val(IO), C))
    w.invL[]    = invL
    s = build_scale_factors(Val(D), Val(IO), invL)
    w.scales[]  = _has_intercept(w) ? vcat(SVector{1, T}(one(T)), s) : s
    w.version[] = pg.meta.geometry_version
    return w
end

# Kernel body (see Backend.jl): inverse moment matrix of particle `i`.
@inline function _moment_body!(i, inv_moment, ok, ranges, dists, wts, invL, val_io, val_c)
    inv_moment[i], ok[i] = _inverse_moment(ranges[i], dists, wts, invL, val_io, inv_moment[i], val_c)
    return nothing
end

# Inverse of the moment matrix of one particle (the same fit as `_mls_solve`, regularised only if the stencil
# is degenerate); `template` only fixes the return type when the factorisation fails.
function _inverse_moment(nb_slice::OrdinalRange{Int, Int}, distVec::AbstractVector, wVec::AbstractVector{T}, invL::T,
                         ::Val{IO}, template::SMatrix{B_LEN, B_LEN, T}, val_c::Val = Val(false)) where {T, IO, B_LEN}
    N_s = zero(SMatrix{B_LEN, B_LEN, T, B_LEN * B_LEN})
    @inbounds for k in nb_slice
        p_s = _fit_basis(val_c, Val(IO), distVec[k] * invL)
        N_s += wVec[k] * (p_s * p_s')
    end
    C, ok = _cholesky_moment(N_s)
    ok || return (zero(template), false)
    return (SMatrix{B_LEN, B_LEN, T, B_LEN * B_LEN}(inv(C)), true)
end

"""
    apply_weights(w::MLSWeights, i, nb_slice, distVec, wVec, dfVec) -> SVector{B_LEN, State{M, T}}

The MLS coefficients of the differences `dfVec` over `nb_slice` of particle `i` from the cached inverse moment
matrix (replaces the factorisation of `Interpolator{D, IO, 1}`).
"""
@inline function apply_weights(w::AbstractMLSWeights{B_LEN, IO, T}, i::Int, nb_slice::OrdinalRange{Int, Int},
                               distVec::AbstractVector, wVec::AbstractVector{T},
                               dfVec::AbstractVector{State{M, T}}) where {B_LEN, IO, T, M}
    invL = _cache_invL(w)
    b_s = zero(SMatrix{B_LEN, M, T, B_LEN * M})
    @inbounds for k in nb_slice
        p_s = build_basis(Val(IO), distVec[k] * invL)
        b_s += wVec[k] * (p_s * dfVec[k]')
    end
    return _scaled_rows(w.inv_moment[i] * b_s, _cache_scales(w))
end

# Rows of the coefficient matrix as States, scaled to physical derivatives (as in `_mls_solve`).
@inline function _scaled_rows(c_s::SMatrix{B_LEN, M, T}, scales::SVector{B_LEN, T}) where {B_LEN, M, T}
    return SVector{B_LEN, State{M, T}}(ntuple(Val(B_LEN)) do b
        State{M, T}(ntuple(m -> c_s[b, m], Val(M))) * scales[b]
    end)
end

"""
    apply_weights_divergence(w::MLSWeights, i, nb_slice, distVec, wVec, dfFluxVec) -> State{M, T}

`Σ_d ∂_d F_d` of the flux differences `dfFluxVec` over `nb_slice` of particle `i` from the cached inverse
moment matrix (replaces the factorisation in `_mls_divergence`).
"""
@inline function apply_weights_divergence(w::AbstractMLSWeights{B_LEN, IO, T}, i::Int, nb_slice::OrdinalRange{Int, Int},
                                          distVec::AbstractVector, wVec::AbstractVector{T},
                                          dfFluxVec::AbstractVector{Flux{D, M, T}}) where {B_LEN, IO, T, D, M}
    invL = _cache_invL(w)
    C    = _has_intercept(w)
    b_s = zero(SMatrix{B_LEN, D * M, T, B_LEN * D * M})
    @inbounds for k in nb_slice
        p_s = _fit_basis(Val(C), Val(IO), distVec[k] * invL)
        F   = dfFluxVec[k]
        row = SVector{D * M, T}(ntuple(q -> F[(q - 1) ÷ M + 1][(q - 1) % M + 1], Val(D * M)))
        b_s += wVec[k] * (p_s * row')
    end
    c_s = w.inv_moment[i] * b_s
    off = C ? 1 : 0                                    # the constant term comes first
    return State{M, T}(ntuple(Val(M)) do m
        acc = zero(T)
        for d in 1:D
            acc += c_s[d + off, (d - 1) * M + m]
        end
        acc * invL
    end)
end

include("FluxFunctions.jl")

include("Central.jl")

include("MUSCL.jl")
include("Limiter.jl")
include("Conservation.jl")

include("Upwind.jl")

include("WENO.jl")
