export MUSCL

# =========================================================================
# DIVERGENCE ORDER DISPATCH
# =========================================================================
# Interpolator-table entry (fit degree + 1) of the divergence for consistency order `o`. On (nearly)
# symmetric stencils the derivative of a least-squares fit of degree q is O(h^q) for even q (the next
# monomial d^(q+1) is odd and aliases into the gradient) but O(h^(q+1)) for odd q (the next monomial is
# even and orthogonal to it). Degree o - 1 therefore gives order o for even o only; odd orders use degree o
# (MUSCL3: 2nd → 3rd order, MUSCL5: 4th → 5th). In 3-D the bump stops at degree 3 (basis size).
@inline function _div_idx(::Val{D}, o::Int) where {D}
    o <= 2 && return 2
    return (isodd(o) && (D < 3 || o <= 3)) ? o + 1 : o
end
@inline _resolve_div_idx(::Val{0}, ::Val{D}, p_order) where {D} = _div_idx(Val(D), p_order)
@inline _resolve_div_idx(::Val{DO}, ::Val{D}, p_order) where {DO, D} = _div_idx(Val(D), DO)

# The divergence entry of a particle with `num_nb` neighbours. The fit of degree q (with its constant term)
# has basis_count(D, q) + 1 unknowns. The odd-order degree bump is only used on stencils with at least 1.5×
# as many neighbours (an exactly determined fit is fragile on moving grids: 1-D, R = 2.5Δx has 4 neighbours
# for the 4 unknowns of degree 3); any fit needs at least as many neighbours as unknowns.
@inline function _divergence_index(::Val{DO}, ::Val{D}, p_order::Int, num_nb::Int) where {DO, D}
    idx  = _resolve_div_idx(Val(DO), Val(D), p_order)
    base = max(2, DO == 0 ? p_order : DO)                         # entry without the odd-order bump
    if idx > base && 2 * num_nb < 3 * (basis_count(D, idx - 1) + 1)
        idx = base
    end
    while idx > 2 && num_nb < basis_count(D, idx - 1) + 1
        idx -= 1
    end
    return idx
end

@inline function (::ConstantReconstruction)(
    nb_slice, dist_all, w_all, df_flux, df_scratch; scale::T
) where {T}
    return zero(T)
end

# =========================================================================
# MUSCL STRUCTURE
# =========================================================================

"""
    MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, INTERPS, L, NF} <: DivergenceInterpolator
    MUSCL(::Type{T}, dimension::Int, M::Int, max_order::Int, limiter, flux; div_order::Int=0)
    (muscl::MUSCL)(eq, i, f_i, nb_slice, pg, ib)

A divergence interpolator executing MUSCL-type interface reconstruction and flux evaluation. It dynamically tracks particle polynomial limits strictly via the underlying grid core.

# Constructors

    MUSCL(::Type{T}, dimension::Int, M::Int, max_order::Int, limiter, flux; div_order::Int=0)

- `T`: The numeric type (e.g., `Float64`).
- `dimension`: Spatial dimension.
- `M`: Number of state components.
- `max_order`: Maximum consistency order; the reconstruction has degree `max_order - 1.
- `limiter`: The slope/gradient limiter applied during the pre-gather pass to enforce monotonicity.
- `flux`: The numerical flux function used to resolve interface states.
- `div_order`: Target order for the final divergence calculation (defaults to `0`, adapting to the maximum available order). The divergence is fitted with degree `order - 1` for even and degree `order` for odd orders (see `_div_idx`), with a constant term so that its weights sum to zero.
- `streamed` (keyword, default `false`): accumulate the divergence in the flux pass with cached per-slot weights instead of buffering the flux differences and fitting them afterwards; same numbers up to round-off, no per-slot flux buffer, and the per-neighbour cost no longer grows with the number of fit terms (see [Streamed divergence](@ref)).

# Callable / Functor

    (muscl::MUSCL)(eq, i, f_i, nb_slice, pg, ib, strat) -> State{M, T}

Computes the divergence of the flux at particle `i`. Note that `MUSCL` relies on a pre-calculation pass (`update_content!`) to compute and limit the local gradients beforehand. The execution follows these steps:
- Retrieves the pre-calculated, limited gradients for particle `i` and its neighbors.
- Evaluates the reconstructed states at the midpoint between particles using distance-scaled spatial basis polynomials.
- Interface degrees are chosen by the MOOD strategy via `_get_interface_orders` (EPD0/StrictEPD0: own order per side; EPD1: edge minimum; EPD2: neighbourhood minimum).
- Resolves the interface using the provided numerical flux function and evaluates non-conservative jumps.
- Computes the final divergence using the dynamically resolved divergence order.

# Fields
- `interpolators`: Pre-allocated tuple of interpolators up to `MAX_ORDER`, using `ConstantReconstruction` for the first-order base.
- `limiter`: The assigned slope limiter.
- `flux`: The associated numerical flux function (`NF`).
- `gradients`: A pre-allocated vector storing the reconstructed, limited spatial gradients for all particles.
- `grad_weights`, `div_weights`: cached inverse MLS moment matrices (`MLSWeights`) of the full-order reconstruction and divergence fits, refreshed by `prepare_geometry!` when the grid geometry changes.
- `dweights`: per-slot derivative weights of the full-degree divergence fit, used by the streamed divergence and, through their signs, by [`AlignedRusanovFlux`](@ref) (refreshed with the divergence cache; empty if neither is used).
- `streamed`: `Val(true)` if the flux pass accumulates the divergence with `dweights` instead of writing the flux differences to the per-slot buffer and fitting them afterwards (keyword `streamed` of the constructor, see [Streamed divergence](@ref)).

The per-particle caches are `Growable` buffers whose array type is the `storage` keyword of the constructor
(`Array` by default; `ROCArray` etc. for a grid built with `device_grid`).
"""
struct MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, INTERPS, L, NF, VG <: AbstractVector{SVector{B_LEN, State{M, T}}}, WG, WD,
             VS <: AbstractVector{Space{D, T}}, SF <: Val} <: DivergenceInterpolator
    interpolators::INTERPS
    limiter::L
    flux::NF
    gradients::VG
    # Cached inverse MLS moment matrices for a frozen geometry (see `MLSWeights`): the full-order reconstruction and the
    # divergence fit. They are the same object when both use the same degree. Particles at reduced (MOOD)
    # order fall back to the direct solves.
    grad_weights::WG
    div_weights::WD
    # Derivative weights c_ij of the full-degree divergence fit, one per neighbour slot (physical units, so
    # 2 V_i c_ij is the face vector): the streamed divergence accumulates Σ_j c_ij · ΔF_ij with them and
    # `AlignedRusanovFlux` sorts by their signs. Filled with the divergence cache (`update_divergence_weights!`),
    # empty when neither is used.
    dweights::VS
    # `Val(true)`: the flux pass accumulates the divergence with the per-slot weights (no per-slot flux
    # buffer, no fit after the loop); `Val(false)`: the flux differences are written per slot and fitted.
    streamed::SF
end

# Interpolator-table entry the divergence uses at full order (see `_resolve_div_idx`).
@inline _full_div_idx(::Val{MAX_ORDER}, ::Val{DIV_ORDER}, ::Val{D}) where {MAX_ORDER, DIV_ORDER, D} =
    _div_idx(Val(D), DIV_ORDER == 0 ? MAX_ORDER : DIV_ORDER)

@inline _extract_order(::MUSCL{D, M, T, B_LEN, MAX_ORDER}) where {D, M, T, B_LEN, MAX_ORDER} = MAX_ORDER

function MUSCL(
    ::Type{T}, dimension::Int, M::Int, max_order::Int,
    limiter, flux;
    div_order::Int=0, storage::Type{AT} = Array, streamed::Bool = false
) where {T, AT <: AbstractArray}
    @assert max_order >= 1 "MUSCL must have a maximum order of at least 1."
    @assert div_order >= 0 "Divergence order only supports 0 (adaptive) or positive values!"
    
    if div_order > max_order; div_order = max_order; end
    
    max_degree = max(1, max_order - 1)
    B_LEN_VAL = basis_length(Val(dimension), Val(max_degree))
    B_LEN = typeof(B_LEN_VAL).parameters[1]

    # Entry k reconstructs with degree k - 1 (entry 1: piecewise constant). The divergence always uses
    # an entry ≥ 2 (`_resolve_div_idx`), so a first-order scheme still needs the degree-1 interpolator
    # for its divergence; without it the flux divergence was identically zero. Odd orders fit the
    # divergence one degree above the reconstruction (`_div_idx`), so the table can be one entry longer.
    full_div = _full_div_idx(Val(max_order), Val(div_order), Val(dimension))
    interps = ntuple(Val(max(2, max_order, full_div))) do k
        k == 1 ? ConstantReconstruction() : Interpolator{dimension, k - 1, 1}()
    end

    # Cached weights: the reconstruction degree at full order and the divergence degree at full order (the
    # divergence fit has a constant term, so it never shares the reconstruction matrices).
    div_degree = full_div - 1
    grad_weights = MLSWeights{B_LEN, max_degree, T}(AT)
    div_weights  = MLSWeights{typeof(basis_length(Val(dimension), Val(div_degree))).parameters[1] + 1, div_degree, T}(AT; intercept = true)

    return MUSCL{dimension, M, T, B_LEN, max_order, div_order}(interps, limiter, flux, Growable{SVector{B_LEN, State{M, T}}}(AT),
                                                              grad_weights, div_weights, Growable{Space{dimension, T}}(AT), Val(streamed))
end

# Infers the storage types (host vectors, or device views when adapted for a kernel).
MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER}(interps, limiter, flux, gradients, grad_weights, div_weights, dweights, streamed) where {D, M, T, B_LEN, MAX_ORDER, DIV_ORDER} =
    MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, typeof(interps), typeof(limiter), typeof(flux), typeof(gradients),
          typeof(grad_weights), typeof(div_weights), typeof(dweights), typeof(streamed)}(interps, limiter, flux, gradients, grad_weights, div_weights, dweights, streamed)

"`true` if the scheme accumulates its divergence in the flux pass (keyword `streamed`)."
@inline _streamed(m::MUSCL) = m.streamed isa Val{true}

# A streamed scheme never touches the per-slot flux buffers of the stepper.
@inline uses_slot_buffers(m::MUSCL) = !_streamed(m)

function prepare_geometry!(muscl::MUSCL, pg::ParticleGrid)
    update_weights!(muscl.grad_weights, pg)
    wd = muscl.div_weights
    v0 = wd.version[]
    wd === muscl.grad_weights || update_weights!(wd, pg)
    if (_streamed(muscl) || _aligned_dissipation(muscl.flux)) &&
       (v0 != wd.version[] || length(muscl.dweights) < length(pg.neighbor.indices))
        update_divergence_weights!(muscl, pg)
    end
    return nothing
end

"""
    update_divergence_weights!(muscl::MUSCL, pg::ParticleGrid)

The derivative weights of the full-degree divergence fit for every neighbour slot (`dweights`), from the
current divergence cache: rows 2..D+1 of `inv(N) w_j p̃_j`, scaled to physical units. Zero where the cache
has no factorisation for the particle (such a particle takes a dynamic fit, whose weights are solved per slot).
Called by `prepare_geometry!` when the cache was refreshed, for streamed schemes and the aligned flux.
"""
function update_divergence_weights!(muscl::MUSCL{D, M, T}, pg::ParticleGrid{D, M, T}) where {D, M, T}
    wd = muscl.div_weights
    nb = pg.neighbor
    ensure_capacity!(muscl.dweights, length(nb.indices))
    foreach_particle(_weights_body!, get_backend(), pg.meta.N,
        (muscl.dweights, wd.inv_moment, wd.ok, nb.ranges, nb.distances, nb.weights, _cache_invL(wd),
         _fit_degree(wd), Val(_has_intercept(wd)), Val(D)))
    return nothing
end

@inline _fit_degree(::AbstractMLSWeights{B, IO, T}) where {B, IO, T} = Val(IO)

# Kernel body (see Backend.jl): per-slot derivative weights of particle `i` (D·B flops per slot with the
# derivative rows of the cached inverse).
@inline function _weights_body!(i, dweights, inv_moment, ok, ranges, dists, wts, invL, val_io, val_c::Val{C}, valD::Val{D}) where {C, D}
    if ok[i]
        R = _derivative_rows(inv_moment[i], valD, Val(C ? 1 : 0))
        @inbounds for k in ranges[i]
            dweights[k] = (R * (wts[k] * _fit_basis(val_c, val_io, dists[k] * invL))) * invL
        end
    else
        @inbounds for k in ranges[i]
            dweights[k] = zero(eltype(dweights))
        end
    end
    return nothing
end

# The direction that sorts the interface states of slot `k`: the distance for every flux but
# `AlignedRusanovFlux`, which uses the signs of `w_k`, the divergence weights the slot's flux difference is
# multiplied with, on the pairs where either particle is below the full order and on every pair of a
# first-order scheme (K = 1) (`below`). The rule is symmetric in the pair, so both sides evaluate the same
# flux; elsewhere the geometric sorting keeps the error constant of the plain flux.
@inline _dissipation_direction(::NumericalFluxFunction, w_k, dist_k, below::Bool) = dist_k
@inline _dissipation_direction(::AlignedRusanovFlux, w_k, dist_k, below::Bool) = below ? sign.(w_k) : dist_k

function update_size!(muscl::MUSCL{D, M, T, B_LEN, MAX_ORDER}, N_particles::Int) where {D, M, T, B_LEN, MAX_ORDER}
    ensure_capacity!(muscl.gradients, N_particles)
    fill!(muscl.gradients, zero(eltype(muscl.gradients)))
    return nothing
end

# =========================================================================
# PRE-GATHER PASS: CALCULATE AND STORE GRADIENTS
# =========================================================================

function update_content!(
    muscl::MUSCL{D, M, T, B_LEN, MAX_ORDER}, i::Int, f_i::State{M, T}, nb_slice::OrdinalRange{Int, Int},
    pg::ParticleGrid{D, M, T}, U::AbstractVector{State{M, T}}, ib::InteractionBuffer{D, M, T}
) where {D, M, T, B_LEN, MAX_ORDER}
    df = neighbour_differences(U, pg, f_i)
    num_nb = length(nb_slice)
    p_order = pg.core.particle_orders[i]

    req_deg = p_order - 1
    req_nb = basis_count(D, req_deg)
    
    # Dynamically clamp spatial order in core arrays if neighborhood is starved
    if num_nb < req_nb
        pg.core.particle_orders[i] = 1 
        p_order = 1
    end

    dist_all = get_distances(pg)
    w_all = get_weights(pg)

    wg = muscl.grad_weights
    raw_grad = if p_order == MAX_ORDER && p_order > 1 && is_current(wg, pg) && wg.ok[i]
        apply_weights(wg, i, nb_slice, dist_all, w_all, df)   # cached full-order fit
    else
        scale_val = minimum(pg.meta.dx)
        dispatch_interpolator(muscl.interpolators, p_order, nb_slice, dist_all, w_all, df, scale_val, Val(B_LEN), State{M, T})
    end

    if p_order == 1
        muscl.gradients[i] = raw_grad
    else
        muscl.gradients[i] = _limit_slopes(muscl.limiter, raw_grad, nb_slice, f_i, neighbour_states(U, pg), pg, dist_all, Val(MAX_ORDER - 1))
    end
    return nothing
end

# =========================================================================
# FLUX PASS: RECONSTRUCT INTERFACES AND COMPUTE DIVERGENCE
# =========================================================================

function (muscl::MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER})(
    eq::HyperbolicPDE, i::Int, f_i::State{M, T}, nb_slice::OrdinalRange{Int, Int},
    pg::ParticleGrid{D, M, T}, U::AbstractVector{State{M, T}}, ib::InteractionBuffer{D, M, T}, strategy::MOODStrategy
) where {D, M, T, B_LEN, MAX_ORDER, DIV_ORDER}

    if isempty(nb_slice)
        return zero(State{M, T})
    end

    eff_order_i = pg.core.particle_orders[i]
    div_idx = _divergence_index(Val(DIV_ORDER), Val(D), eff_order_i, length(nb_slice))
    wd      = muscl.div_weights
    cached  = is_current(wd, pg) && wd.ok[i] && div_idx == _full_div_idx(Val(MAX_ORDER), Val(DIV_ORDER), Val(D))

    # Without the cached fit the aligned flux needs the weights of the dynamic fit in every slot, so it takes the
    # per-slot path also when the scheme is buffered (same fit as `compute_dynamic_divergence`).
    div = if _streamed(muscl) && cached
        _flux_loop(muscl, eq, i, f_i, nb_slice, pg, U, ib, strategy, eff_order_i, CachedWeights())
    elseif (_streamed(muscl) || _aligned_dissipation(muscl.flux)) && !cached
        _dispatch_streamed_dynamic(div_idx, Val(_full_div_idx(Val(MAX_ORDER), Val(DIV_ORDER), Val(D))),
                                   muscl, eq, i, f_i, nb_slice, pg, U, ib, strategy, eff_order_i)
    else
        _flux_loop(muscl, eq, i, f_i, nb_slice, pg, U, ib, strategy, eff_order_i, BufferedFlux())
        if cached
            apply_weights_divergence(wd, i, nb_slice, get_distances(pg), get_weights(pg), ib.df_flux)   # cached full-order fit
        else
            compute_dynamic_divergence(muscl.interpolators, div_idx, nb_slice, get_distances(pg), get_weights(pg),
                                       ib.df_flux, ib.df_scratch, pg.meta.dx)
        end
    end

    return T(2.0) * div
end

# What the flux loop does with the flux difference ΔF_ij = F*_ij - F_i + jump of a slot: write it to the
# per-slot buffer for the fit afterwards, or accumulate the divergence Σ_j c_ij · ΔF_ij with per-slot weights
# c_ij (cached, or solved from the Cholesky factor of the particle's own moment matrix of degree q).
# `_slot_weight` gives c_ij once per slot, for the contraction and the aligned orientation. A buffered slot
# is only aligned on the cached fit (the dynamic one takes the per-slot path), whose weights are `dweights`.
struct BufferedFlux end
struct CachedWeights end
struct DynamicWeights{q, C, T}
    chol::C
    invL::T
end
DynamicWeights{q}(chol::C, invL::T) where {q, C, T} = DynamicWeights{q, C, T}(chol, invL)

@inline _slot_weight(::BufferedFlux, muscl::MUSCL{D, M, T}, k, dist_k, pg) where {D, M, T} =
    _aligned_dissipation(muscl.flux) ? (@inbounds muscl.dweights[k]) : zero(Space{D, T})
@inline _slot_weight(::CachedWeights, muscl, k, dist_k, pg) = @inbounds muscl.dweights[k]
@inline function _slot_weight(dw::DynamicWeights{q}, muscl::MUSCL{D, M, T}, k, dist_k, pg) where {q, D, M, T}
    c = dw.chol \ ((@inbounds get_weights(pg)[k]) * _intercept_basis(Val(q), dist_k * dw.invL))
    return Space{D, T}(ntuple(d -> c[d + 1], Val(D))) * dw.invL
end

@inline function _consume(::BufferedFlux, ib, k, w_k, dF::Flux{D, M, T}) where {D, M, T}
    @inbounds ib.df_flux[k] = dF
    return zero(State{M, T})
end
@inline _consume(::Union{CachedWeights, DynamicWeights}, ib, k, w_k, dF::Flux) = _contract(w_k, dF)
@inline function _contract(w::Space{D, T}, F::Flux{D, M, T}) where {D, M, T}
    acc = zero(State{M, T})
    for d in 1:D
        acc += w[d] * F[d]
    end
    return acc
end

# The pass over the neighbours of particle `i`: interface reconstructions, numerical flux and non-conservative
# jump per slot, then `_consume` for the flux difference. Returns the accumulated divergence (zero when the
# flux differences are buffered).
@inline function _flux_loop(muscl::MUSCL{D, M, T, B_LEN, MAX_ORDER}, eq, i, f_i::State{M, T}, nb_slice, pg, U, ib, strategy,
                            eff_order_i, how) where {D, M, T, B_LEN, MAX_ORDER}
    grad_i     = muscl.gradients[i]
    F_i        = flux(eq, f_i)
    dist_all   = get_distances(pg)
    nb_indices = pg.neighbor.indices
    orders     = pg.core.particle_orders
    degree     = Val(max(1, MAX_ORDER - 1))   # reconstruction degree; matches B_LEN of the cached gradients
    acc        = zero(State{M, T})

    @inbounds for global_idx in nb_slice
        dist_k = dist_all[global_idx]
        j_idx  = nb_indices[global_idx]
        f_j    = U[j_idx]
        grad_j = muscl.gradients[j_idx]

        p_interface_i_raw = build_basis(degree,  T(0.5) * dist_k)
        p_interface_j_raw = build_basis(degree, -T(0.5) * dist_k)

        # Interface degrees resolved by the MOOD strategy (EPD0 / EPD1 / EPD2 / ...)
        interface_order_i, interface_order_j = _get_interface_orders(strategy, pg, i, j_idx)

        p_interface_i = mask_basis(p_interface_i_raw, interface_order_i, Val(D))
        p_interface_j = mask_basis(p_interface_j_raw, interface_order_j, Val(D))

        fij = f_i + sum(grad_i .* p_interface_i)
        fji = f_j + sum(grad_j .* p_interface_j)

        F_ij = flux(eq, fij)
        F_ji = flux(eq, fji)

        w_k   = _slot_weight(how, muscl, global_idx, dist_k, pg)
        below = MAX_ORDER == 1 || eff_order_i < MAX_ORDER || orders[j_idx] < MAX_ORDER   # K = 1: every edge
        f_L, f_R, F_L, F_R = sort_flux(fij, fji, F_ij, F_ji, _dissipation_direction(muscl.flux, w_k, dist_k, below))
        F_num   = muscl.flux(f_L, f_R, F_L, F_R, eq)
        nc_jump = evaluate_nc_jump(eq, f_L, f_R, dist_k)

        acc += _consume(how, ib, global_idx, w_k, F_num - F_i + nc_jump)
    end
    return acc
end

# Streamed divergence of a particle whose fit is not the cached full-degree one (a reduced order, or no
# factorisation in the cache), for streamed schemes and the aligned flux: the moment matrix of its degree is
# assembled and factorised once, and every slot's weights are solved inside the flux loop. The per-slot buffer
# is never used. A singular moment matrix gives a zero divergence, as in `compute_dynamic_divergence`.
@generated function _dispatch_streamed_dynamic(div_idx::Int, ::Val{IMAX}, muscl, eq, i, f_i, nb_slice, pg, U, ib, strategy, eff_order_i) where {IMAX}
    expr = :(_streamed_dynamic(Val($(IMAX - 1)), muscl, eq, i, f_i, nb_slice, pg, U, ib, strategy, eff_order_i))
    for idx in (IMAX - 1):-1:2
        expr = :(div_idx == $idx ? _streamed_dynamic(Val($(idx - 1)), muscl, eq, i, f_i, nb_slice, pg, U, ib, strategy, eff_order_i) : $expr)
    end
    return expr
end

@inline _streamed_dynamic(::Val{q}, muscl::MUSCL{D, M, T}, args...) where {q, D, M, T} =
    _streamed_dynamic_b(basis_length(Val(D), Val(q)), Val(q), muscl, args...)

@inline function _streamed_dynamic_b(::Val{B0}, ::Val{q}, muscl::MUSCL{D, M, T}, eq, i, f_i, nb_slice, pg, U, ib, strategy,
                                     eff_order_i) where {B0, q, D, M, T}
    B    = B0 + 1
    invL = one(T) / minimum(pg.meta.dx)
    dists, wts = get_distances(pg), get_weights(pg)
    N_s = zero(SMatrix{B, B, T, B * B})
    @inbounds for k in nb_slice
        p = _intercept_basis(Val(q), dists[k] * invL)
        N_s += wts[k] * (p * p')
    end
    C, ok = _cholesky_moment(N_s)
    ok || return zero(State{M, T})
    return _flux_loop(muscl, eq, i, f_i, nb_slice, pg, U, ib, strategy, eff_order_i, DynamicWeights{q}(C, invL))
end

include("MUSCL_hooks.jl")