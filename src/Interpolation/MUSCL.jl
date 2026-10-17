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

The per-particle caches are `Growable` buffers whose array type is the `storage` keyword of the constructor
(`Array` by default; `ROCArray` etc. for a grid built with `device_grid`).
"""
struct MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, INTERPS, L, NF, VG <: AbstractVector{SVector{B_LEN, State{M, T}}}, WG, WD} <: DivergenceInterpolator
    interpolators::INTERPS
    limiter::L
    flux::NF
    gradients::VG
    # Cached inverse MLS moment matrices for a frozen geometry (see `MLSWeights`): the full-order reconstruction and the
    # divergence fit. They are the same object when both use the same degree. Particles at reduced (MOOD)
    # order fall back to the direct solves.
    grad_weights::WG
    div_weights::WD
end

# Interpolator-table entry the divergence uses at full order (see `_resolve_div_idx`).
@inline _full_div_idx(::Val{MAX_ORDER}, ::Val{DIV_ORDER}, ::Val{D}) where {MAX_ORDER, DIV_ORDER, D} =
    _div_idx(Val(D), DIV_ORDER == 0 ? MAX_ORDER : DIV_ORDER)

@inline _extract_order(::MUSCL{D, M, T, B_LEN, MAX_ORDER}) where {D, M, T, B_LEN, MAX_ORDER} = MAX_ORDER

function MUSCL(
    ::Type{T}, dimension::Int, M::Int, max_order::Int,
    limiter, flux;
    div_order::Int=0, storage::Type{AT} = Array
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
                                                              grad_weights, div_weights)
end

# Infers the storage types (host vectors, or device views when adapted for a kernel).
MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER}(interps, limiter, flux, gradients, grad_weights, div_weights) where {D, M, T, B_LEN, MAX_ORDER, DIV_ORDER} =
    MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER, typeof(interps), typeof(limiter), typeof(flux), typeof(gradients),
          typeof(grad_weights), typeof(div_weights)}(interps, limiter, flux, gradients, grad_weights, div_weights)

function prepare_geometry!(muscl::MUSCL, pg::ParticleGrid)
    update_weights!(muscl.grad_weights, pg)
    muscl.div_weights === muscl.grad_weights || update_weights!(muscl.div_weights, pg)
    return nothing
end

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

    grad_i     = muscl.gradients[i]
    F_i        = flux(eq, f_i)
    dist_all   = get_distances(pg)
    nb_indices = pg.neighbor.indices

    eff_order_i = pg.core.particle_orders[i]
    degree      = Val(max(1, MAX_ORDER - 1))   # reconstruction degree; matches B_LEN of the cached gradients

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

        f_L, f_R, F_L, F_R = sort_flux(fij, fji, F_ij, F_ji, dist_k)
        F_num   = muscl.flux(f_L, f_R, F_L, F_R, eq)
        nc_jump = evaluate_nc_jump(eq, f_L, f_R, dist_k)

        ib.df_flux[global_idx] = F_num - F_i + nc_jump
    end

    div_idx = _divergence_index(Val(DIV_ORDER), Val(D), eff_order_i, length(nb_slice))
    wd = muscl.div_weights
    div = if div_idx == _full_div_idx(Val(MAX_ORDER), Val(DIV_ORDER), Val(D)) && is_current(wd, pg) && wd.ok[i]
        apply_weights_divergence(wd, i, nb_slice, dist_all, get_weights(pg), ib.df_flux)   # cached full-order fit
    else
        compute_dynamic_divergence(muscl.interpolators, div_idx, nb_slice, dist_all, get_weights(pg),
                                   ib.df_flux, ib.df_scratch, pg.meta.dx)
    end

    return T(2.0) * div
end

include("MUSCL_hooks.jl")