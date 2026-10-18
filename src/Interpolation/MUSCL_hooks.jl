# =========================================================================
# MUSCL ↔ TIME-STEPPING HOOKS
# Append this block to the end of MUSCL.jl (or `include` it right after MUSCL.jl). Everything the
# stepper, MOOD and the grid movers need to know about MUSCL is declared here, so the
# TimeStepping/ files never mention the MUSCL type and can be loaded first.
# =========================================================================

# --- MOOD locality: how far (hops) a particle's flux depends on neighbours' orders ---
# Neighbours read grad_p and (o_i, o_j); EPD2 reads neighbourhood minima.
@inline _order_reach(::MUSCL, ::MOODStrategy) = 1
@inline _order_reach(::MUSCL, ::EPD2) = 2

# --- CFL: the degree of the full-order divergence fit (`_div_idx`) ---
@inline _divergence_degree(::MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER}) where {D, M, T, B_LEN, MAX_ORDER, DIV_ORDER} =
    _full_div_idx(Val(MAX_ORDER), Val(DIV_ORDER), Val(D)) - 1

# --- MOODu2: curvature coefficients come from the cached limited gradients ---
@inline has_cached_gradients(::MUSCL) = true
@inline cached_gradients(m::MUSCL) = m.gradients
