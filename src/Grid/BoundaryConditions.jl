export FixedDirichlet, OutflowBC

"""
    FixedDirichlet <: AbstractBoundaryCondition
    (::FixedDirichlet)(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq, t)

A strict, unchanging boundary condition strategy.

# Callable / Functor

Enforces a static Dirichlet condition upon boundary particles. Iterates over the grid and resets the state of any boundary particle matching the specified `tag` back to its initial original state stored in `pg.rhos`.
"""
struct FixedDirichlet <: AbstractBoundaryCondition end

"""
    OutflowBC <: AbstractBoundaryCondition
    (::OutflowBC)(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq, t)

A zero-gradient, transmissive boundary condition strategy.

# Callable / Functor

Enforces a zero-gradient outflow utilizing a multi-pass nearest-donor algorithm. 
- Initiates all active interior particles as valid state donors.
- Executes up to 5 symmetrical passes outward, dynamically locating the nearest resolved neighbor using squared distances and copying its state to the target particle.
- Safely falls back to the original initial state for any completely orphaned boundary particles that fail to resolve a donor.
"""
struct OutflowBC <: AbstractBoundaryCondition end

# =========================================================================
# BOUNDARY CONDITION DISPATCHER
# =========================================================================

"""
    apply_boundary_conditions!(pg::ParticleGrid, rhos_buffer, ts, eq, t)

Iterates over the active boundary condition map stored in the grid's geometry configuration and dispatches the corresponding strategy for each registered domain tag.
The map itself (a `Dict`) stays on the host; the conditions shipped with the package run as kernel
launches, so they work on a grid built with `device_grid` too. A user-defined condition is any callable
`(bc)(pg, rhos_buffer, tag, ts, eq, t)`.
"""
function apply_boundary_conditions!(pg::ParticleGrid{D, M, T}, rhos_buffer::AbstractVector{State{M, T}}, ts::TimeStepper, eq::HyperbolicPDE, t::Real) where {D, M, T}
    for (tag, bc_functor) in pg.geometry.bc_map
        bc_functor(pg, rhos_buffer, tag, ts, eq, t)
    end
    return nothing
end

# =========================================================================
# CONCRETE BOUNDARY CONDITIONS
# =========================================================================

# 1. Fixed Dirichlet
"""
    (::FixedDirichlet)(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq, t)

Enforces a static Dirichlet condition upon boundary particles. 
- Iterates over the grid and resets the state of any particle matching the specified boundary tag back to its initial original state stored in `pg.rhos`.
"""
function (::FixedDirichlet)(pg::ParticleGrid{D, M, T}, rhos_buffer::AbstractVector{State{M, T}}, tag::Int, ts, eq::HyperbolicPDE, t::Real) where {D, M, T}
    foreach_particle(_dirichlet_body!, get_backend(), pg.meta.N, (rhos_buffer, pg.rhos, pg.core.tags, pg.core.is_boundary, tag))
    return nothing
end

# Kernel body (see Backend.jl): a tagged boundary particle keeps its initial state.
@inline function _dirichlet_body!(i, buf, rhos, tags, is_boundary, tag)
    (is_boundary[i] && tags[i] == tag) && (buf[i] = rhos[i])
    return nothing
end

# 2. Outflow (Zero-Divergence)
"""
    (::OutflowBC)(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq, t)

Enforces a zero-gradient outflow boundary condition using a multi-pass nearest-donor algorithm.
- Initiates all active interior particles as valid state donors, and marks boundary particles matching the specific tag as requiring resolution.
- Executes up to 5 symmetrical passes outward, dynamically locating the nearest resolved neighbor using squared distances and copying its state to the target particle.
- Safely falls back to the original initial state for any completely orphaned boundary particles that failed to resolve a donor.
"""
function (::OutflowBC)(pg::ParticleGrid{D, M, T}, rhos_buffer::AbstractVector{State{M, T}}, tag::Int, ts, eq::HyperbolicPDE, t::Real) where {D, M, T}
    N       = pg.meta.N
    nb      = pg.neighbor
    status  = pg.shared.int_buffer      # 1: donor (interior), 0: unresolved target, -1: other boundary;
    backend = get_backend()             # pass + 1 once resolved in `pass`

    foreach_particle(_outflow_init!, backend, N, (status, pg.core.tags, pg.core.is_boundary, tag))

    # Each pass resolves from donors of earlier passes only, so no two particles read what another writes.
    for pass in 1:5
        foreach_particle(_outflow_pass!, backend, N, (rhos_buffer, status, nb.ranges, nb.indices, nb.distances, pass, T(Inf)))
        any(==(0), view(status, 1:N)) || break
    end

    foreach_particle(_outflow_fallback!, backend, N, (rhos_buffer, pg.rhos, status))
    return nothing
end

# Kernel bodies of the outflow donor search (see Backend.jl).
@inline function _outflow_init!(i, status, tags, is_boundary, tag)
    status[i] = !is_boundary[i] ? 1 : (tags[i] == tag ? 0 : -1)
    return nothing
end

@inline function _outflow_pass!(i, buf, status, nb_slices, nb_indices, dists, pass, inf)
    status[i] == 0 || return nothing
    closest = -1
    min_d2  = inf
    @inbounds for k in nb_slices[i]
        j = nb_indices[k]
        s = status[j]
        if 0 < s <= pass
            d2 = sum(abs2, dists[k])
            if d2 < min_d2
                min_d2  = d2
                closest = j
            end
        end
    end
    if closest != -1
        buf[i]    = buf[closest]
        status[i] = pass + 1
    end
    return nothing
end

@inline function _outflow_fallback!(i, buf, rhos, status)
    status[i] == 0 && (buf[i] = rhos[i])
    return nothing
end