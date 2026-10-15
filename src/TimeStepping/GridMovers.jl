export LagrangianMover, ScaledLagrangianMover, PrescribedMover
export ALEFrame, grid_velocity, fluid_velocity, refresh_geometry!, set_positions!

# =========================================================================
# MOVING GRIDS — ALE COUPLING
# =========================================================================
#
# Particle positions are promoted to stage variables of the same Butcher tableau:
#
#     dx_i/dt = w_i(x, ρ, t)                              (explicit part of the tableau only)
#     dρ_i/dt = -div (F(ρ) − w_i ρ)|_i + S_i                (at the current stage positions)
#
# The second line is the ALE form for a collocation (pointwise) meshfree scheme: for a particle
# moving with w, Dρ/Dt = ∂_t ρ + w·∇ρ = −div F + w·∇ρ = −div(F − w_i ρ) for a frozen w_i.
# The relative flux is handled by wrapping the PDE in an `ALEFrame`, so the numerical flux,
# its dissipation and the limiters all see the relative velocity. There is no operator
# splitting, so the coupled system inherits the order of the tableau.
#
# Within a step the neighbour TOPOLOGY is frozen (indices from the start-of-step search) and only
# distances and weights are refreshed at each stage position; pairs that drift beyond R get zero
# weight, pairs that enter R during the step are picked up by the full search at the end of the
# step. With CFL-limited dt and R ≈ 3 dx this is the usual "neighbour skin" assumption. Set
# `full_search_per_stage = true` on the stepper to rebuild the topology at every stage instead.

# -------------------------------------------------------------------------
# Movers
# -------------------------------------------------------------------------
"""
    LagrangianMover <: GridMover

`w = fluid_velocity(eq, U)` (the PDE must implement `fluid_velocity`).
"""
struct LagrangianMover <: GridMover end

"""
    ScaledLagrangianMover(α) <: GridMover

`w = α · fluid_velocity(eq, U)`; `α = 1` is Lagrangian, `α = 0` is Eulerian.
"""
struct ScaledLagrangianMover{T} <: GridMover
    alpha::T
end

"""
    PrescribedMover(f) <: GridMover

`w = f(x::Space, t)`, independent of the state (e.g. a rigid rotation or a mesh-smoothing field).
"""
struct PrescribedMover{F} <: GridMover
    velocity::F
end

"""
    grid_velocity(mover, eq, p_idx, U, pg, t) -> Space{D, T}

Grid velocity of particle `p_idx` at the current stage (positions in `pg.core.positions`).
"""
@inline grid_velocity(::NoGridMover, eq, p_idx, U, pg::ParticleGrid{D, M, T}, t) where {D, M, T} = zero(Space{D, T})
@inline grid_velocity(::LagrangianMover, eq, p_idx, U, pg, t) = fluid_velocity(eq, U)
@inline grid_velocity(m::ScaledLagrangianMover, eq, p_idx, U, pg, t) = m.alpha * fluid_velocity(eq, U)
@inline grid_velocity(m::PrescribedMover, eq, p_idx, U, pg, t) = m.velocity(pg.core.positions[p_idx], t)

# -------------------------------------------------------------------------
# ALE frame: the relative flux F − w_i ρ
# -------------------------------------------------------------------------
"""
    ALEFrame(eq, w) <: HyperbolicPDE

`eq` seen from a frame moving with the constant velocity `w`: `flux = F(U) − w ⊗ U`, Jacobians
shifted by `−w_d I`, wave speeds by `|λ − w_d|`. The stepper wraps the PDE with the particle's
own grid velocity `w_i` before calling the divergence interpolator, so for a constant `w_i`
`div(F − w_i ρ) = div F − w_i·∇ρ` holds exactly, the numerical dissipation scales with the
relative speed, and for `w = u` every interface flux vanishes (exact Lagrangian transport).
"""
struct ALEFrame{D, M, T, EQ <: HyperbolicPDE{D, M, T}} <: HyperbolicPDE{D, M, T}
    eq::EQ
    w::Space{D, T}
end

@inline function flux(a::ALEFrame{D, M, T}, U::State{M, T}) where {D, M, T}
    F = flux(a.eq, U)
    return Flux{D, M, T}(ntuple(d -> F[d] - a.w[d] * U, Val(D)))
end
@inline max_eigenvalue(a::ALEFrame, U::State, d::Int) = max_eigenvalue(a.eq, U, d, a.w[d])
@inline velocity(a::ALEFrame{D, M, T}, U::State, d::Int) where {D, M, T} =
    velocity(a.eq, U, d) - a.w[d] * SMatrix{M, M, T}(I)
@inline prim2cons(a::ALEFrame, U::State) = prim2cons(a.eq, U)
@inline cons2prim(a::ALEFrame, W::State) = cons2prim(a.eq, W)
@inline evaluate_nc_jump(a::ALEFrame{D, M, T}, f_L::Flux{D, M, T}, f_R::Flux{D, M, T}, dist_k::Space{D, T}) where {D, M, T} =
    evaluate_nc_jump(a.eq, f_L, f_R, dist_k)

# -------------------------------------------------------------------------
# Geometry refresh with frozen topology
# -------------------------------------------------------------------------
"""
    refresh_geometry!(pg)

Recomputes neighbour distances and weights from the current `pg.core.positions` for the existing
neighbour lists. Pairs now farther than `R` get weight zero.
"""
function refresh_geometry!(pg::ParticleGrid{D, M, T}) where {D, M, T}
    N         = pg.meta.N
    pos       = pg.core.positions
    nb        = pg.neighbor
    L, L_inv  = pg.domain.L_wrap, pg.domain.invL_wrap
    R_sq      = pg.meta.R^2
    wf        = nb.weight_func
    use_threads = get_use_threads()
    zero_T    = zero(T)   # captured as a value (static parameters are runtime objects in a @batch body)

    @smart_parallel use_threads for i in 1:N
        @inbounds for k in nb.ranges[i]
            j    = nb.indices[k]
            dist = get_distance(pos, i, j, L, L_inv)
            d2   = sum(abs2, dist)
            nb.distances[k] = dist
            nb.weights[k]   = d2 <= R_sq ? wf(d2) : zero_T
        end
    end
    pg.meta.geometry_version += 1
    return nothing
end

"""
    set_positions!(pg, X, full_search::Bool)

Writes stage positions `X` into the grid and refreshes the geometry: distances/weights only
(`full_search = false`) or a complete neighbour search (`true`, also re-wraps periodic axes).
"""
function set_positions!(pg::ParticleGrid, X::AbstractVector, full_search::Bool)
    N = pg.meta.N
    pg.core.positions[1:N] .= view(X, 1:N)
    full_search ? pg.neighbor(pg) : refresh_geometry!(pg)
    return nothing
end
