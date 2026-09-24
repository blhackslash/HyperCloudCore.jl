export FixedDirichlet, OutflowBC
"""
    FixedDirichlet
    OutflowBC

Abstract structures designating boundary condition strategies. 
- `FixedDirichlet`: A strict, unchanging boundary condition.
- `OutflowBC`: A zero-gradient, transmissive boundary condition.
"""
struct FixedDirichlet <: AbstractBoundaryCondition end
struct OutflowBC <: AbstractBoundaryCondition end

# =========================================================================
# BOUNDARY CONDITION DISPATCHER
# =========================================================================

"""
    apply_boundary_conditions!(pg::ParticleGrid, rhos_buffer, ts, eq, t)

Iterates over the active boundary condition map stored in the grid's geometry configuration and dispatches the corresponding strategy for each registered domain tag.
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
    @inbounds for i in 1:pg.meta.N
        if pg.core.tags[i] == tag && pg.core.is_boundary[i]
            rhos_buffer[i] = pg.rhos[i] # Just uses initial state
        end
    end
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
    dist_vec = get_distances(pg)
    status = pg.shared.int_buffer 
    fill!(status, -1) # Default to ignored
    
    # Generation 1: Interior particles are donors, target tags are unresolved
    @inbounds for i in 1:pg.meta.N
        if !pg.core.is_boundary[i]
            status[i] = 1  # Valid donor
        elseif pg.core.tags[i] == tag
            status[i] = 0  # Needs resolution
        end
        # Other boundaries remain -1 (ignored)
    end
    
    # Symmetrically propagate the boundary condition outwards
    for pass in 1:5 
        all_resolved = true
        
        @inbounds for i in 1:pg.meta.N
            if status[i] == 0
                nb_slice = pg.neighbor.ranges[i]
                closest_j = -1
                min_dist_sq = T(Inf)
                
                for k in nb_slice
                    j = pg.neighbor.indices[k]
                    
                    if status[j] > 0 && status[j] <= pass
                        d2 = sum(abs2, dist_vec[k])
                        if d2 < min_dist_sq
                            min_dist_sq = d2
                            closest_j = j
                        end
                    end
                end
                
                if closest_j != -1
                    rhos_buffer[i] = rhos_buffer[closest_j]
                    status[i] = pass + 1 
                else
                    all_resolved = false
                end
            end
        end
        
        if all_resolved; break; end
    end
    
    # Fallback for completely orphaned particles
    @inbounds for i in 1:pg.meta.N
        if status[i] == 0
            rhos_buffer[i] = pg.rhos[i]
        end
    end
    return nothing
end