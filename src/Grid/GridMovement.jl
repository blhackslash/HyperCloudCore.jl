# =========================================================================
# ACCESSOR FUNCTIONS
# =========================================================================

# 1D: Reinterpret the Vector of SVectors directly into a Vector of Floats
@inline get_grid_velocities(gm::PhysicalGridMover{1}) = reinterpret(Float64, gm.grid_velocities)
# 2D/3D: Return as-is
@inline get_grid_velocities(gm::PhysicalGridMover{D}) where {D} = gm.grid_velocities

# =========================================================================
# 1. NoGridMover
# =========================================================================

function (gm::NoGridMover)(pg::ParticleGrid, dt::Real; managed=false)
    return
end

function update_grid_velocities!(pg::ParticleGrid, ::NoGridMover)
    return
end

# # =========================================================================
# # 2. CustomGridMover
# # =========================================================================

# function (gm::CustomGridMover)(pg::ParticleGrid{D, M}, dt::Real) where {D, M}
#     positions = get_positions(pg)
#     rhos = pg.rhos
#     vel_func = gm.vel_func
    
#     for p_idx = 1:pg.meta.N
#         # Pass a scalar for 1D equations, or a matrix row view for systems
#         rho_val = M == 1 ? rhos[p_idx, 1] : @view(rhos[p_idx, :])
        
#         # User function returns velocity components
#         v = vel_func(positions[p_idx], rho_val, gm.params)
        
#         # Branch explicitly on Dimension to avoid SVector/Float mismatches
#         if D == 1
#             positions[p_idx] += v[1] * dt
#         else
#             positions[p_idx] += SVector{D, Float64}(v...) * dt
#         end
#     end
#     pg.neighbor(pg)
#     return
# end 

# function update_grid_velocities!(pg::ParticleGrid, ::CustomGridMover)
#     return
# end

# # =========================================================================
# # 3a. SCALAR PDE MOVEMENTS (Inline Updates, No Buffer Needed)
# # =========================================================================

# function (gm::PhysicalGridMover{BurgersEquation{a}, I, Nothing, 1})(pg::ParticleGrid{1, 1}, dt::Real) where {a, I}
#     pos = get_positions(pg)
#     for p_idx = 1:pg.meta.N
#         pos[p_idx] += (a * pg.rhos[p_idx, 1] * dt)
#     end
#     pg.neighbor(pg)
#     return    
# end

# function (gm::PhysicalGridMover{TestU3Equation{a}, I, Nothing, 1})(pg::ParticleGrid{1, 1}, dt::Real) where {a, I}
#     pos = get_positions(pg)
#     for p_idx = 1:pg.meta.N
#         pos[p_idx] += (a * (pg.rhos[p_idx, 1])^2 * dt)
#     end
#     pg.neighbor(pg)
#     return    
# end

# function (gm::PhysicalGridMover{LinearAdvection{1}, I, Nothing, 1})(pg::ParticleGrid{1, 1}, dt::Real) where {I}
#     pos = get_positions(pg)
#     for p_idx = 1:pg.meta.N
#         pos[p_idx] += (1.0 * dt)
#     end
#     pg.neighbor(pg)
#     return    
# end

# # =========================================================================
# # 3b. SYSTEM PDE MOVEMENTS (Buffered Updates)
# # =========================================================================

# function (gm::PhysicalGridMover{E, I, V, D})(pg::ParticleGrid{D, M}, dt::Real; managed = true) where {E, I, V, D, M}
#     pos = get_positions(pg)
#     vel = get_grid_velocities(gm)
    
#     # Move particles using the pre-computed buffer
#     # Because of get_grid_velocities, 'pos' and 'vel' match types natively!
#     for p_idx in 1:pg.meta.N
#         pos[p_idx] += vel[p_idx] * dt
#     end
#     pg.neighbor(pg)
#     return
# end 

# # =========================================================================
# # 3c. UPDATE SYSTEM GRID VELOCITIES (Pre-computation pass)
# # =========================================================================

# function update_grid_velocities!(pg::ParticleGrid{D, M}, gm::PhysicalGridMover{E, I, V, D}) where {D, M, E, I, V}
#     N = pg.meta.N
#     if length(gm.grid_velocities) < N
#         resize!(gm.grid_velocities, ceil(Int, N * 1.25))
#     end
    
#     vel = get_grid_velocities(gm)
    
#     Threads.@threads for i in 1:N
#         rho_sum = 0.0
#         # Iterate over the subset of kinetic variables that define "velocity"
#         for k in pg.km.ranges[]
#             rho_sum += pg.rhos[i, k]
#         end
        
#         u_grid = rho_sum 
        
#         if D == 1
#             vel[i] = u_grid # Directly assign the float
#         else
#             vel[i] = SVector{D, Float64}(fill(u_grid, D)...)
#         end
#     end
# end

# # Specialized version if the underlying system explicitly passes Burgers
# function update_grid_velocities!(pg::ParticleGrid{D, M}, gm::PhysicalGridMover{BurgersEquation{a}, I, V, D}) where {D, M, a, I, V}
#     N = pg.meta.N
#     if length(gm.grid_velocities) < N
#         resize!(gm.grid_velocities, ceil(Int, N * 1.25))
#     end    
    
#     vel = get_grid_velocities(gm)
    
#     Threads.@threads for i in 1:N
#         rho_sum = 0.0
#         for k in gm.vel_kinetic_indices
#             rho_sum += pg.rhos[i, k]
#         end
        
#         u_grid = a * rho_sum 
        
#         if D == 1
#             vel[i] = u_grid # Directly assign the float
#         else
#             vel[i] = SVector{D, Float64}(fill(u_grid, D)...)
#         end
#     end
# end