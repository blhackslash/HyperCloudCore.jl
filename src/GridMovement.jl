module GridMovement

using ..ParticleGrids
using ..HyperbolicPDEs
using StaticArrays

export GridMover, NoGridMover, CustomGridMover, PhysicalGridMover, update_grid_velocities!, get_Lagrange_Correction, get_effective_vel

abstract type GridMover end

# ---------------------------------------------------------
# 1. NoGridMover
# ---------------------------------------------------------
struct NoGridMover <: GridMover end

function (gm::NoGridMover)(pg::ParticleGrid, dt::Real; managed=false)
    return
end

function update_grid_velocities!(pg::ParticleGrid, ::NoGridMover)
    return
end

# ---------------------------------------------------------
# 2. CustomGridMover
# ---------------------------------------------------------
struct CustomGridMover{F, P} <: GridMover
    vel_func::F
    params::P
end

function (gm::CustomGridMover)(pg::ParticleGrid{D, M}, dt::Real; managed=true) where {D, M}
    positions = get_positions(pg)
    rhos = pg.rhos
    vel_func = gm.vel_func
    
    for p_idx = 1:pg.meta.N
        # Pass a scalar for 1D equations, or a matrix row view for systems
        rho_val = M == 1 ? rhos[p_idx, 1] : @view(rhos[p_idx, :])
        
        # User function returns velocity components
        v = vel_func(positions[p_idx], rho_val, gm.params)
        
        positions[p_idx] += SVector{D, Float64}(v...) * dt
    end
    
    if managed
        reorder_particles!(pg)
        updateNeighbors!(pg)
        manage_particles!(pg)
    end
    return
end 

function update_grid_velocities!(pg::ParticleGrid, ::CustomGridMover)
    return
end

# ---------------------------------------------------------
# 3. PhysicalGridMover
# ---------------------------------------------------------
mutable struct PhysicalGridMover{E, I, V, D} <: GridMover
    pde::E
    interpolator::I
    vel_kinetic_indices::V                       # Indices of the driving kinetic variables
    grid_velocities::Vector{SVector{D, Float64}} # Pre-allocated workspace buffer

    # Constructor 1: For Scalar Equations (No indices needed)
    function PhysicalGridMover(pde::E, interp::I) where {E, I}
        new{E, I, Nothing, 1}(pde, interp, nothing, Vector{SVector{1, Float64}}(undef, 0))
    end

    # Constructor 2: For Systems (Takes driving indices and Dimension)
    function PhysicalGridMover(pde::E, interp::I, vel_indices::V, ::Val{D}) where {E, I, V, D}
        new{E, I, V, D}(pde, interp, vel_indices, Vector{SVector{D, Float64}}(undef, 0))
    end
end

# --- 3a. SCALAR PDE MOVEMENTS (Inline Updates, No Buffer Needed) ---

function (gm::PhysicalGridMover{BurgersEquation{a}, I, Nothing, 1})(pg::ParticleGrid{1, 1}, dt::Real; managed=true) where {a, I}
    for p_idx = 1:pg.meta.N
        get_positions(pg)[p_idx] += SVector{1, Float64}(a * pg.rhos[p_idx, 1] * dt)
    end
    if managed
        reorder_particles!(pg)
        updateNeighbors!(pg)
        manage_particles!(pg)  
    end
    return    
end

function (gm::PhysicalGridMover{TestU3Equation{a}, I, Nothing, 1})(pg::ParticleGrid{1, 1}, dt::Real; managed=true) where {a, I}
    for p_idx = 1:pg.meta.N
        get_positions(pg)[p_idx] += SVector{1, Float64}(a * (pg.rhos[p_idx, 1])^2 * dt)
    end
    if managed
        reorder_particles!(pg)
        updateNeighbors!(pg)
        manage_particles!(pg)  
    end
    return    
end

function (gm::PhysicalGridMover{LinearAdvection{1}, I, Nothing, 1})(pg::ParticleGrid{1, 1}, dt::Real; managed=true) where {I}
    for p_idx = 1:pg.meta.N
        get_positions(pg)[p_idx] += SVector{1, Float64}(1.0 * dt)
    end
    if managed
        reorder_particles!(pg)
        updateNeighbors!(pg)
        manage_particles!(pg)  
    end
    return    
end

# --- 3b. SYSTEM PDE MOVEMENTS (Buffered Updates) ---

function (gm::PhysicalGridMover{E, I, V, D})(pg::ParticleGrid{D, M}, dt::Real; managed = true) where {E, I, V, D, M}
    # Move particles using the pre-computed buffer
    for p_idx in 1:pg.meta.N
        get_positions(pg)[p_idx] += gm.grid_velocities[p_idx] * dt
    end
    
    if managed
        reorder_particles!(pg)
        updateNeighbors!(pg)
        manage_particles!(pg) 
    end
end 

# --- 3c. UPDATE SYSTEM GRID VELOCITIES (Pre-computation pass) ---

function update_grid_velocities!(pg::ParticleGrid{D, M}, gm::PhysicalGridMover{E, I, V, D}) where {D, M, E, I, V}
    N = pg.meta.N
    if length(gm.grid_velocities) < N
        resize!(gm.grid_velocities, ceil(Int, N * 1.25))
    end
    
    Threads.@threads for i in 1:N
        rho_sum = 0.0
        # Iterate over the subset of kinetic variables that define "velocity"
        for k in gm.vel_kinetic_indices
            rho_sum += pg.rhos[i, k]
        end
        
        # Physics implementation (Default mapping)
        u_grid = rho_sum 
        
        if D == 1
            gm.grid_velocities[i] = SVector{1, Float64}(u_grid)
        else
            gm.grid_velocities[i] = SVector{D, Float64}(fill(u_grid, D)...)
        end
    end
end

# Specialized version if the underlying system explicitly passes Burgers
function update_grid_velocities!(pg::ParticleGrid{D, M}, gm::PhysicalGridMover{BurgersEquation{a}, I, V, D}) where {D, M, a, I, V}
    N = pg.meta.N
    if length(gm.grid_velocities) < N
        resize!(gm.grid_velocities, ceil(Int, N * 1.25))
    end    
    
    Threads.@threads for i in 1:N
        rho_sum = 0.0
        for k in gm.vel_kinetic_indices
            rho_sum += pg.rhos[i, k]
        end
        
        u_grid = a * rho_sum 
        
        if D == 1
            gm.grid_velocities[i] = SVector{1, Float64}(u_grid)
        else
            gm.grid_velocities[i] = SVector{D, Float64}(fill(u_grid, D)...)
        end
    end
end

end # module