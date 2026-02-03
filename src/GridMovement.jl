module GridMovement

using ..ParticleGrids
using ..HyperbolicPDEs

export GridMover, NoGridMover, CustomGridMover, PhysicalGridMover, get_effective_vel, update_grid_velocities!

abstract type GridMover end

struct NoGridMover <: GridMover end

struct CustomGridMover <: GridMover
    vel_func::Function
    params::Tuple
end

struct PhysicalGridMover{E} <: GridMover
    pde::E
end

# Default function, no grid move
function (gm::NoGridMover)(pg::ParticleGrid, dt::Real); return; end

function (gm::CustomGridMover)(pg::ParticleGrid, dt::Real); 
    positions = pg.positions
    rhos = pg.rhos
    vel_func = gm.vel_func
    for p_idx = 1:pg.N
        rho = rhos[p_idx]
        positions[p_idx] += vel_func(positons[p_idx],rho,gm.params) * dt
    end
    sort_1d_particles!(pg)
    updateNeighbors!(pg)
    manage_particles!(pg)
    sort_1d_particles!(pg)
    updateNeighbors!(pg)    
    return
end 

function (gm::PhysicalGridMover{BurgersEquation{a}})(pg::ParticleGrid1D, dt::Real) where {a}
    rhos = pg.rhos
    positions = pg.positions
    for p_idx = 1:pg.N
        positions[p_idx] += a * rhos[p_idx] * dt
    end
    sort_1d_particles!(pg)
    updateNeighbors!(pg)
    manage_particles!(pg)  
    return    
end

function (gm::PhysicalGridMover{TestU3Equation{a}})(pg::ParticleGrid1D, dt::Real) where {a}
    rhos = pg.rhos
    positions = pg.positions
    for p_idx = 1:pg.N
        positions[p_idx] += a * (rhos[p_idx])^2 * dt
    end
    sort_1d_particles!(pg)
    updateNeighbors!(pg)
    manage_particles!(pg)  
    return    
end

# General grid movement based on predetermined velocities
function (gm::GridMover)(pgs::ParticleGridSystem{N_grids,1}, dt::Real) where {N_grids}
    N_test = pgs[1].N
    for pg in pgs
        positions = pg.positions
        for p_idx in 1:pg.N
            @assert pg.N == N_test "Different grid sizes found!"
            positions[p_idx] += pgs.grid_velocities[p_idx] * dt
        end
        sort_1d_particles!(pg)
        updateNeighbors!(pg)
        manage_particles!(pg) 
        updateNeighbors!(pg)
    end
end

function (gm::PhysicalGridMover{LinearAdvection{1}})(pg::ParticleGrid1D, dt::Real)
    positions = pg.positions
    for p_idx = 1:pg.N
        positions[p_idx] += 1. * dt
    end
    sort_1d_particles!(pg)
    updateNeighbors!(pg)
    manage_particles!(pg)  
    return    
end

# In GridMovement.jl or MeshfreeSystemTimeSteppers.jl

"""
    update_grid_velocities!(pgs::ParticleGridSystem, system_eqs)

Calculates the grid velocity for every particle based on the densities of the 
species specified in `pgs.velocity_indices`.
"""
function update_grid_velocities!(pgs::ParticleGridSystem{1, N_grids}, system_eqs) where {N_grids}
    # 1. Access the buffer and grids
    grid_vels = pgs.grid_velocities
    N = pgs.grids[1].N
    if length(grid_vels) < N
        resize!(grid_vels, Int(ceil(N * 1.2)))
    end
    # 2. Loop over particles (Thread-safe here)
    Threads.@threads for i in 1:N
        # A. Calculate total rho for the "driving" species
        rho_sum = 0.0
        for k in pgs.velocity_indices
            rho_sum += pgs.grids[k].rhos[i]
        end
        
        # B. Calculate u_grid based on your physics (e.g., Burgers-like)
        # Note: You can customize this logic or dispatch based on system_eqs
        # For this example, we assume u_grid = rho_sum (like Burgers)
        u_grid = rho_sum 
        
        # C. Store in buffer
        grid_vels[i] = u_grid
    end
end

function update_grid_velocities!(pgs::ParticleGridSystem, ::NoGridMover)
    return
end
"""
    update_grid_velocities!(pgs::ParticleGridSystem, system_eqs)

Calculates the grid velocity for every particle based on the densities of the 
species specified in `pgs.velocity_indices`.
"""
function update_grid_velocities!(pgs::ParticleGridSystem{N_grids,1},::PhysicalGridMover{BurgersEquation{a}}) where {N_grids,a}
    # 1. Access the buffer and grids
    grid_vels = pgs.grid_velocities
    N = pgs.grids[1].N
    if length(grid_vels) < N
        resize!(grid_vels, Int(ceil(N * 1.2)))
    end    
    # 2. Loop over particles (Thread-safe here)
    Threads.@threads for i in 1:N
        # A. Calculate total rho for the "driving" species
        rho_sum = 0.0
        for k in pgs.velocity_indices
            rho_sum += pgs.grids[k].rhos[i]
        end
        
        # B. Calculate u_grid based on your physics (e.g., Burgers-like)
        # Note: You can customize this logic or dispatch based on system_eqs
        # For this example, we assume u_grid = rho_sum (like Burgers)
        u_grid = a * rho_sum 
        
        # C. Store in buffer
        grid_vels[i] = u_grid
    end
end

@inline function get_effective_vel(eq::ScalarHyperbolicPDE{1},vel::Real)
    v = eq.vel[1]
    return v - sign(v) * vel
end

@inline function get_effective_vel(eq::ScalarHyperbolicPDE{2},vel::NTuple{2,Float64})
    vx = eq.vel[1]
    vy = eq.vel[2]
    return vx - sign(vx) *vel[1], vy - sign(vy) * vel[2]
end

end