function MUSCLWorkspace1D0O(initial_flat_cap::Int = 1000)
    new(zeros(initial_flat_cap))
end

function MUSCLWorkspace1D1O(
    initial_particle_cap::Int = 100, 
    initial_flat_cap::Int = 1000
)
    MUSCLWorkspace1D1O(
        zeros(initial_flat_cap), zeros(initial_flat_cap), # alfaij_bars, betaijs
        zeros(initial_particle_cap), zeros(initial_particle_cap)  # slopes, curves_xx
    )
end

function MUSCLWorkspace1D2O(
    initial_particle_cap::Int = 100, 
    initial_flat_cap::Int = 1000
)
    MUSCLWorkspace1D2O(
        zeros(initial_flat_cap), zeros(initial_flat_cap), # alfaij_bars, betaijs
        zeros(initial_particle_cap), zeros(initial_particle_cap)  # slopes, curves_xx
    )
end



function MUSCLWorkspace1D3O(
    initial_particle_cap::Int = 100, 
    initial_flat_cap::Int = 1000,
    initial_neighbor_cap::Int = 30 # Max neighbors for temp buffer
)

    MUSCLWorkspace1D3O(
        zeros(initial_flat_cap), zeros(initial_flat_cap), zeros(initial_flat_cap), # [cite: 10]
        zeros(initial_particle_cap), zeros(initial_particle_cap), zeros(initial_particle_cap), # [cite: 10]
        #thread_Q_buffers
    )
end

function MUSCLWorkspace1D4O(
    initial_particle_cap::Int = 100, 
    initial_flat_cap::Int = 1000
)
    MUSCLWorkspace1D4O(
        zeros(initial_flat_cap), zeros(initial_flat_cap), 
        zeros(initial_flat_cap), zeros(initial_flat_cap),
        zeros(initial_particle_cap), zeros(initial_particle_cap), 
        zeros(initial_particle_cap), zeros(initial_particle_cap)
    )
end



function MUSCLWorkspace2D0O(initial_flat_cap::Int = 1000)
    MUSCLWorkspace2D0O(zeros(initial_flat_cap), zeros(initial_flat_cap))
end



function MUSCLWorkspace2D1O(
    initial_particle_cap::Int = 100, 
    initial_flat_cap::Int = 1000 # Capacity for total interactions
)
    
    MUSCLWorkspace2D1O(
            zeros(initial_flat_cap), zeros(initial_flat_cap), # alfaijs, betaijs
            zeros(initial_particle_cap), zeros(initial_particle_cap) # slopes_x, slopes_y
        )
end


function MUSCLWorkspace2D2O(
    initial_particle_cap::Int = 100, 
    initial_flat_cap::Int = 1000 # Capacity for total interactions
)
MUSCLWorkspace2D2O(
        zeros(initial_flat_cap), zeros(initial_flat_cap), # alfaijs, betaijs
        zeros(initial_flat_cap), zeros(initial_flat_cap), # alfaij_bars, betaij_bars
        zeros(initial_flat_cap), # gammaijs
        zeros(initial_particle_cap), zeros(initial_particle_cap), # slopes_x, slopes_y
        zeros(initial_particle_cap), zeros(initial_particle_cap), # curves_xx, curves_yy
        zeros(initial_particle_cap), # curves_xy
    ) 
end

"""
Sets all temporary buffers and coefficient storage in the MUSCL workspace to zero.
"""
function zero_workspace!(ws::MUSCLWorkspace1D)
    # Zero out temporary buffers
    fill!(ws.dx_buffer, 0.0)
    fill!(ws.w_buffer, 0.0)
    fill!(ws.A_buffer, 0.0)

    # Clear out all previously calculated coefficients
    for p_idx in 1:length(ws.alfaijs)
        empty!(ws.alfaijs[p_idx])
        empty!(ws.alfaij_bars[p_idx])
        empty!(ws.betaijs[p_idx])
        empty!(ws.gammaijs[p_idx])
    end
end
function ensure_capacity!(ws::MUSCLWorkspace1D, n::Int)
    # Check if the required number of neighbors `n` exceeds the current buffer capacity.
    
    if n > length(ws.dx_buffer)
        # Calculate a new capacity with a 25% buffer to avoid frequent re-allocations.
        new_capacity = n + n ÷ 4
        
        # Resize all vectors at once using broadcasting.
        resize!.((ws.dx_buffer, ws.w_buffer), new_capacity)
        
        # Re-create the matrix buffer with the new size.
      
        # Using `undef` is slightly faster than `zeros` if it's always overwritten.
        ws.A_buffer = Matrix{Float64}(undef, new_capacity, 4)
    end
    return nothing
end

function _ensure_coeff_vectors_sized!(ws::MUSCLWorkspace1D, p_idx::Int, n::Int)
    # Check if the current buffer for this particle is too small
    if length(ws.alfaijs[p_idx]) < n
        # Calculate a new capacity with a 25% buffer
        new_capacity = n #+ n ÷ 4
        
   
     # Resize all coefficient vectors for this particle at once
        resize!.((
            ws.alfaijs[p_idx], 
            ws.alfaij_bars[p_idx], 
            ws.betaijs[p_idx], 
            ws.gammaijs[p_idx]
        ), new_capacity)
    end
    return nothing
end

function ensure_particle_capacity!(ws::MUSCLWorkspace1D, N::Int)
    current_size = length(ws.alfaijs)
 
   if current_size < N
        new_capacity = N #+ N ÷ 4
        num_to_add = new_capacity - current_size
        
        # Grow the outer vector of vectors
        for _ in 1:num_to_add
            push!(ws.alfaijs, Float64[])
            push!(ws.alfaij_bars, Float64[])
         
   push!(ws.betaijs, Float64[])
            push!(ws.gammaijs, Float64[])
        end
        
        # Resize the simple vector to the new capacity
        resize!(ws.slopes, new_capacity)
        resize!(ws.curves_xx, new_capacity)
        resize!(ws.curves_yy, new_capacity)
    end
    return nothing
end

# --- NEW: ensure_particle_capacity! for 2D workspaces ---
function ensure_particle_capacity!(ws::MUSCLWorkspace2D1O, N::Int)
    if length(ws.slopes_x) < N
        resize!.((
            ws.slopes_x, ws.slopes_y
        ), N)
    end
    return nothing
end

function ensure_particle_capacity!(ws::MUSCLWorkspace2D2O, N::Int)
    if length(ws.slopes_x) < N
        resize!.((
            ws.slopes_x, ws.slopes_y,
            ws.curves_xx, ws.curves_yy, ws.curves_xy
        ), N)
    end
    return nothing
end


# --- NEW: ensure_coefficients_capacity! for 2D workspaces ---
"""
Ensures the flat coefficient arrays can hold data for every neighbor interaction.
"""
function ensure_coefficients_capacity!(ws::MUSCLWorkspace2D1O, grid::ParticleGrid2D{S}) where S
    required_len = length(grid.neighbor.indices)
    if length(ws.alfaijs) < required_len
   
     # Resize all flat coefficient arrays at once
        new_capacity = required_len + required_len ÷ 4
        resize!.((
            ws.alfaijs, ws.betaijs
        ), new_capacity)
    end
    return nothing
end

function ensure_coefficients_capacity!(ws::MUSCLWorkspace2D2O, grid::ParticleGrid2D{S}) where S
    required_len = length(grid.neighbor.indices)
    if length(ws.alfaijs) < required_len
   
     # Resize all flat coefficient arrays at once
        new_capacity = required_len + required_len ÷ 4
        resize!.((
            ws.alfaijs, ws.betaijs, 
            ws.alfaij_bars, ws.betaij_bars, 
            ws.gammaijs
        ), new_capacity)
    end
    return nothing
end


# --- NEW: ensure_capacity! for 2D (temporary buffers) ---
# This handles the temporary arrays used for a single particle's neighbors.
# For 1st Order, no temp buffers are needed for initGI!
function ensure_capacity!(ws::MUSCLWorkspace2D1O, n::Int)
    return nothing # No temp buffers to resize
end
