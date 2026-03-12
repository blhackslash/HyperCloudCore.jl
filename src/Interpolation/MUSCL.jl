function MUSCL(
    order::Int, 
    dimension::Int; 
    limiter::L=NoLimiter(), 
    numericalFlux=RusanovFlux(),
    mood::M = NoMOOD(),
) where {L<:AbstractSlopeLimiter, M <: MOODCriterion}
    
    local ws::MUSCLWorkspace 
    
    if dimension == 1
        if order == 0
            ws = MUSCLWorkspace1D0O()
        elseif order == 1
            ws = MUSCLWorkspace1D1O()
        elseif order == 2
            ws = MUSCLWorkspace1D2O()
        elseif order == 3
            ws = MUSCLWorkspace1D3O()
        elseif order == 4
            ws = MUSCLWorkspace1D4O()
        else
            error("Order $order not supported for 1D workspace.")
        end
    elseif dimension == 2
        if order == 0
             ws = MUSCLWorkspace2D0O()
        elseif order == 1
            ws = MUSCLWorkspace2D1O()
        elseif order == 2
            ws = MUSCLWorkspace2D2O()
        else
            error("Order $order not fully supported for 2D workspace yet.")
        end
    else
        error("Dimension $dimension not supported.")
    end
    
    res_size = (dimension == 1) ? order : (order == 1 ? 2 : 5)
    # For order 0, res_size can be 0 or 1, it doesn't matter much as we don't store gradients
    if order == 0; res_size = 1; end 
    WS = typeof(ws)
    
    if order == 0
        return MUSCL{dimension, MUSCLORDER0, L, typeof(numericalFlux), WS, M}(MUSCLORDER0(), limiter, zeros(res_size), numericalFlux, ws, mood)
    elseif order == 1
        return MUSCL{dimension, MUSCLORDER1, L, typeof(numericalFlux), WS, M}(MUSCLORDER1(), limiter, zeros(res_size), numericalFlux, ws, mood)
    elseif order == 2
        return MUSCL{dimension, MUSCLORDER2, L, typeof(numericalFlux), WS, M}(MUSCLORDER2(), limiter, zeros(res_size), numericalFlux, ws, mood)
    elseif order == 3
        return MUSCL{dimension, MUSCLORDER3, L, typeof(numericalFlux), WS, M}(MUSCLORDER3(), limiter, zeros(res_size), numericalFlux, ws, mood)
    elseif order == 4
        return MUSCL{dimension, MUSCLORDER4, L, typeof(numericalFlux), WS, M}(MUSCLORDER4(), limiter, zeros(res_size), numericalFlux, ws, mood)
    else
        error("Order must be 0, 1, 2, 3, or 4.")
    end
end

"""
    initGIBuffers!(g::MUSCL, pg)

A high-level wrapper that ensures all buffers within the MUSCL gradient
interpolator's workspace are adequately sized for the given particle grid.

This function automatically dispatches to the correct implementation based 
on the type of `g.workspace`.
"""
function initGIBuffers!(g::MUSCL, pg)
    # Dispatch to the specific implementation based on the workspace type
    initGIBuffers!(g.workspace, pg)
    return nothing
end

"""
(2D, Order 1 Implementation) Ensures 2D1O workspace buffers are sized.
- Resizes per-particle arrays (slopes_x, slopes_y) to size N[cite: 21].
- Resizes flat coefficient arrays (alfaijs, betaijs) based on the 
  total number of interactions, plus a 25% buffer[cite: 22].
"""
function initGIBuffers!(ws::MUSCLWorkspace, pg::ParticleGrid)
    N = pg.meta.N
    
    # 1. Ensure capacity for per-particle buffers (size N)
    ensure_particle_capacity!(ws, N)
    
    # 2. Ensure capacity for flat coefficient buffers (size M + 25%)
    # This existing function already adds the 25% buffer [cite: 22]
    ensure_coefficients_capacity!(ws, pg)

    # 3. Temporary buffers: Not needed for 2D1O [cite: 24]
    
    return nothing
end
# --- Buffer Initialization ---

function initGIBuffers!(ws::MUSCLWorkspace1D0O, pg::ParticleGrid1D)
    M = length(pg.neighbor.indices)
    if length(ws.alfaij_bars) < M
        resize!(ws.alfaij_bars, M)
    end
end

function initGIBuffers!(ws::MUSCLWorkspace2D0O, pg::ParticleGrid)
    # Order 0 only needs coefficient capacity, no particle arrays
    ensure_coefficients_capacity!(ws, pg)
end

# Helper for 2D0O
function ensure_coefficients_capacity!(ws::MUSCLWorkspace2D0O, grid::ParticleGrid2D)
    required_len = length(grid.neighbor.indices)
    if length(ws.alfaijs) < required_len
        new_capacity = required_len + required_len ÷ 4
        resize!.((ws.alfaijs, ws.betaijs), new_capacity)
    end
end
# --- NEW: initGIBuffers! for 1D workspaces ---
function initGIBuffers!(ws::Union{MUSCLWorkspace1D1O,MUSCLWorkspace1D2O}, pg::ParticleGrid1D)
    N = pg.meta.N
    M = length(pg.neighbor.indices) # Total interactions
    
    if length(ws.slopes) < N
        resize!.((ws.slopes, ws.curves_xx), N)
    end
    if length(ws.alfaij_bars) < M
        resize!.((ws.alfaij_bars, ws.betaijs), M)
    end
end

function initGIBuffers!(ws::MUSCLWorkspace1D3O, pg::ParticleGrid1D)
    N = pg.meta.N
    M = length(pg.neighbor.indices)
    
    # --- Resize derivative and coefficient buffers (as before) ---
    if length(ws.slopes) < N
        resize!.((ws.slopes, ws.curves_xx, ws.d3fdx3), N)
    end
    if length(ws.alfaijs) < M
        resize!.((ws.alfaijs, ws.alfaij_bars, ws.betaijs), M)
    end

    # # --- NEW: Resize thread-local buffers based on max_nb ---
    # max_nb = pg.meta.max_nb 
    
    # # Check if the *current* buffers are inadequately sized
    # if size(ws.thread_Q_buffers[1], 1) < max_nb
    #     # Re-allocate all thread-local buffers to the new, correct size
    #     for tid in 1:Threads.nthreads()
    #         # Note: This allocates, but only *once* per simulation setup,
    #         # not inside the time-stepping loop.
    #         ws.thread_Q_buffers[tid] = zeros(Float64, max_nb, 3)
    #     end
    # end
end

function initGIBuffers!(ws::MUSCLWorkspace1D4O, pg::ParticleGrid1D)
    N = pg.meta.N
    M = length(pg.neighbor.indices)
    
    if length(ws.slopes) < N
        resize!.((ws.slopes, ws.curves_xx, ws.d3fdx3, ws.d4fdx4), N)
    end
    if length(ws.alfaijs) < M
        resize!.((ws.alfaijs, ws.alfaij_bars, ws.betaijs, ws.gammaijs), M)
    end
end

# --- NEW: Localized Helper Functions for initGI! (2D, Order 1) ---


include("MUSCLCoeffs.jl")
include("MUSCLUtils.jl")
include("MUSCLLimiter.jl")

"""
    initGI!(muscl::MUSCL{2}, ...)

Main Orchestrator function to calculate and store derivatives.
Dispatches calculation based on muscl.order and ws type.
"""
function initGI!(
    muscl::MUSCL{D},
    i::Int,                         # Current particle index
    f_i::Real,
    nb_slice::UnitRange{Int},                      # Value of f at particle i
    pg::ParticleGrid{D},               # Grid object
    neighbor_fs::AbstractVector,    # The flat neighbor-value buffer
    neighbor_dfs::AbstractVector    # The flat neighbor-difference buffer
) where D
    ws = muscl.workspace # ws will be MUSCLWorkspace2D1O or MUSCLWorkspace2D2O
    if i < 0
        # 1. Set 1st-order slopes to zero [cite: 76]
        slopes = D == 1 ? 0. : ntuple(x -> 0., D)
        
        # 2. Get the correctly-shaped tuple of zeros for higher derivatives
        # (e.g., () for O1, (0.0,) for 1D O2, (0.0, 0.0, 0.0) for 2D O2)
        # We do this by calling the helper with an empty slice[cite: 3, 4, 12, 13].
        nb_slice_empty = 1:0 
        higher_derivatives_zeros = _calculate_higher_derivatives(muscl.order, nb_slice_empty, neighbor_dfs, ws)
        
        # 3. Save these zero-derivatives and return [cite: 9, 15, 16]
        _save_derivatives!(ws, -i, slopes, higher_derivatives_zeros) 
        return
    end    

    num_nb = length(nb_slice)

    # # --- Handle zero-neighbor case ---
    # if num_nb == 0
    #     _zero_coeffs!(nb_slice, ws) # Zero coefficients
    #     slopes = D == 1 ? 0. : ntuple(x -> 0., D)
    #     higher_derivatives = _calculate_higher_derivatives(muscl.order, nb_slice, neighbor_dfs, ws) # Returns () or (0,0,0)
    #     _save_derivatives!(ws, i, slopes, higher_derivatives) # Save zero derivatives
    #     return
    # end

    # --- 1. Compute Coefficients ---
    # Dispatches based on muscl.order AND ws type implicitly
    _compute_coeffs!(muscl.order, nb_slice, ws, pg)
    slopes = _calculate_slopes(nb_slice, neighbor_dfs, ws)
    # --- 3. Limit Slopes ---
    # (Uses the existing _limit_slopes helpers)
    slopes = _limit_slopes(muscl.limiter, slopes, nb_slice, f_i, neighbor_fs, pg)
    # --- 4. Calculate Higher Derivatives ---
    # Dispatches based on muscl.order and ws type
    higher_derivatives = _calculate_higher_derivatives(muscl.order, nb_slice, neighbor_dfs, ws)
    

    # # --- 5. Store Final Derivatives --
    # Dispatches based on ws type
    _save_derivatives!(ws, i, slopes, higher_derivatives) 
    return
end

# --- In MUSCL.jl, replace the old 1D functor ---

"""
    (muscl::MUSCL{D, ORDER})(...)
Calculates the divergence for a single particle `i`
using pre-calculated slopes and neighbor data.
"""
function (muscl::MUSCL{1, ORDER})(
    eq::ScalarHyperbolicPDE,
    i::Int,                         # Current particle index
    f_i::Real,                      # Value of f at particle i
    neighbor_slice::UnitRange{Int}, # Slice into GLOBAL neighbor arrays
    pg::ParticleGrid,               # Grid object (will be 1D)
    f_neighbors::AbstractVector,    # View of neighbor f-values
    df_neighbors::AbstractVector    # View of neighbor df-values (not used by functor)
) where {ORDER<:MUSCLORDER}
    
    div = 0.0
    # Assert that the workspace is the 1D abstract type
    ws = muscl.workspace::MUSCLWorkspace1D 
    nFlux = muscl.numericalFlux

    # Get refs to global 1D buffers
    dx = get_xdistance(pg)
    nb_indices = pg.neighbor.indices

    # 1D flux
    fx = flux(eq, f_i)

    # Loop over neighbors using the global index
    @inbounds for k_global in neighbor_slice
        
        # Get data
        nbIndex = nb_indices[k_global]
        deltaPos = dx[k_global]
        f_j = f_neighbors[k_global]
        
        # Get pre-calculated coefficient from flat buffer
        coeff = ws.alfaij_bars[k_global]
        
        # This call uses pre-calculated slopes/curves from the workspace
        # It dispatches on ws's concrete type (e.g., MUSCLWorkspace1D2O)
        
        fij, fji = reconstruct_interface_states(muscl.order, ws, f_i, f_j, i, nbIndex, deltaPos)

        if muscl.mood(muscl, i, fij, neighbor_slice, fji, pg, f_neighbors)
            fij, fji = reconstruct_interface_states(MUSCLORDER0(), ws, f_i, f_j, i, nbIndex, deltaPos)
        end
        # 1D sortFlux
        fm, fp = sortFlux(fij, fji, deltaPos)
        
        # 1D divergence sum
        div += coeff * (nFlux(fm, fp, eq) - fx)
    end
    
    # The factor of 2 is part of the 1D scheme derivation
    return 2 * div
end
function (muscl::MUSCL{2, ORDER})(
    eq::ScalarHyperbolicPDE,
    i::Int,                         # Current particle index
    f_i::Float64,
    neighbor_slice::UnitRange{Int},                      # Value of f at particle i
    pg::ParticleGrid,
    f_neighbors::AbstractVector{Float64},    # View of neighbor f-values
    df_neighbors::AbstractVector{Float64},   # View of neighbor df-values
) where {ORDER<:MUSCLORDER}
    
    div = 0.0
    ws = muscl.workspace
    nFlux = muscl.numericalFlux

    dx = get_xdistance(pg)
    dy = get_ydistance(pg)
    nb_indices = pg.neighbor.indices

    fx, fy = flux(eq, f_i)
    # Loop over neighbors using the local index `k_local`
    @inbounds for k_global in neighbor_slice
        # Get global index for coefficient arrays
        
        # Get data from views
        nbIndex = nb_indices[k_global]
        deltaX = dx[k_global]
        deltaY = dy[k_global]
        @inbounds f_j = f_neighbors[k_global]
        
        # Get pre-calculated coefficients
        alfaij = ws.alfaijs[k_global]
  
        betaij = ws.betaijs[k_global]
        
# This call uses pre-calculated slopes from the workspace
        fij, fji = reconstruct_interface_states(muscl.order, ws, f_i, f_j, i, nbIndex, deltaX, deltaY)
        fmx, fpx, fmy, fpy = sortFlux(fij, fji, deltaX, deltaY)
        
        # New 2D-Simultaneous Flux Calculation
        num_fx, num_fy = nFlux(fmx, fpx, fmy, fpy, eq)
        
        div += alfaij * (num_fx - fx) + betaij * (num_fy - fy)
    end
    
    return 2. * div
end
