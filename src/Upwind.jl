# ------------------------------- Upwind -------------------------------
abstract type UpwindAlgorithm end  # Only relevant in 2D. In 1D, all algorithms are the same.
abstract type TiwariAlgorithm <: UpwindAlgorithm end  # Split domain in left and right for d/dx, and up and down for d/dy.
abstract type PraveenAlgorithm <: UpwindAlgorithm end  # Praveen C. postive upwind scheme.
abstract type NonLinearPraveenAlgorithm <: UpwindAlgorithm end  # Praveen C. postive upwind scheme.
abstract type ClassicAlgorithm <: UpwindAlgorithm end  # Take all points 'behind' center point. 
abstract type DecompositionAlgorithm <: UpwindAlgorithm end

abstract type UpwindWorkspace end

#using InteractiveUtils

"""
    sortFlux(flux_ij::Real, flux_ji::Real, deltaX::Real)::Tuple{<:Real, <:Real}

Given a reconstruction of the state at the midpoint from the cell center flux1, and a state reconstruction from the neighbouring point, return the left and right state based on the relative orientation of the points.
"""
function sortFlux(flux_ij::Float64, flux_ji::Float64, deltaX::Float64)::Tuple{Float64, Float64}
    if deltaX > 0.0
        return (flux_ij, flux_ji)  # left state, right state
    else
        return (flux_ji, flux_ij)
    end
end

"""
    sortFlux(flux_ij::Real, flux_ji::Real, deltaX::Real)::Tuple{<:Real, <:Real}

Given a reconstruction of the state at the midpoint from the cell center flux1, and a state reconstruction from the neighbouring point, return the left and right state in x and y direction.
"""
function sortFlux(flux_ij::Float64, flux_ji::Float64, deltaX::Float64, deltaY::Float64)::Tuple{Float64, Float64, Float64, Float64}
    if deltaX > 0.0 && deltaY > 0.0
        return (flux_ij, flux_ji, flux_ij, flux_ji)
    elseif deltaX > 0.0 && deltaY < 0.0 
        return (flux_ij, flux_ji, flux_ji, flux_ij)
    elseif deltaX < 0.0 && deltaY > 0.0
        return (flux_ji, flux_ij, flux_ij, flux_ji)
    else
        return (flux_ji, flux_ij, flux_ji, flux_ij)
    end
end

function populate_buffers!(dxVec, dyVec, dfVec, neighbors, xdist, ydist, fVec, vel, particleIndex)
    
    # --- STEP 1: Vectorized Calculation (Branchless) ---
    # Perform the check for all neighbors at once using dot syntax.
    # This is extremely fast and uses SIMD.
    mask = (xdist .* vel[1] .+ ydist .* vel[2]) .< 0
    
    # --- STEP 2: Find Indices That Passed the Test ---
    # `findall` returns the indices where the mask is `true`.
    passing_indices = findall(mask)
    
    # --- STEP 3: Gather Data Using the Filtered Indices ---
    # This loop is much shorter and contains no 'if' statement.
    count = length(passing_indices)
    for i in 1:count
        # The index into the original neighbor arrays
        original_idx = passing_indices[i]
        
        # Get the neighbor's global index
        nbIndex = neighbors[original_idx]
        
        # Populate the final buffers
        dxVec[i] = xdist[original_idx]
        dyVec[i] = ydist[original_idx]
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
    end
    
    return count
end

"""
A thread-local workspace for the Upwind TiwariAlgorithm.
Holds temporary buffers for all neighbors and for the filtered stencils.
"""
struct UpwindWorkspaceTA <: UpwindWorkspace
    # Buffers to hold ALL neighbor data initially (size num_neighbors)
    dxVec::Vector{Float64}
    dyVec::Vector{Float64}
    dfVec::Vector{Float64}
    wVec::Vector{Float64}  # For calculated weights

    # BitVectors to mark upwind neighbors for each direction
    xWindow::BitVector
    yWindow::BitVector

    function UpwindWorkspaceTA(max_neighbors::Int=100) # Preallocate
        new(
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            falses(max_neighbors),
            falses(max_neighbors)
        )
    end
end

"""
A minimal, thread-local workspace for the Upwind ClassicAlgorithm.
It holds temporary buffers for the filtered "upwind" neighbors.
"""
struct UpwindWorkspaceCA <: UpwindWorkspace
    dxVec::Vector{Float64} # Filtered dx (upwind)
    dyVec::Vector{Float64} # Filtered dy (upwind)
    dfVec::Vector{Float64} # Filtered df (upwind)
    wVec::Vector{Float64}  # Filtered w (upwind)

    function UpwindWorkspaceCA(max_neighbors::Int=100) # Preallocate
        new(
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors)
        )
    end
end

"""
A minimal, thread-local workspace for the Upwind PraveenAlgorithm.
Stores only essential coefficients.
"""
struct UpwindWorkspacePA <: UpwindWorkspace
    # Buffers to hold calculated coefficients (size num_neighbors)
    coeff_x_Vec::Vector{Float64}
    coeff_y_Vec::Vector{Float64}
    cijVec::Vector{Float64}     # Final coefficient

    function UpwindWorkspacePA(max_neighbors::Int=100) # Preallocate
        new(
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors),
            Vector{Float64}(undef, max_neighbors)
        )
    end
end


# Helper to ensure workspace vectors are large enough
function ensure_capacity!(ws::UpwindWorkspace, n::Int)
    if length(ws.dxVec) < n
        N = n + n ÷ 4
        resize!.((ws.dxVec, ws.dyVec, ws.dfVec, ws.fVec, ws.wVec, ws.xWindow, ws.yWindow, ws.coeff_x_Vec, ws.coeff_y_Vec, ws.aij_x_Vec, ws.aij_y_Vec, ws.nxVec, ws.nyVec, ws.ujVec, ws.nb_buffer), N)
    end
end
"""
Ensures all buffers in the minimal UpwindWorkspaceCA are large enough.
"""
function ensure_capacity!(ws::UpwindWorkspaceCA, n::Int)
    _ensure_capacity!(ws.dxVec, n)
    _ensure_capacity!(ws.dyVec, n)
    _ensure_capacity!(ws.dfVec, n)
    _ensure_capacity!(ws.wVec, n)
end
struct UpwindGradient{D, WS <: UpwindWorkspace, I <: Interpolator, Algorithm <: UpwindAlgorithm} <: GradientInterpolator
    order::Int
    numericalFlux::NumericalFluxFunction
    workspaces::Vector{WS}
    interpolator::I
    # --- Modify the UpwindGradient Constructor ---
    function UpwindGradient(order, dimension; numericalFlux::NumericalFluxFunction=UpwindFlux(), algType::String="Classic")
        @assert order >= 1 "Order must be larger or equal to one."
        
        local alg_type
        local WS_eltype::Type 
        
        if algType == "Classic"
            alg_type = ClassicAlgorithm
            WS_eltype = UpwindWorkspaceCA 
        elseif algType == "Tiwari"
            alg_type = TiwariAlgorithm
            WS_eltype = UpwindWorkspaceTA 
        elseif algType == "Praveen"
            alg_type = PraveenAlgorithm # <-- NEW
            WS_eltype = UpwindWorkspacePA # <-- NEW
            @assert order == 1
        elseif algType == "Decomposition" # <-- ADD THIS CASE
            alg_type = DecompositionAlgorithm
            WS_eltype = UpwindWorkspaceCA
            @assert dimension == 2 "DecompositionAlgorithm is for 2D only."
            @assert order == 1 "DecompositionAlgorithm currently only supports order 1."
        else
            error("Algorithm type $algType not fully configured for workspace selection.")
        end

        n_threads = Threads.nthreads()
        workspaces = [WS_eltype(100) for _ in 1:n_threads] 

        local interpolator
        if algType == "Decomposition"
            # This algorithm is 2D, but it MUST use a 1D interpolator
            interpolator = Interpolator{1, order, 1}()
        else
            # Default behavior
            interpolator = Interpolator{dimension, order, 1}()
        end
        I = typeof(interpolator)

        new{dimension, WS_eltype, I, alg_type}(order, numericalFlux, workspaces, interpolator)
    end
end

# Uses the simple _ensure_capacity! helper from your code
"""
Ensures all buffers in the UpwindWorkspaceTA are large enough.
"""
function ensure_capacity!(ws::UpwindWorkspaceTA, n::Int)
    _ensure_capacity!(ws.dxVec, n)
    _ensure_capacity!(ws.dyVec, n)
    _ensure_capacity!(ws.dfVec, n)
    _ensure_capacity!(ws.wVec, n)
    _ensure_capacity!(ws.xWindow, n)
    _ensure_capacity!(ws.yWindow, n)
end

# Uses the simple _ensure_capacity! helper
"""
Ensures all buffers in the minimal UpwindWorkspacePA are large enough.
"""
function ensure_capacity!(ws::UpwindWorkspacePA, n::Int)
    _ensure_capacity!(ws.coeff_x_Vec, n)
    _ensure_capacity!(ws.coeff_y_Vec, n)
    _ensure_capacity!(ws.cijVec, n)
end

# The _init_buffers_internal! and initGIBuffers! functions 
# will adapt correctly based on the dispatch for ensure_capacity!

# Add dispatch for the Tiwari workspace vector to the existing helper
"""
Helper function to resize a vector of workspaces.
Handles thread-count changes and resizes all individual workspaces.
"""
function _init_buffers_internal!(workspaces::Vector{WS}, max_neighbors::Int) where WS <: UpwindWorkspace # Adjust constraint if needed
    n_threads = Threads.nthreads()
    
    if length(workspaces) != n_threads
        empty!(workspaces)
        for _ in 1:n_threads
            push!(workspaces, WS(max_neighbors)) 
        end
    end
    
    for ws in workspaces
        ensure_capacity!(ws, max_neighbors) # Calls correct overload based on WS
    end
end



# The main initGIBuffers! function remains unchanged, as it uses the helper above.
"""
Buffer initialization hook for UpwindGradient. Finds the max neighbors
[cite_start]from the grid and resizes all thread-local buffers. [cite: 30, 35]
"""
function initGIBuffers!(g::UpwindGradient, pg::ParticleGrid)
    max_nb = pg.max_nb
    _init_buffers_internal!(g.workspaces, max_nb)
end


function initGI!(g::UpwindGradient, kwargs...)
    return
end
# ... (keep all other content in Interpolations.jl) ...

#==============================================================================
  UPWIND GRADIENT (Optimized for SoA Grids)
==============================================================================#

# --- REFACTORED 1D Upwind Functor (ClassicAlgorithm) ---

"""
Functor for 1D UpwindGradient (ClassicAlgorithm) using the 'fused' signature.
Calculates the upwind gradient for a single particle `i`.
"""
function (upwind::UpwindGradient{1, <:UpwindWorkspaceCA, <:Any, ClassicAlgorithm})(
    eq::PDE,
    i::Int,                         # Current particle index
    f_i::Real,                      # Value of f at particle i
    nb_slice::UnitRange{Int},       # Slice into GLOBAL neighbor arrays
    pg::ParticleGrid1D,             # Grid object
    f_neighbors::AbstractVector,    # Pre-gathered f_j
    df_neighbors::AbstractVector    # Pre-gathered f_j - f_i
)::Real where {PDE <: HyperbolicPDE}
    
    # --- 1. Get thread-local workspace, interpolator ---
    thread_idx = mod1(Threads.threadid(),Threads.nthreads())
    ws = upwind.workspaces[thread_idx]
    interp = upwind.interpolator
    nFlux = upwind.numericalFlux

    # Get references to GLOBAL grid data arrays
    dx_all_full = pg.neighbor_xdistance
    w_all_full = pg.neighbor_weights # Use pre-gathered weights

    num_nb = length(nb_slice)
    if num_nb == 0; return 0.0; end
    ensure_capacity!(ws, num_nb)

    # --- 2. The Filter & Compact Loop ---
    # We now interpolate FLUX differences, not state differences.
    # The upwind stencil is not needed; we use all neighbors
    # just like the old code.
    
    # Get the flux at the center particle
    flux_i = flux(eq, f_i)
    
    @inbounds for (local_idx, global_idx) in enumerate(nb_slice)
        dx_k = dx_all_full[global_idx]
        f_j = f_neighbors[global_idx]
        
        # Sort states for flux function
        f_L, f_R = sortFlux(f_i, f_j, dx_k)
        
        # Calculate numerical flux at the interface
        flux_num = nFlux(f_L, f_R, eq)

        # Store the flux difference in dfVec
        ws.dxVec[local_idx] = dx_k
        ws.dfVec[local_idx] = flux_num - flux_i # <-- Store F_num - F_i
        ws.wVec[local_idx]  = w_all_full[global_idx]
    end

    if num_nb < upwind.order; return 0.0; end
    
    local res1
    if upwind.order == 1
        res1 = interp(1:num_nb, ws.dxVec, ws.wVec, ws.dfVec; scale = pg.dx)
    elseif upwind.order == 2
        res_tuple = interp(1:num_nb, ws.dxVec, ws.wVec, ws.dfVec; scale = pg.dx)
        res1 = res_tuple[1]
    end
    
    # The interpolator gives res1 ≈ (F_num - F_i) / dx
    # The divergence formula is 2 * (F_num - F_i) / dx (from old code)
    return 2.0 * res1
end

"""
Helper function to resize a vector of workspaces.
Handles thread-count changes and resizes all individual workspaces.
"""
function _init_buffers_internal!(workspaces::Vector{WS}, max_neighbors::Int) where WS
    n_threads = Threads.nthreads()
    # Re-create workspaces if thread count changed
    if length(workspaces) != n_threads
        empty!(workspaces)
        for _ in 1:n_threads
            push!(workspaces, WS(max_neighbors)) # Create new ones
        end
    end
    
    # Ensure all workspaces have enough capacity
    for ws in workspaces
        ensure_capacity!(ws, max_neighbors) # Calls the correct overload
    end
end

"""
Functor for 2D UpwindGradient (ClassicAlgorithm) using the 'fused' signature.
Calculates the divergence for a single particle `i` by interpolating
flux differences, which is stable for non-linear equations.
"""
function (upwind::UpwindGradient{2, <:UpwindWorkspaceCA, <:Any, ClassicAlgorithm})(
    eq::PDE,
    i::Int,                         # Current particle index
    f_i::Real,                      # Value of f at particle i
    nb_slice::UnitRange{Int},       # Slice into GLOBAL neighbor arrays
    pg::ParticleGrid2D,             # Grid object
    f_neighbors::AbstractVector,    # Pre-gathered f_j
    df_neighbors::AbstractVector    # Pre-gathered f_j - f_i (NOT USED)
)::Real where {PDE <: ScalarHyperbolicPDE}

    # --- 1. Get workspace, interpolator, and refs ---
    thread_idx = mod1(Threads.threadid(),Threads.nthreads())
    ws = upwind.workspaces[thread_idx]
    interp = upwind.interpolator
    nFlux = upwind.numericalFlux

    dx_all_full = pg.neighbor_xdistance
    dy_all_full = pg.neighbor_ydistance
    w_all_full = pg.neighbor_weights

    num_nb = length(nb_slice)
    # Need at least 2 points for 1st order 2D LSQ
    if num_nb < upwind.order; return 0.0; end 
    
    ensure_capacity!(ws, num_nb)
    
    # Get local handles to workspace buffers
    dx_buf = ws.dxVec
    dy_buf = ws.dyVec
    w_buf = ws.wVec
    df_buf = ws.dfVec # This will hold flux differences

    # --- 2. Get central flux ---
    flux_i_x, flux_i_y = flux(eq, f_i)

    # --- 3. Fill buffers for X-Flux Interpolation ---
    @inbounds for (local_idx, global_idx) in enumerate(nb_slice)
        dx_k = dx_all_full[global_idx]
        dy_k = dy_all_full[global_idx]
        f_j = f_neighbors[global_idx]

        # Store geometry and weight
        dx_buf[local_idx] = dx_k
        dy_buf[local_idx] = dy_k
        w_buf[local_idx]  = w_all_full[global_idx]
        
        # Sort states for flux function
        fmx, fpx, fmy, fpy = sortFlux(f_i, f_j, dx_k, dy_k)
        
        # Calculate X-Flux difference
        flux_num_x = nFlux(fmx, fpx, eq, 1) # Get X-flux
        df_buf[local_idx] = flux_num_x - flux_i_x # Store Fx_num - Fx_i
    end

    # --- 4. Calculate dFx/dx ---
    scale = min(pg.dx, pg.dy)
    # Call interpolator: res_Fx = (dFx/dx, dFx/dy)
    res_Fx = interp(1:num_nb, dx_buf, dy_buf, w_buf, df_buf; scale = scale)
    dFx_dx = res_Fx[1]

    # --- 5. Fill buffer for Y-Flux Interpolation ---
    @inbounds for (local_idx, global_idx) in enumerate(nb_slice)
        # Geometry is already in dx_buf, dy_buf, w_buf
        # We just need to overwrite df_buf
        
        dx_k = dx_buf[local_idx] # Read from buffer
        dy_k = dy_buf[local_idx] # Read from buffer
        f_j = f_neighbors[global_idx] # Need to re-fetch f_j
        
        # Sort states for flux function
        fmx, fpx, fmy, fpy = sortFlux(f_i, f_j, dx_k, dy_k)
        
        # Calculate Y-Flux difference
        flux_num_y = nFlux(fmy, fpy, eq, 2) # Get Y-flux
        df_buf[local_idx] = flux_num_y - flux_i_y # Store Fy_num - Fy_i
    end

    # --- 6. Calculate dFy/dy ---
    # Call interpolator: res_Fy = (dFy/dx, dFy/dy)
    # Pass the *same* geometry buffers, but the *new* df_buf
    res_Fy = interp(1:num_nb, dx_buf, dy_buf, w_buf, df_buf; scale = scale)
    dFy_dy = res_Fy[2]
    
    # --- 7. Final Divergence ---
    # div(F) = dFx/dx + dFy/dy
    # The correct formula from the old code is 2 * div(F)
    return 2.0 * (dFx_dx + dFy_dy)
end

"""
Functor for 2D UpwindGradient (DecompositionAlgorithm) using the 'fused' signature.
This algorithm avoids cross-derivative contamination by performing two
separate 1D LSQ fits for dFx/dx and dFy/dy, using a 1D interpolator.
"""
function (upwind::UpwindGradient{2, <:UpwindWorkspaceCA, <:Interpolator{1,1,1}, DecompositionAlgorithm})(
    eq::PDE,
    i::Int,                         # Current particle index
    f_i::Real,                      # Value of f at particle i
    nb_slice::UnitRange{Int},       # Slice into GLOBAL neighbor arrays
    pg::ParticleGrid2D,             # Grid object
    f_neighbors::AbstractVector,    # Pre-gathered f_j
    df_neighbors::AbstractVector    # Pre-gathered f_j - f_i (NOT USED)
)::Real where {PDE <: ScalarHyperbolicPDE}

    # --- 1. Get workspace, interpolator, and refs ---
    thread_idx = mod1(Threads.threadid(),Threads.nthreads())
    ws = upwind.workspaces[thread_idx]
    # 'interp' is the 1D interpolator: Interpolator{1, 1, 1} [cite: 546-548]
    interp = upwind.interpolator
    nFlux = upwind.numericalFlux

    dx_all_full = pg.neighbor_xdistance
    dy_all_full = pg.neighbor_ydistance
    w_all_full = pg.neighbor_weights

    num_nb = length(nb_slice)
    # Need at least 1 point for 1D LSQ
    if num_nb < 1; return 0.0; end 
    
    ensure_capacity!(ws, num_nb)
    
    # Get local handles to workspace buffers
    dx_buf = ws.dxVec  # Will hold 'dx' for x-fit
    dy_buf = ws.dyVec  # Will hold 'dy' for y-fit
    w_buf = ws.wVec
    df_buf = ws.dfVec # Will hold flux differences

    # Get central flux
    flux_i_x, flux_i_y = flux(eq, f_i)
    
    # --- 2. Scaling factors ---
    scale_x = pg.dx
    scale_y = pg.dy
    if scale_x < 1e-14; scale_x = 1.0; end
    if scale_y < 1e-14; scale_y = 1.0; end

    # --- 3. Fill Buffers (Common to both passes) ---
    # We fill dx_buf, dy_buf, and w_buf once.
    # We fill df_buf twice (once for x, once for y).
    @inbounds for (local_idx, global_idx) in enumerate(nb_slice)
        dx_k = dx_all_full[global_idx]
        dy_k = dy_all_full[global_idx]

        # Store geometry
        dx_buf[local_idx] = dx_k
        dy_buf[local_idx] = dy_k
        w_buf[local_idx]  = w_all_full[global_idx]
    end

    # --- 4. Calculate dFx/dx (1D LSQ fit) ---
    # Fill df_buf with x-flux differences
    @inbounds for (local_idx, global_idx) in enumerate(nb_slice)
        dx_k = dx_buf[local_idx] # Read from buffer
        dy_k = dy_buf[local_idx] # Read from buffer
        f_j = f_neighbors[global_idx]
        
        fmx, fpx, fmy, fpy = sortFlux(f_i, f_j, dx_k, dy_k)
        flux_num_x = nFlux(fmx, fpx, eq, 1)
        df_buf[local_idx] = flux_num_x - flux_i_x
    end

    # Call the 1D INTERPOLATOR
    # It expects (slice, dx_buffer, w_buffer, df_buffer)
    # We must pass the 'dx' buffer.
    dFx_dx = interp(1:num_nb, dx_buf, w_buf, df_buf; scale = scale_x)

    # --- 5. Calculate dFy/dy (1D LSQ fit) ---
    # Refill df_buf with y-flux differences
    @inbounds for (local_idx, global_idx) in enumerate(nb_slice)
        dx_k = dx_buf[local_idx] # Read from buffer
        dy_k = dy_buf[local_idx] # Read from buffer
        f_j = f_neighbors[global_idx]

        fmx, fpx, fmy, fpy = sortFlux(f_i, f_j, dx_k, dy_k)
        flux_num_y = nFlux(fmy, fpy, eq, 2)
        df_buf[local_idx] = flux_num_y - flux_i_y
    end

    # Call the 1D INTERPOLATOR again.
    # This time we must pass the 'dy' buffer as the geometry.
    dFy_dy = interp(1:num_nb, dy_buf, w_buf, df_buf; scale = scale_y)

    # --- 6. Final Divergence ---
    # The correct formula from the 1D code is 2 * div(F) [cite: 741]
    return 2.0 * (dFx_dx + dFy_dy)
end

function (upwind::UpwindGradient{2, <:UpwindWorkspaceTA, <:Any, TiwariAlgorithm})(
    eq::PDE,
    i::Int,                         # Current particle index
    f_i::Real,                      # Value of f at particle i
    nb_slice::UnitRange{Int},       # Slice into GLOBAL neighbor arrays
    pg::ParticleGrid2D,             # Grid object
    f_neighbors::AbstractVector,    # (Not used)
    df_neighbors::AbstractVector,   # Pre-gathered diffs
)::Real where {PDE <: ScalarHyperbolicPDE}
    
    vel = velocity(eq,f_i)
    
    # --- 1. Get workspace, interpolator, and global refs ---
    thread_idx = mod1(Threads.threadid(),Threads.nthreads())
    ws = upwind.workspaces[thread_idx] 
    interp = upwind.interpolator

    dx_all_full = pg.neighbor_xdistance
    dy_all_full = pg.neighbor_ydistance
    w_all_full = pg.neighbor_weights 

    num_neighbors = length(nb_slice)
    
    if num_neighbors < upwind.order 
         return 0.0 
    end
    
    scale = min(pg.dx,pg.dy)
    scale = 1.
    ensure_capacity!(ws, num_neighbors) 
    
    # Get local handles to workspace buffers
    dxVec = ws.dxVec
    dyVec = ws.dyVec
    dfVec = ws.dfVec
    wVec  = ws.wVec 

    ddx = 0.0
    ddy = 0.0
    
    # --- 3. X-Derivative Calculation (Filter & Compact) ---
    stencil_size_x = 0 # Counter for x-stencil
    
    @inbounds for global_idx in nb_slice
        dx_k = dx_all_full[global_idx]
        
        # X-Upwind check
        if (vel[1] * dx_k <= 0.0) 
            stencil_size_x += 1
            dy_k = dy_all_full[global_idx]
            
            # Compact data into workspace
            dxVec[stencil_size_x] = dx_k
            dyVec[stencil_size_x] = dy_k
            dfVec[stencil_size_x] = df_neighbors[global_idx] # <-- Use global_idx
            wVec[stencil_size_x]  = w_all_full[global_idx]
        end
    end

    if stencil_size_x >= upwind.order
        # Call interpolator with workspace arrays and the calculated stencil size
        res_x = interp(1:stencil_size_x, dxVec, dyVec, wVec, dfVec; scale = scale) 
        
        # No scaling on result
        ddx = res_x[1] 
    end
        
    # --- 4. Y-Derivative Calculation (Filter & Compact) ---
    stencil_size_y = 0 # Counter for y-stencil

    @inbounds for global_idx in nb_slice
        dy_k = dy_all_full[global_idx]

        # Y-Upwind check
        if (vel[2] * dy_k <= 0.0)
            stencil_size_y += 1
            dx_k = dx_all_full[global_idx]
            
            # Compact data into workspace (overwrites X-data, which is fine)
            dxVec[stencil_size_y] = dx_k
            dyVec[stencil_size_y] = dy_k
            dfVec[stencil_size_y] = df_neighbors[global_idx] # <-- Use global_idx
            wVec[stencil_size_y]  = w_all_full[global_idx]
        end
    end
    
    if stencil_size_y >= upwind.order
        # Call interpolator with workspace arrays and the calculated stencil size
        res_y = interp(1:stencil_size_y, dxVec, dyVec, wVec, dfVec; scale = scale) 

        # No scaling on result
        ddy = res_y[2] 
    end

    # --- 5. Final Result ---
    return ddx * vel[1] + ddy * vel[2] 
end

function (upwind::UpwindGradient{2, <:UpwindWorkspacePA, <:Any, PraveenAlgorithm})(
    eq::PDE,
    i::Int,                         # Current particle index
    f_i::Real,                      # Value of f at particle i
    nb_slice::UnitRange{Int},       # Slice into GLOBAL neighbor arrays
    pg::ParticleGrid2D,             # Grid object
    f_neighbors::AbstractVector,    # (Not used directly by Praveen)
    df_neighbors::AbstractVector    # Pre-gathered view of (f_j - f_i)
)::Real where {PDE <: ScalarHyperbolicPDE}
    
    vel = velocity(eq,f_i)
    
    # --- 1. Get workspace, scaling factor, and refs ---
    thread_idx = mod1(Threads.threadid(),Threads.nthreads())
    ws = upwind.workspaces[thread_idx]

    dx_all_full = pg.neighbor_xdistance
    dy_all_full = pg.neighbor_ydistance
    w_all_full = pg.neighbor_weights

    num_neighbors = length(nb_slice)
    # Praveen needs at least 3 points for a non-singular 2D gradient
    if num_neighbors < 3; return 0.0; end 
    
    scale = min(pg.dx, pg.dy)
    if scale < 1e-14; return 0.0; end # Prevent division by zero
    invL = 1.0 / scale
    
    ensure_capacity!(ws, num_neighbors) 

    # --- 2. Build Scaled Normal Matrix N' ---
    N11_s = 0.0; N12_s = 0.0; N22_s = 0.0
    
    @inbounds for global_idx in nb_slice
        w_k = w_all_full[global_idx]
        # p'1 = dx/L, p'2 = dy/L
        dx_s = dx_all_full[global_idx] * invL
        dy_s = dy_all_full[global_idx] * invL
        
        # N'_ij = sum(w_k * p'_i * p'_j)
        N11_s += w_k * dx_s * dx_s
        N12_s += w_k * dx_s * dy_s
        N22_s += w_k * dy_s * dy_s
    end
    
    # --- 3. Hardcoded Cholesky Decomposition (N' = L'L'^T) ---
    l11_s_sq = N11_s
    if l11_s_sq < 1e-14; return 0.0; end
    l11_s = sqrt(l11_s_sq)
    inv_l11_s = 1.0 / l11_s # Store inverse for use in loop
    
    l21_s = N12_s * inv_l11_s
    
    l22_s_sq = N22_s - l21_s * l21_s # This is (Det(N') / N11_s)
    if l22_s_sq < 1e-14; return 0.0; end
    l22_s = sqrt(l22_s_sq)
    inv_l22_s = 1.0 / l22_s # Store inverse for use in loop

    # --- 4. Divergence Calculation (Loop 2) ---
    div = 0.0 # Accumulator for the final dot product

    @inbounds for global_idx in nb_slice
        w_k  = w_all_full[global_idx] 
        dx_k = dx_all_full[global_idx] # Unscaled dx
        dy_k = dy_all_full[global_idx] # Unscaled dy
        
        # --- 4a. Build Scaled RHS b'_k ---
        # b'_k = (A'^T W)_k = [w_k * (dx_k/L), w_k * (dy_k/L)]
        b1_s = w_k * dx_k * invL
        b2_s = w_k * dy_k * invL
        
        # --- 4b. Solve L'y' = b' (Forward sub) ---
        y1_s = b1_s * inv_l11_s
        y2_s = (b2_s - l21_s * y1_s) * inv_l22_s

        # --- 4c. Solve L'^T c' = y' (Backward sub) ---
        c_y_s = y2_s * inv_l22_s
        c_x_s = (y1_s - l21_s * c_y_s) * inv_l11_s

        # --- 4d. Unscale coefficients ---
        coeff_x = c_x_s * invL
        coeff_y = c_y_s * invL

        # --- 4e. Rotational math (uses unscaled geometry) ---
        hyp = hypot(dx_k, dy_k)
        nx, ny = if hyp < 1e-14
            (1.0, 0.0) # Handle dx=dy=0 case
        else
            invHyp = 1.0 / hyp
            (dx_k * invHyp, dy_k * invHyp)
        end
        sx = -ny 
        sy = nx  

        # Compute adapted coefficients (uses unscaled coefficients)
        alfaBar = nx * coeff_x + ny * coeff_y
        betaBar = sx * coeff_x + sy * coeff_y

        # Dot products with velocity
        vel_n = vel[1] * nx + vel[2] * ny
        vel_s = vel[1] * sx + vel[2] * sy

        # Calculate bracket terms
        bracketMinus1 = min(vel_n, 0.0)
        bracketMinus2 = min(betaBar * vel_s, 0.0)
    
        # Calculate final coefficient `cij`
        cij = alfaBar * bracketMinus1 + bracketMinus2
        
        # Accumulate dot product using precalculated df
        div += cij * df_neighbors[global_idx] 
    end
    
    # --- 5. Return Final Result ---
    return 2 * div
end

