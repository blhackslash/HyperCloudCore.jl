
abstract type WENOWorkspace end
abstract type WENOGI <:GradientInterpolator end

function initGI!(weno::WENOGI, kwargs...)
    return
end

# --- 2. ensure_capacity! (Simplified) ---
# This now only needs to resize the scratch buffers.
function ensure_capacity!(ws::WENOWorkspace1D, n::Int)
    if n > length(ws.dx_stencil)
        new_capacity = n + n ÷ 4
        resize!.((ws.dx_stencil, ws.df_stencil, ws.w_stencil), new_capacity)
    end
    return nothing
end


"""
Ensures the stencil buffers in WENOWorkspace2D are large enough.
"""
function ensure_capacity!(ws::WENOWorkspace2D, n::Int)
    if n > length(ws.dx_stencil)
        new_capacity = n + n ÷ 4
        resize!.((ws.dx_stencil, ws.dy_stencil, ws.df_stencil, ws.w_stencil), new_capacity)
    end
    return nothing
end



function WENO(order::Int, dimension::Int; numericalFlux::NumericalFluxFunction = RusanovFlux())
    @assert order >= 2 "WENO requires order >= 2 for second derivatives."
    
    # Determine the workspace type based on dimension
    WS_eltype = dimension == 1 ? WENOWorkspace1D : WENOWorkspace2D
    
    # --- NEW: Create a workspace for each thread ---
    n_threads = Threads.nthreads()
    workspaces = [WS_eltype() for _ in 1:n_threads]
    # --- END NEW ---
    
    interpolator = Interpolator{dimension, order, 1}() 
    I = typeof(interpolator)
    
    # Note: The struct parameter WS is WS_eltype (e.g., WENOWorkspace2D)
    WENO{dimension, WS_eltype, I, typeof(numericalFlux)}(order, workspaces, interpolator, numericalFlux)
end

"""
Buffer initialization hook for WENO.
Finds the max neighbors from the grid and resizes all thread-local buffers.
(This is analogous to the initGIBuffers! for UpwindGradient).
"""
function initGIBuffers!(g::WENO, pg::ParticleGrid)
    # 1. Find max neighbors
    max_nb = pg.meta.max_nb[]
    
    # 2. Check for thread-count changes
    n_threads = Threads.nthreads()
    if length(g.workspaces) != n_threads
        WS_eltype = typeof(g.workspaces[1])
        empty!(g.workspaces)
        for _ in 1:n_threads
            push!(g.workspaces, WS_eltype(max_nb)) 
        end
    end
  
    # 3. Ensure capacity for all workspaces
    for ws in g.workspaces
        ensure_capacity!(ws, max_nb) 
    end
end

function (weno::WENO{1})(
    eq, #::LinearAdvection{1}
    i::Int,                         # Current particle index
    f_i::Real,                      # Value of f at particle i
    nb_slice::UnitRange{Int},       # Slice into GLOBAL neighbor arrays
    pg::ParticleGrid1D,             # Grid object
    f_neighbors::AbstractVector,    # (Not used)
    df_neighbors::AbstractVector    # Pre-gathered diffs
)
    
    # --- 1. Get Workspace, Interpolator, Velocity ---
    ws = weno.workspaces[mod1(Threads.threadid(),Threads.nthreads())]::WENOWorkspace1D # Get thread-local ws
    interp = weno.interpolator
    vel = velocity(eq, f_i)

    # --- 2. Get Global Array References ---
    dx_all_full = get_xdistance(pg)
    w_all_full = get_weights(pg) 
    
    num_neighbors = length(nb_slice)
    if num_neighbors < weno.order; return 0.0; end
    
    # --- 3. Ensure workspace capacity (for S-stencil) ---
    ensure_capacity!(ws, num_neighbors)
    
    # --- 4. Central Stencil (C-stencil) Calculation ---
    # Call the "bufferless" interpolator directly on the global arrays.
    # This assumes the interpolator is thread-safe or uses its own local buffers.
    resC_tuple = interp(nb_slice, dx_all_full, w_all_full, df_neighbors)
    resC1, resC2 = resC_tuple[1], resC_tuple[2]

    # --- 5. Build One-Sided Stencil (S-Stencil) ---
    # Get local buffer handles for the scratch space
    dx_s = ws.dx_stencil
    df_s = ws.df_stencil
    w_s  = ws.w_stencil

    stencil_size = 0
    use_left_stencil = vel > 0.0
    
    @inbounds for global_idx in nb_slice
        dx_k = dx_all_full[global_idx]
        
        # Filter-and-compact loop
        if (use_left_stencil && dx_k < 0.0) || (!use_left_stencil && dx_k >= 0.0)
            stencil_size += 1
            dx_s[stencil_size] = dx_k
            df_s[stencil_size] = df_neighbors[global_idx]
            w_s[stencil_size]  = w_all_full[global_idx]
        end
    end

    # --- 6. Call Interpolator (S-stencil) ---
    local resS1, resS2, betaS
    if stencil_size < weno.order
        betaS = 0.0 # Not enough points, disable this stencil
        resS1 = 0.0; resS2 = 0.0 # Set to zero
    else
        # Call interpolator using the populated scratch buffers
        resS_tuple = interp(1:stencil_size, dx_s, w_s, df_s)
        resS1, resS2 = resS_tuple[1], resS_tuple[2]
        
        # Calculate beta (smoothness)
        e = 1e-6
        dx2 = pg.meta.dx^2; dx4 = dx2^2
        betaS = 0.5 / ((resS1^2 * dx2 + resS2^2 * dx4 + e)^2)
    end

    # --- 7. Calculate Weights & Final Divergence ---
    e = 1e-6
    dx2 = pg.meta.dx^2; dx4 = dx2^2
    betaC = 0.5 / ((resC1^2 * dx2 + resC2^2 * dx4 + e)^2)
    
    sum_beta = betaC + betaS
    if sum_beta < 1e-14; return 0.0; end
    
    ω_s = betaS / sum_beta
    ω_c = betaC / sum_beta
    
    return (resS1*ω_s + resC1*ω_c) * vel
end


"""
Functor for 2D WENO using the 'fused' signature.
This is called ONCE per particle and calculates the full divergence.
"""
function (weno::WENO{2})(
    eq::ScalarHyperbolicPDE,
    i::Int,                         # Current particle index
    f_i::Real,                      # Value of f at particle i
    nb_slice::UnitRange{Int},       # Slice into GLOBAL neighbor arrays
    pg::ParticleGrid,               # Grid object
    f_neighbors::AbstractVector,    # (Not used)
    df_neighbors::AbstractVector    # Pre-gathered diffs (GLOBAL view)
)
    
    vel = (eq::LinearAdvection{2}).vel
    
    # --- 1. Get thread-local workspace, interpolator, and global refs ---
    thread_idx = mod1(Threads.threadid(),Threads.nthreads())
    ws = weno.workspaces[thread_idx]
    interp = weno.interpolator
    scale = min(pg.meta.dx,pg.dy)
    
    dx_all_full = get_xdistance(pg)
    dy_all_full = get_ydistance(pg)
    w_all_full = get_weights(pg) 

    num_neighbors = length(nb_slice)
    
    if num_neighbors < weno.order
        return 0.0 # Not enough points for any interpolation
    end
    
    # Get local handles to the stencil buffers
    dx_s = ws.dx_stencil
    dy_s = ws.dy_stencil
    df_s = ws.df_stencil
    w_s  = ws.w_stencil


    # Call interpolator (assumes it returns a 5-tuple for order 2)
    resC_tuple = interp(nb_slice, dx_all_full, dy_all_full, w_all_full, df_neighbors; scale = scale)
    resCx, resCy, resCxx, resCyy, resCxy = resC_tuple

    # --- 3. Horizontal Stencil (H-stencil) Calculation ---
    stencil_size_h = 0
    use_left_stencil = vel[1] > 0.0
    
    @inbounds for global_idx in nb_slice
        dx_k = dx_all_full[global_idx]
        
        # Check if neighbor is in the horizontal upwind direction
        if (use_left_stencil && dx_k < 0.0) || (!use_left_stencil && dx_k >= 0.0)
            stencil_size_h += 1
            dx_s[stencil_size_h] = dx_k
            dy_s[stencil_size_h] = dy_all_full[global_idx]
            df_s[stencil_size_h] = df_neighbors[global_idx]
            w_s[stencil_size_h]  = w_all_full[global_idx]
        end
    end

    local resHx, resHy, resHxx, resHyy, resHxy, betaH
    if stencil_size_h < weno.order
        betaH = 0.0 # Not enough points, disable this stencil
        resHx = 0.0; resHy = 0.0 # Set to zero
    else
        resH_tuple = interp(1:stencil_size_h, dx_s, dy_s, w_s, df_s; scale = scale)
        resHx, resHy, resHxx, resHyy, resHxy = resH_tuple
        
        # Calculate beta (smoothness)
        e = 1e-12
        dx2 = pg.meta.dx^2; dx4 = dx2^2 
        betaH = 0.5 / ((resHx^2 + resHy^2)*dx2 + (resHxx^2 + resHyy^2 + resHxy^2)*dx4 + e)^2
    end

    # --- 4. Vertical Stencil (V-stencil) Calculation ---
    stencil_size_v = 0
    use_top_stencil = vel[2] < 0.0 # vel[2] < 0 (down) -> use top (dy > 0)
    
    @inbounds for global_idx in nb_slice
        dy_k = dy_all_full[global_idx]
        
        # Check if neighbor is in the vertical upwind direction
        if (use_top_stencil && dy_k > 0.0) || (!use_top_stencil && dy_k <= 0.0)
            stencil_size_v += 1
            dx_s[stencil_size_v] = dx_all_full[global_idx]
            dy_s[stencil_size_v] = dy_k
            df_s[stencil_size_v] = df_neighbors[global_idx]
            w_s[stencil_size_v]  = w_all_full[global_idx]
        end
    end

    local resVx, resVy, resVxx, resVyy, resVxy, betaV
    if stencil_size_v < weno.order
        betaV = 0.0 # Not enough points, disable this stencil
        resVx = 0.0; resVy = 0.0 # Set to zero
    else
        resV_tuple = interp(1:stencil_size_v, dx_s, dy_s, w_s, df_s; scale = scale)
        resVx, resVy, resVxx, resVyy, resVxy = resV_tuple
        
        # Calculate beta (smoothness)
        e = 1e-12
        dx2 = pg.meta.dx^2; dx4 = dx2^2 
        betaV = 0.5 / ((resVx^2 + resVy^2)*dx2 + (resVxx^2 + resVyy^2 + resVxy^2)*dx4 + e)^2
    end

    # --- 5. Non-linear Weights ---
    e = 1e-12
    dx2 = pg.meta.dx^2; dx4 = dx2^2 
    betaC = 0.5 / ((resCx^2 + resCy^2)*dx2 + (resCxx^2 + resCyy^2 + resCxy^2)*dx4 + e)^2
    
    sum_beta_h = betaH + betaC
    sum_beta_v = betaV + betaC
    
    local wH, wCx, wV, wCy
    if sum_beta_h < 1e-14
        wH = 0.0; wCx = 1.0 # Fallback to central
    else
        wH = betaH / sum_beta_h; wCx = betaC / sum_beta_h
    end
    
    if sum_beta_v < 1e-14
        wV = 0.0; wCy = 1.0 # Fallback to central
    else
        wV = betaV / sum_beta_v; wCy = betaC / sum_beta_v
    end

    # --- 6. Final Divergence Calculation ---
    # No setCurvature logic, as this is now handled by initGI!
    
    # Return the combined divergence
    return (wH*resHx + wCx*resCx)*vel[1] + (wV*resVy + wCy*resCy)*vel[2]
end

# --- In your Interpolations.jl file ---

#==============================================================================
  Dumbser WENO Scheme (Optimized for SoA Grids & Workspace)
==============================================================================#
# --- In WENO.jl, replace the old DumbserWENO struct and constructor ---
function getStencil(deltaX::Real, deltaY::Real, s::Int)
    # This robust version maps the angle from atan to an integer sector [0, s-1]
    angle = atan(deltaY, deltaX)
    # Shift angle to be in [0, 2*pi]
    if angle < 0.0
        angle += 2.0 * pi
    end
    # Normalize to [0, s] and floor to get the integer index
    stencil = floor(Int, (angle * s) / (2.0 * pi))
    # Clamp to ensure it's in the range [0, s-1] due to floating point nuances
    return clamp(stencil, 0, s - 1)
end
# --- In WENO.jl ---

mutable struct DumbserWENOWorkspace
    # Scratch buffers for building a single stencil
    dx_stencil::Vector{Float64}
    dy_stencil::Vector{Float64}
    df_stencil::Vector{Float64}
    w_stencil::Vector{Float64}
    
    # Buffers specific to Dumbser WENO logic
    window_matrix::Matrix{Bool}
    gradients::Matrix{Float64}
    weights::Vector{Float64}
    
    max_neighbors::Int
    
    function DumbserWENOWorkspace(s::Int=8, initial_capacity::Int=40)
        new(
            Vector{Float64}(undef, initial_capacity), # dx_stencil
            Vector{Float64}(undef, initial_capacity), # dy_stencil
            Vector{Float64}(undef, initial_capacity), # df_stencil
            Vector{Float64}(undef, initial_capacity), # w_stencil
            falses(initial_capacity, s + 1), # s one-sided stencils + 1 central
            zeros(5, s + 1), # 5 derivatives (x, y, xx, yy, xy) for each stencil
            zeros(s + 1),
            initial_capacity
        )
    end
end

# Specialize ensure_capacity! for the new workspace
function ensure_capacity!(ws::DumbserWENOWorkspace, n::Int)
    if n > ws.max_neighbors
        ws.max_neighbors = n
        # Resize stencil buffers
        resize!(ws.dx_stencil, n); resize!(ws.dy_stencil, n);
        resize!(ws.df_stencil, n); resize!(ws.w_stencil, n);
        # Recreate window_matrix
        ws.window_matrix = falses(n, size(ws.window_matrix, 2))
    end
end

# --- In WENO.jl ---

mutable struct DumbserWENO{I <: Interpolator} <: WENOGI
    order::Int
    s::Int # amount of one-sided stencils
    workspaces::Vector{DumbserWENOWorkspace} # <-- CHANGED
    interpolator::I                          # <-- ADDED

    function DumbserWENO(order::Int=2; s::Int=8) # <-- weightFunction removed
        @assert order == 2 "DumbserWENO currently only supports order=2."
        
        n_threads = Threads.nthreads()
        workspaces = [DumbserWENOWorkspace(s) for _ in 1:n_threads]
        
        # DumbserWENO is 2D, so we hardcode dimension 2
        interpolator = Interpolator{2, order, 1}()
        I = typeof(interpolator)
        
        # --- `weightFunction` removed from new() call ---
        new{I}(order, s, workspaces, interpolator)
    end
end

# --- In WENO.jl, add these new functions ---

"""
Buffer initialization hook for DumbserWENO.
Finds the max neighbors from the grid and resizes all thread-local buffers.
"""
function initGIBuffers!(g::DumbserWENO, pg::ParticleGrid)
    # 1. Find max neighbors
    max_nb = pg.meta.max_nb
    
    # 2. Check for thread-count changes
    n_threads = Threads.nthreads()
    if length(g.workspaces) != n_threads
        empty!(g.workspaces)
        for _ in 1:n_threads
            # Recreate with correct stencil count and max_nb
            push!(g.workspaces, DumbserWENOWorkspace(g.s, max_nb)) 
        end
    end
  
    # 3. Ensure capacity for all workspaces
    for ws in g.workspaces
        ensure_capacity!(ws, max_nb) 
    end
end

# --- In WENO.jl ---

"""
Functor for 2D DumbserWENO using the 'fused' signature.
It filters pre-calculated weights for each stencil.
"""
function (weno::DumbserWENO)(
    eq::ScalarHyperbolicPDE,
    i::Int,                         # Current particle index
    f_i::Real,                      # Value of f at particle i
    nb_slice::UnitRange{Int},       # Slice into GLOBAL neighbor arrays
    pg::ParticleGrid,               # Grid object
    f_neighbors::AbstractVector,    # (Not used)
    df_neighbors::AbstractVector    # Pre-gathered diffs (GLOBAL view)
)
    
    # --- 1. Get thread-local workspace, interpolator, and global refs ---
    thread_idx = mod1(Threads.threadid(),Threads.nthreads())
    ws = weno.workspaces[thread_idx]
    interp = weno.interpolator
    vel = (eq::LinearAdvection{2}).vel
    
    num_neighbors = length(nb_slice)
    
    # Min points for 2nd order 2D is 5
    if num_neighbors < 5; return 0.0; end 

    # --- 2. Get global data and workspace buffers ---
    dx_all_full = get_xdistance(pg)
    dy_all_full = get_ydistance(pg)
    w_all_full = get_weights(pg) # <-- USE PRE-CALCULATED WEIGHTS

    # Get workspace buffers
    dx_s = ws.dx_stencil
    dy_s = ws.dy_stencil
    df_s = ws.df_stencil
    w_s  = ws.w_stencil
    windowMatrix = @view ws.window_matrix[1:num_neighbors, :]

    fill!(windowMatrix, false)
    windowMatrix[:, 1] .= true # Central stencil

    # --- 3. Populate Stencil Map ---
    # This loop is now very light, just doing stencil checks.
    @inbounds for (local_idx, global_idx) in enumerate(nb_slice)
        deltaX = dx_all_full[global_idx]
        deltaY = dy_all_full[global_idx]
        
        stencil = getStencil(deltaX, deltaY, weno.s)
        windowMatrix[local_idx, stencil + 2] = true # +1 for central, +1 for 1-based index
    end

    # --- 4. Calculate Gradients for Each Stencil ---
    for stencil_idx in 1:(weno.s + 1)
        stencil_view = @view windowMatrix[:, stencil_idx]
        
        # --- NEW: Filter-and-Compact Loop ---
        num_stencil_points = 0
        @inbounds for (local_idx, global_idx) in enumerate(nb_slice)
            if stencil_view[local_idx] # Check if this neighbor is in the stencil
                num_stencil_points += 1
                dx_s[num_stencil_points] = dx_all_full[global_idx]
                dy_s[num_stencil_points] = dy_all_full[global_idx]
                df_s[num_stencil_points] = df_neighbors[global_idx]
                w_s[num_stencil_points]  = w_all_full[global_idx] # <-- THE FIX
            end
        end
        # --- End Filter-and-Compact ---
        
        if num_stencil_points < 5
            ws.gradients[:, stencil_idx] .= 1e10 
            continue
        end
        
        try
            # Call the bufferless interpolator
            res_tuple = interp(1:num_stencil_points, dx_s, dy_s, w_s, df_s)
            
            # Store results (no scaling)
            ws.gradients[1, stencil_idx] = res_tuple[1]
            ws.gradients[2, stencil_idx] = res_tuple[2]
            ws.gradients[3, stencil_idx] = res_tuple[3]
            ws.gradients[4, stencil_idx] = res_tuple[4]
            ws.gradients[5, stencil_idx] = res_tuple[5]
        catch e
            if e isa SingularException
                ws.gradients[:, stencil_idx] .= 1e10
            else
                rethrow(e)
            end
        end
    end

    # --- 5. Compute Non-Linear Weights ---
    r = 4
    eps = 1e-14
    
    for k in 1:(weno.s + 1)
        lambda = (k == 1) ? 1e5 : 1.0 # High weight for central stencil
        smoothness = sum(ws.gradients[j, k]^2 for j in 1:5)
        ws.weights[k] = lambda / ((eps + smoothness)^r)
    end

    # Normalize weights
    sum_weights = sum(ws.weights)
    if sum_weights < 1e-14; return 0.0; end
    ws.weights ./= sum_weights
    
    # --- 6. Calculate Final Divergence ---
    div_x = dot(ws.weights, @view ws.gradients[1, :])
    div_y = dot(ws.weights, @view ws.gradients[2, :])
    
    return div_x * vel[1] + div_y * vel[2]
end