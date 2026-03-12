function CentralGradient(order::Int, dimension::Int)
    @assert order >= 1 "Order must be 1 or greater."       

    interpolator = Interpolator{dimension, order, 1}()
    I = typeof(interpolator)

    new{dimension, I}(order, interpolator)
end
# --- 4. Buffer Initialization Hooks (Adapted from Upwind.jl) ---

"""
Buffer initialization hook for CentralGradient. Finds the max neighbors
from the grid and resizes all thread-local buffers.
"""
function initGIBuffers!(g::CentralGradient, pg::ParticleGrid)
    return
end

"""
initGI! is a no-op for this interpolator, as all calculations
are performed in the functor.
"""
function initGI!(g::CentralGradient, kwargs...)
    return
end

# --- 5. Refactored CentralGradient Functors (Fused-Loop Signature) ---

"""
(1D Functor) Calculates the central gradient divergence for particle `i`.
"""
function (central::CentralGradient{1})(
    eq::PDE,
    i::Int,                         # Current particle index
    f_i::Real,                      # Value of f at particle i
    nb_slice::UnitRange{Int},       # Slice into GLOBAL neighbor arrays
    pg::ParticleGrid1D,             # Grid object
    f_neighbors::AbstractVector,    # (Not used)
    df_neighbors::AbstractVector    # Pre-gathered diffs
)::Real where {PDE <: ScalarHyperbolicPDE} # Use ScalarHyperbolicPDE for velocity

    vel = velocity(eq, f_i)

    interp = central.interpolator

    # Get references to GLOBAL grid data arrays
    dxVec = get_xdistance(pg)
    wVec = get_weights(pg)

    num_nb = length(nb_slice)
    
    # Check if we have enough neighbors for the interpolation order
    if num_nb < central.order; return 0.0; end

    # --- 4. Call Interpolator ---
    # Note: Curvature (res2) is calculated but not stored here,
    # matching the Upwind.jl functor's structure.
    res = interp(nb_slice, dxVec, wVec, df_neighbors)

    # --- 5. Return Scaled Divergence ---
    return vel * res[1]
end


"""
(2D Functor) Calculates the central gradient divergence for particle `i`.
"""
function (central::CentralGradient{2})(
    eq::PDE,
    i::Int,                         # Current particle index
    f_i::Real,                      # Value of f at particle i
    nb_slice::UnitRange{Int},       # Slice into GLOBAL neighbor arrays
    pg::ParticleGrid2D,             # Grid object
    f_neighbors::AbstractVector,    # (Not used)
    df_neighbors::AbstractVector    # Pre-gathered diffs
)::Real where {PDE <: ScalarHyperbolicPDE}

    vel = velocity(eq, f_i)

    interp = central.interpolator

    # Get references to GLOBAL grid data arrays
    dxVec = get_xdistance(pg)
    dyVec = get_ydistance(pg)
    wVec = get_weights(pg)

    num_nb = length(nb_slice)
    
    if num_nb < central.order; return 0.0; end

    # --- 4. Call Interpolator ---
    # Note: Curvature (res3, res4) is calculated but not stored here.
    res = interp(nb_slice, dxVec, dyVec, wVec, df_neighbors)
    
    # --- 5. Return Scaled Divergence ---
    return (vel[1] * res[1] + vel[2] * res[2])
end