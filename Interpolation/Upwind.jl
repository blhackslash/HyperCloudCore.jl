using LinearAlgebra
using StaticArrays

# =========================================================================
# WORKSPACE CONSTRUCTORS & CAPACITY MANAGERS
# =========================================================================

function UpwindWorkspaceTA{D, M}(max_neighbors::Int=100) where {D, M}
    UpwindWorkspaceTA{D, M}(
        Vector{Space{D}}(undef, max_neighbors),
        Vector{State{M}}(undef, max_neighbors),
        Vector{Float64}(undef, max_neighbors),
        falses(max_neighbors),
        falses(max_neighbors)
    )
end

function UpwindWorkspaceCA{D, M}(max_neighbors::Int=100) where {D, M}
    UpwindWorkspaceCA{D, M}(
        Vector{Space{D}}(undef, max_neighbors),
        Vector{State{M}}(undef, max_neighbors),
        Vector{Float64}(undef, max_neighbors),
        Vector{Flux{D, M}}(undef, max_neighbors) # <-- Init
    )
end

function ensure_capacity!(ws::UpwindWorkspaceCA, n::Int)
    if length(ws.distVec) < n
        N = n + n ÷ 4
        resize!.((ws.distVec, ws.dfVec, ws.wVec, ws.dfFluxVec), N) # <-- Resize
    end
end

function ensure_capacity!(ws::UpwindWorkspaceTA, n::Int)
    if length(ws.distVec) < n
        N = n + n ÷ 4
        resize!.((ws.distVec, ws.dfVec, ws.wVec, ws.xWindow, ws.yWindow), N)
    end
end

function ensure_capacity!(ws::UpwindWorkspacePA, n::Int)
    return
end

# =========================================================================
# UPWIND GRADIENT SETUP
# =========================================================================

function UpwindGradient(order, dimension, M; numericalFlux::NumericalFluxFunction=UpwindFlux(), algType::String="Classic")
    @assert order >= 1 "Order must be larger or equal to one."
    
    local alg_type
    local WS_eltype::Type 
    
    if algType == "Classic"
        alg_type = ClassicAlgorithm
        WS_eltype = UpwindWorkspaceCA 
    elseif algType == "Tiwari"
        alg_type = TiwariAlgorithm
        WS_eltype = UpwindWorkspaceTA 
        @assert M == 1 "Tiwari Algorithm only supports Scalar Equations."
    elseif algType == "Praveen"
        alg_type = PraveenAlgorithm 
        WS_eltype = UpwindWorkspacePA 
        @assert order == 1 "Praveen only supports 1st order."
        @assert M == 1 "Praveen Algorithm only supports Scalar Equations."
    else
        error("Algorithm type $algType not fully configured for workspace selection.")
    end
    n_threads = Threads.nthreads()
    workspaces = [WS_eltype{dimension, M}() for _ in 1:n_threads] 

    interpolator = Interpolator{dimension, order, 1}()
    I = typeof(interpolator)

    UpwindGradient{dimension, WS_eltype{dimension, M}, I, alg_type}(order, numericalFlux, workspaces, interpolator)
end

function _init_buffers_internal!(workspaces::Vector{WS}, max_neighbors::Int) where WS <: UpwindWorkspace 
    n_threads = Threads.nthreads()
    if length(workspaces) != n_threads
        empty!(workspaces)
        for _ in 1:n_threads
            push!(workspaces, WS(max_neighbors)) 
        end
    end
    for ws in workspaces
        ensure_capacity!(ws, max_neighbors) 
    end
end

function initGIBuffers!(g::UpwindGradient, pg::ParticleGrid)
    max_nb = pg.meta.max_nb
    _init_buffers_internal!(g.workspaces, max_nb)
end

function initGI!(g::UpwindGradient, kwargs...)
    return
end

#==============================================================================
  UPWIND GRADIENT FUNCTORS
==============================================================================#

"""
Functor for UpwindGradient (ClassicAlgorithm).
Works for 1D, 2D, 3D, and natively supports both Scalars and Systems via `State{M}`.
"""
function (upwind::UpwindGradient{D, <:UpwindWorkspaceCA{D, M}, <:Any, ClassicAlgorithm})(
    eq::PDE, i::Int, f_i::State{M}, nb_slice::UnitRange{Int},       
    pg::ParticleGrid{D}, f_neighbors::AbstractVector{State{M}},    
    df_neighbors::AbstractVector{State{M}}    
 ) where {D, M, PDE <: HyperbolicPDE}
    
    num_nb = length(nb_slice)
    
    # We can extract the IO (Interpolation Order) parameter directly from the interpolator type
    if num_nb < upwind.order
        return State{M}(ntuple(_->0.0, Val(M)))
    end

    thread_idx = mod1(Threads.threadid(), Threads.nthreads())
    ws = upwind.workspaces[thread_idx]
    ensure_capacity!(ws, num_nb)

    # 1. Base Physical Flux
    F_i = flux(eq, f_i) 
    dist_all_full = get_distances(pg)
    w_all_full = get_weights(pg) 
    
    # 2. Extract, Sort, and compute Numerical Flux Matrices
    @inbounds for (local_idx, global_idx) in enumerate(nb_slice)
        dist_k = dist_all_full[global_idx]
        f_j    = f_neighbors[global_idx]
        F_j    = flux(eq, f_j)
        
        f_L, f_R, F_L, F_R = sort_flux(f_i, f_j, F_i, F_j, dist_k)
        F_num = upwind.numericalFlux(f_L, f_R, F_L, F_R, eq)
        nc_jump = evaluate_nc_jump(eq, f_L, f_R, dist_k)
        
        ws.distVec[local_idx]  = dist_k
        ws.wVec[local_idx]     = w_all_full[global_idx]
        ws.dfFluxVec[local_idx] = F_num - F_i + nc_jump
    end
    
    # 3. Dispatched Matrix Interpolation (Passing dfVec as the workspace)
    div = upwind.interpolator(
        num_nb, ws.distVec, ws.wVec, ws.dfFluxVec, ws.dfVec; 
        scale = pg.meta.dx
    )
    
    return 2.0 * div
end
"""
Functor for TiwariAlgorithm. (Restricted to Scalar PDEs)
"""
function (upwind::UpwindGradient{D, <:UpwindWorkspaceTA{D, 1}, <:Any, TiwariAlgorithm})(
    eq,
    i::Int,                         
    f_i::State{1},             
    nb_slice::UnitRange{Int},       
    pg::ParticleGrid{D},             
    f_neighbors::AbstractVector{State{1}},    
    df_neighbors::AbstractVector{State{1}}   
) where {D}
    
    # Extract the velocity vector uniformly into Space{D}
    vel = velocity(eq, f_i)
    
    thread_idx = mod1(Threads.threadid(), Threads.nthreads())
    ws = upwind.workspaces[thread_idx] 
    interp = upwind.interpolator

    dist_all_full = get_distances(pg)
    w_all_full = get_weights(pg) 

    num_nb = length(nb_slice)
    if num_nb < upwind.order; return State{1}(0.0); end
    
    ensure_capacity!(ws, num_nb) 
    
    distVec = ws.distVec
    dfVec   = ws.dfVec
    wVec    = ws.wVec 
    scale   = pg.meta.dx
    
    # 1. Unroll over spatial dimensions natively
    div_tuple = ntuple(Val(D)) do d
        stencil_size = 0 
        
        # 2. Build the upwind-only stencil for this dimension
        @inbounds for global_idx in nb_slice
            dist_k = dist_all_full[global_idx]
            
            if (vel[d] * dist_k[d] <= 0.0) 
                stencil_size += 1
                distVec[stencil_size] = dist_k
                dfVec[stencil_size]   = df_neighbors[global_idx] 
                wVec[stencil_size]    = w_all_full[global_idx]
            end
        end

        # 3. Interpolate the spatial derivative and multiply by dimension velocity
        if stencil_size >= upwind.order
            scale_d = scale[d]
            if upwind.order == 1
                res = interp(1:stencil_size, distVec, wVec, dfVec; scale = scale_d)
                dF_dx = State{1}(res[d, 1])
            else
                res_tuple = interp(1:stencil_size, distVec, wVec, dfVec; scale = scale_d)
                dF_dx = State{1}(res_tuple[1][d, 1])
            end
            return dF_dx * vel[d]
        else
            return State{1}(0.0)
        end
    end

    # Return the aggregated divergence sum
    return sum(div_tuple) 
end
"""
Functor for PraveenAlgorithm. (Restricted to Scalar PDEs in 2D)
"""
function (upwind::UpwindGradient{2, <:UpwindWorkspacePA{2}, <:Any, PraveenAlgorithm})(
    eq,
    i::Int,                         
    f_i::State{1},                  
    nb_slice::UnitRange{Int},       
    pg::ParticleGrid{2},             
    f_neighbors::AbstractVector{State{1}},    
    df_neighbors::AbstractVector{State{1}}    
)
    # Extract 2D Velocity
    vel = Space{2}(velocity(eq, f_i))

    dist_all_full = get_distances(pg)
    w_all_full = get_weights(pg)

    num_nb = length(nb_slice)
    if num_nb < 3; return State{1}(0.0); end 
    
    scale = min(pg.meta.dx[1], pg.meta.dx[2])
    if scale < 1e-14; return State{1}(0.0); end
    invL = 1.0 / scale

    N_s = @SMatrix zeros(Float64, 2, 2)
    
    @inbounds for global_idx in nb_slice
        w_k = w_all_full[global_idx]
        dist_s = dist_all_full[global_idx] * invL
        N_s += w_k * (dist_s * dist_s')
    end
    
    if abs(det(N_s)) < 1e-14; return State{1}(0.0); end

    div = State{1}(0.0)

    @inbounds for global_idx in nb_slice
        w_k    = w_all_full[global_idx] 
        dist_k = dist_all_full[global_idx] 
       
        b_s = w_k * dist_k * invL
        c_s = N_s \ b_s
        coeff = c_s * invL

        hyp = norm(dist_k)
        if hyp < 1e-14
            nx, ny = 1.0, 0.0
        else
            nx, ny = dist_k[1]/hyp, dist_k[2]/hyp
        end
    
        sx = -ny 
        sy = nx  
        
        alfaBar = dot(SVector(nx, ny), coeff)
        betaBar = dot(SVector(sx, sy), coeff)

        vel_n = dot(vel, SVector(nx, ny))
        vel_s = dot(vel, SVector(sx, sy))

        bracketMinus1 = min(vel_n, 0.0)
        bracketMinus2 = min(betaBar * vel_s, 0.0)
    
        cij = alfaBar * bracketMinus1 + bracketMinus2
        
        div += cij * df_neighbors[global_idx] 
    end
    
    return 2.0 * div
end