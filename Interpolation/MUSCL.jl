# =========================================================================
# UNIFIED MUSCL WORKSPACE
# =========================================================================
function MUSCLWorkspace{D, M, B_LEN}(max_nb::Int=100) where {D, M, B_LEN}
    MUSCLWorkspace{D, M, B_LEN}(
        Vector{Space{D}}(undef, max_nb),
        Vector{Float64}(undef, max_nb),
        Vector{State{M}}(undef, max_nb),
        Vector{Flux{D, M}}(undef, max_nb),
        SVector{B_LEN, State{M}}[] # Starts empty, resized to N dynamically
    )
end

function ensure_capacity!(ws::MUSCLWorkspace, num_nb::Int)
    if length(ws.distVec) < num_nb
        N = num_nb + num_nb ÷ 4
        resize!.((ws.distVec, ws.wVec, ws.dfVec, ws.dfFluxVec), N)
    end
end

function initGIBuffers!(ws::MUSCLWorkspace{D, M, B_LEN}, pg::ParticleGrid) where {D, M, B_LEN}
    N = pg.meta.N
    if length(ws.gradients) < N
        # Fill with zero-gradients initially to ensure type stability
        resize!(ws.gradients, N)
        fill!(ws.gradients, zero(SVector{B_LEN, State{M}}))
    end
end

function MUSCL(
    dimension::Int, M::Int, order::Int; 
    limiter=NoLimiter(), numericalFlux=RusanovFlux(), mood=NoMOOD()
)
    # Statically determine the basis length
    B_LEN_VAL = basis_length(Val(dimension), Val(order))
    B_LEN = typeof(B_LEN_VAL).parameters[1] 
    
    n_threads = Threads.nthreads()
    workspaces = [MUSCLWorkspace{dimension, M, B_LEN}(100) for _ in 1:n_threads]
    interp = Interpolator{dimension, order, 1}()
    
    # Pass `order` directly into the type signature!
    return MUSCL{dimension, M, B_LEN, order, typeof(interp), typeof(limiter), typeof(numericalFlux), typeof(mood)}(
        interp, limiter, numericalFlux, mood, workspaces
    )
end

function initGIBuffers!(g::MUSCL, pg::ParticleGrid)
    for ws in g.workspaces
        initGIBuffers!(ws, pg)
    end
end

# =========================================================================
# PRE-GATHER PASS: CALCULATE AND STORE GRADIENTS
# =========================================================================

function initGI!(
    muscl::MUSCL{D, M, B_LEN, ORDER}, i::Int, f_i::State{M}, nb_slice::UnitRange{Int},
    pg::ParticleGrid{D}, f_neighbors::AbstractVector{State{M}}, df_neighbors::AbstractVector{State{M}}
) where {D, M, B_LEN, ORDER}
    
    num_nb = length(nb_slice)
    thread_idx = mod1(Threads.threadid(), Threads.nthreads())
    ws = muscl.workspaces[thread_idx]
    
    # Boundary/Empty check
    if i < 0 || num_nb < ORDER
        ws.gradients[abs(i)] = zero(SVector{B_LEN, State{M}})
        return
    end

    ensure_capacity!(ws, num_nb)

    dist_all = get_distances(pg)
    w_all = get_weights(pg)
    
    @inbounds for (local_idx, global_idx) in enumerate(nb_slice)
        ws.distVec[local_idx] = dist_all[global_idx]
        ws.wVec[local_idx]    = w_all[global_idx]
        ws.dfVec[local_idx]   = df_neighbors[global_idx]
    end

    # 1. Calculate generalized spatial gradient (slopes, curves, etc. in one SVector!)
    raw_grad = muscl.interpolator(
        1:num_nb, ws.distVec, ws.wVec, ws.dfVec; scale = minimum(pg.meta.dx)
    )
    
    # 2. Limit the slopes 
    # (Assuming you update `_limit_slopes` to accept and return the SVector{B_LEN, State{M}})
    limited_grad = _limit_slopes(muscl.limiter, raw_grad, nb_slice, f_i, f_neighbors, pg, ws.distVec)

    # 3. Store for the Flux pass
    ws.gradients[i] = limited_grad
    return
end

# =========================================================================
# FLUX PASS: RECONSTRUCT INTERFACES AND COMPUTE DIVERGENCE
# =========================================================================

function (muscl::MUSCL{D, M, B_LEN, ORDER})(
    eq::HyperbolicPDE, i::Int, f_i::State{M}, nb_slice::UnitRange{Int},       
    pg::ParticleGrid{D}, f_neighbors::AbstractVector{State{M}}, df_neighbors::AbstractVector{State{M}}    
) where {D, M, B_LEN, ORDER}

    num_nb = length(nb_slice)
    
    # Use static ORDER
    if num_nb < ORDER; return State{M}(ntuple(_->0.0, Val(M))); end

    thread_idx = mod1(Threads.threadid(), Threads.nthreads())
    ws = muscl.workspaces[thread_idx]
    ensure_capacity!(ws, num_nb)

    grad_i = ws.gradients[i]
    F_i    = flux(eq, f_i)
    
    dist_all = get_distances(pg)
    w_all    = get_weights(pg)
    
    @inbounds for (local_idx, global_idx) in enumerate(nb_slice)
        dist_k = dist_all[global_idx]
        f_j    = f_neighbors[global_idx]
        grad_j = ws.gradients[pg.neighbor.indices[global_idx]]
        
        # Val(ORDER) is now a strict compile-time constant!
        p_interface_i = build_basis(Val(ORDER),  0.5 * dist_k)
        p_interface_j = build_basis(Val(ORDER), -0.5 * dist_k)

        # Using your optimized SVector broadcasting (or dot product)
        fij = f_i + sum(grad_i .* p_interface_i)
        fji = f_j + sum(grad_j .* p_interface_j)
        
        # MOOD check can go here if needed!
        
        # 3. Evaluate physical fluxes at the interface states
        F_ij = flux(eq, fij)
        F_ji = flux(eq, fji)
        
        # 4. Sort and apply Numerical Flux (Identical to Upwind!)
        f_L, f_R, F_L, F_R = sort_flux(fij, fji, F_ij, F_ji, dist_k)
        F_num = muscl.numericalFlux(f_L, f_R, F_L, F_R, eq)
        
        # 5. Populate workspace for final divergence integration
        ws.distVec[local_idx]   = dist_k
        ws.wVec[local_idx]      = w_all[global_idx]
        ws.dfFluxVec[local_idx] = F_num - F_i
    end
    
    # 6. Integrate the numerical fluxes using the standard interpolator
    div = muscl.interpolator(
        num_nb, ws.distVec, ws.wVec, ws.dfFluxVec, ws.dfVec; scale = pg.meta.dx
    )
    
    return 2.0 * div
end