# =========================================================================
# 1. NoGridMover
# =========================================================================

function (gm::NoGridMover)(kwargs...)
    return nothing
end

# =========================================================================
# 2. PhysicalGridMover (Inline, Zero-Allocation)
# =========================================================================

# --- A. Macroscopic Version (No Kinetic Source Term) ---
function (gm::PhysicalGridMover{D})(pg::ParticleGrid{D, M}, dt::Real, ::AbstractSourceTerm) where {D, M}
    pos = get_positions(pg)
    rhos = pg.rhos
    
    @batch for i in 1:pg.meta.N
        u_macro = rhos[i] # Already macroscopic!
        
        # Extract velocity components based on the configured indices
        v_vec = Space{D}(ntuple(d -> u_macro[gm.vel_indices[d]], Val(D)))
        
        # Update physical position natively
        pos[i] += v_vec * dt
    end
    
    pg.neighbor(pg)
    return nothing
end

# --- B. Kinetic Version (Relaxation Source Term) ---
function (gm::PhysicalGridMover{D})(pg::ParticleGrid{D,K}, dt::Real, st::KineticSourceTerm) where {D, K}
    pos = get_positions(pg)
    rhos = pg.rhos
    
    @batch for i in 1:pg.meta.N
        # 1. Reconstruct Macroscopic State (Returns State{M})
        u_macro = st.kin2macro(rhos[i]) 
        
        # 2. Extract Velocity components
        v_vec = Space{D}(ntuple(d -> u_macro[gm.vel_indices[d]], Val(D)))
        
        # 3. Update physical position natively
        pos[i] += v_vec * dt
    end
    
    pg.neighbor(pg)
    return nothing
end