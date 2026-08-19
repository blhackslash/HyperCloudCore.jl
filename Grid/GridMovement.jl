struct NoGridMover <: GridMover end
struct CustomGridMover{F, P} <: GridMover
    vel_func::F
    params::P
end
struct PhysicalGridMover{D} <: GridMover
    vel_indices::NTuple{D, Int} 
end

function (gm::NoGridMover)(kwargs...)
    return nothing
end

# --- A. Macroscopic Version (No Kinetic Source Term) ---
function (gm::PhysicalGridMover{D})(pg::ParticleGrid{D, M, T}, dt::Real, ::AbstractSourceTerm) where {D, M, T}
    pos = get_positions(pg)
    rhos = pg.rhos
    
    @batch for i in 1:pg.meta.N
        u_macro = rhos[i] 
        v_vec = Space{D, T}(ntuple(d -> u_macro[gm.vel_indices[d]], Val(D)))
        
        pos[i] += v_vec * T(dt)
    end
    
    pg.neighbor(pg)
    return nothing
end

# # --- B. Kinetic Version (Relaxation Source Term) ---
# function (gm::PhysicalGridMover{D})(pg::ParticleGrid{D, K, T}, dt::Real, st::KineticSourceTerm) where {D, K, T}
#     pos = get_positions(pg)
#     rhos = pg.rhos
    
#     @batch for i in 1:pg.meta.N
#         u_macro = st.km(rhos[i]) 
#         v_vec = Space{D, T}(ntuple(d -> u_macro[gm.vel_indices[d]], Val(D)))
        
#         pos[i] += v_vec * T(dt)
#     end
    
#     pg.neighbor(pg)
#     return nothing
# end