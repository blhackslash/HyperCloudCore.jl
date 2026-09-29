export LinearAdvection

# Diagonal Linear Advection Implementation used by Relaxation Methods

struct LinearAdvection{D, M, T, R} <: HyperbolicPDE{D, M, T, R}
    # Stores the diagonal elements as a vector for each dimension
    # This allows `eq.vel[d][k]` to work perfectly in your RelaxationSourceTerm
    vel::SVector{D, State{M, T}}
    rep::R
end

function LinearAdvection(velocities::Tuple; rep::R = Conservative()) where {R <: EquationRepresentation}
    D = length(velocities)
    M = length(velocities[1])
    T_type = eltype(velocities[1])
    
    # Store directly as a State vector (SVector)
    vel_svec = SVector{D, State{M, T_type}}(ntuple(d -> State{M, T_type}(velocities[d]), Val(D)))
    
    return LinearAdvection{D, M, T_type, R}(vel_svec, rep)
end

# =========================================================================
# LINEAR ADVECTION API IMPLEMENTATIONS
# =========================================================================

@inline prim2cons(::LinearAdvection, U::State) = U
@inline cons2prim(::LinearAdvection, W::State) = W

@inline function flux(eq::LinearAdvection{D, M, T, R}, U::State{M, T}) where {D, M, T, R}
    # Element-wise multiplication for diagonal advection
    return Flux{D, M, T}(ntuple(d -> eq.vel[d] .* U, Val(D)))
end

@inline function max_eigenvalue(eq::LinearAdvection{D, M, T, R}, U::State{M, T}, d::Int) where {D, M, T, R}
    # The maximum absolute diagonal entry
    return maximum(abs.(eq.vel[d]))
end

@inline function velocity(eq::LinearAdvection{D, M, T, R}, U::State{M, T}, d::Int) where {D, M, T, R}
    # Dynamically build the M x M SMatrix required by the path integrals
    return SMatrix{M, M, T, M*M}(ntuple(idx -> begin
        i = (idx - 1) % M + 1
        j = (idx - 1) ÷ M + 1
        i == j ? eq.vel[d][i] : zero(T)
    end, Val(M * M)))
end

"""
    kinetic_wave_speed(eq::HyperbolicPDE, d::Int, k::Int)
    
Returns the advection speed of the `k`-th kinetic component in the `d`-th spatial dimension.
Must be implemented by any PDE used as a kinetic relaxation system.
"""
function kinetic_wave_speed(eq::HyperbolicPDE, d::Int, k::Int)
    error("`kinetic_wave_speed` not implemented for $(typeof(eq)).")
end
