"""
    InteractionBuffer{D, M, T}

A thread-safe, pre-allocated workspace designed to hold neighbor interaction data during flux evaluations.

# Fields
- `f::Vector{State{M, T}}`: Stores the direct state of neighboring particles.
- `df::Vector{State{M, T}}`: Stores the raw state differences between neighbors and the target particle.
- `df_flux::Vector{Flux{D, M, T}}`: Stores computed numerical flux differences or non-conservative jumps.
- `df_scratch::Vector{State{M, T}}`: An auxiliary buffer for intermediate moving least squares operations.
- `mask::Vector{Bool}`: A boolean array used to filter specific neighbors dynamically during directional stencil building.
"""
struct InteractionBuffer{D, M, T}
    f::Vector{State{M, T}}
    df::Vector{State{M, T}}
    df_flux::Vector{Flux{D, M, T}}
    df_scratch::Vector{State{M, T}} 
    mask::Vector{Bool} 
    
    InteractionBuffer{D, M, T}() where {D, M, T} = new{D, M, T}(
        State{M, T}[], State{M, T}[], Flux{D, M, T}[], State{M, T}[], Bool[]
    )
end

function update_size!(ib::InteractionBuffer, num_interactions::Int)
    ensure_capacity!(ib.f, num_interactions)
    ensure_capacity!(ib.df, num_interactions)
    ensure_capacity!(ib.df_flux, num_interactions)
    ensure_capacity!(ib.df_scratch, num_interactions)
    ensure_capacity!(ib.mask, num_interactions)
    return nothing
end

@inline function update_content!(
    ib::InteractionBuffer{D, M, T},
    nb_indices::AbstractVector{Int}, 
    f_i::State{M, T}, 
    nb_slice::UnitRange{Int}, 
    fVec::AbstractVector{State{M, T}}
) where {D, M, T}
    
    @inbounds for k in nb_slice
        j = nb_indices[k]
        f_j = fVec[j] 
        
        ib.f[k]  = f_j
        ib.df[k] = f_j - f_i 
    end
    return nothing
end
# =========================================================================
# INTERNAL SOURCE TUPLE UNROLLERS (Zero-Cost Execution Wrappers)
# =========================================================================

@inline @generated function evaluate_sources(sts::Tuple{Vararg{AbstractSourceTerm}}, U, p_idx::Int, pg, t::Real)
    N = length(sts.parameters)
    if N == 0
        return :(zero(U))
    elseif N == 1
        return :(evaluate_source(sts[1], U, p_idx, pg, t))
    else
        expr = :(evaluate_source(sts[1], U, p_idx, pg, t))
        for i in 2:N
            expr = :($expr + evaluate_source(sts[$i], U, p_idx, pg, t))
        end
        return expr
    end
end

@inline function pre_solve_updates!(sts::Tuple{Vararg{AbstractImplicitSourceTerm}}, Y_stage, pg, t::Real)
    for st in sts
        pre_solve_update!(st, Y_stage, pg, t)
    end
end

@inline @generated function implicit_solve(sts::Tuple{Vararg{AbstractImplicitSourceTerm}}, U_in, dt_coeff::Real, p_idx::Int, pg, t::Real)
    N = length(sts.parameters)
    if N == 0; return :(U_in); end
    quote
        U_out = U_in
        Base.Cartesian.@nexprs $N i -> U_out = implicit_solve(sts[i], U_out, dt_coeff, p_idx, pg, t)
        return U_out
    end
end

include("RKButcherTableaus.jl")
include("IMEXButcherTableaus.jl")
include("MeshfreeRKTimeSteppers.jl")
include("MeshfreeIMEXTimeSteppers.jl")
include("MOOD.jl")