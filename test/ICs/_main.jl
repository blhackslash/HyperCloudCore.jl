abstract type InitialCondition end
abstract type SmoothInitialCondition <: InitialCondition end

function set_initial_conditions!(pg::ParticleGrid{D, M}, eq::HyperbolicPDE, IC::InitialCondition) where {D, M}
    positions = pg.core.positions
    
    @inbounds for i in 1:pg.meta.N
        # Strict assignment: IC(pos) must return an State{M}
        pg.rhos[i] = IC(positions[i])
    end
    return nothing
end

"""
    build_ic(params::ParamDict, D::Int, ::Type{T}) where {T}

Reads the parameters dict and dispatches to the correct IC builder using `Val`.
"""
function build_ic(ic_conf::Dict, context::Dict)
    if !haskey(ic_conf, :name)
        error("Initial Condition configuration must include a strictly typed :name Symbol (e.g., :gauss, :box).")
    end
    
    ic_name = ic_conf[:name]::Symbol
    return build_ic(Val(ic_name), ic_conf, context)
end

# Fallback error for missing implementations
build_ic(name::Val, p, D::Int, ::Type{T}) where {T} = error("Unknown initFunc name: $(typeof(name))")

include("box.jl")
include("gauss.jl")