"""
    extract_namespace(params::Dict, prefix::Symbol; separator::String="_")

Extracts keys starting with `prefix` followed by `separator`. 
Leaves the values exactly as they are (preserving tuples for hashing).
"""
function extract_namespace(params::Dict, prefix::Symbol; separator::String="_")
    prefix_str = string(prefix) * separator
    len = length(prefix_str)
    
    extracted = Dict{Symbol, Any}()
    for (k, v) in params
        k_str = string(k)
        if startswith(k_str, prefix_str)
            new_key = Symbol(k_str[len+1:end])
            extracted[new_key] = v
        end
    end
    return extracted
end

@inline _unwrap(v::SVector{1, T}) where {T} = v[1]
@inline _unwrap(v) = v

@inline get_D(::HyperbolicPDE{D}) where {D} = D
@inline get_M(::HyperbolicPDE{D,M}) where {D,M} = M

include("direct.jl")
include("kinetic.jl")