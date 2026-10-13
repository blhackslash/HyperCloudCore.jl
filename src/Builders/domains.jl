# =========================================================================
# MODULAR DOMAIN BUILDER
# =========================================================================

# --- Boundary Condition Parsing ---
function parse_bc(domain_conf::Dict)
    raw_bc = get(domain_conf, :bc, get(domain_conf, :bc_map, Dict{Int, AbstractBoundaryCondition}()))
    
    if raw_bc isa String
        raw_bc = eval(Meta.parse(raw_bc))
    end
    
    bc_map = Dict{Int, AbstractBoundaryCondition}()
    
    for (tag, bc_obj) in raw_bc
        if bc_obj isa AbstractBoundaryCondition
            bc_map[tag] = bc_obj
        else
            bc_sym = Symbol(bc_obj)
            if bc_sym === :outflow || bc_sym === :OutflowBC
                bc_map[tag] = OutflowBC()
            elseif bc_sym === :fixed_dirichlet || bc_sym === :FixedDirichlet
                bc_map[tag] = FixedDirichlet()
            else
                if isdefined(HyperCloudCore, bc_sym)
                    bc_map[tag] = getfield(HyperCloudCore, bc_sym)()
                elseif isdefined(Main, bc_sym)
                    bc_map[tag] = getfield(Main, bc_sym)()
                else
                    error("Boundary Condition '$bc_sym' could not be found.")
                end
            end
        end
    end
    
    return bc_map
end

# --- Main Domain Builder ---
function build_domain(domain_conf::Dict, context::Dict)

    for (_,val) in domain_conf
        if val isa GeometricDomain; return val end
    end

    if !haskey(domain_conf, :domain)
        error("Domain configuration must strictly include a :domain key (e.g., :rectangular, :spherical).")
    end

    domain_name = Symbol(domain_conf[:domain])   # accepts Symbol or String
    return build_domain(Val(domain_name), domain_conf, context)
end

# Generic fallback
build_domain(name::Val, domain_conf::Dict, context::Dict) = error("Unknown domain shape: $(typeof(name))")

# --- Specific Domain Builders ---
function build_domain(::Val{:rectangular}, domain_conf::Dict, context::Dict)
    T = real_type(context)
    
    bc_map = parse_bc(domain_conf)
    
    # Directly pull periodicity (expected to be Bool or NTuple{D, Bool})
    is_per = get(domain_conf, :periodic, false)
    
    req_mins = Tuple(T.(domain_conf[:mins]))
    req_maxs = Tuple(T.(domain_conf[:maxs]))
    
    return get_rectangular_domain(T, req_mins, req_maxs; bc_map = bc_map, is_periodic = is_per)
end

function build_domain(::Val{:spherical}, domain_conf::Dict, context::Dict)
    T = real_type(context)
    D = context[:D]::Int
    
    bc_map = parse_bc(domain_conf)
    
    # Directly pull periodicity
    is_per = get(domain_conf, :periodic, false)
    
    req_mins = Tuple(T.(domain_conf[:mins]))
    req_maxs = Tuple(T.(domain_conf[:maxs]))
    
    center = ntuple(d -> (req_mins[d] + req_maxs[d]) / 2.0, Val(D))
    radius = (req_maxs[1] - req_mins[1]) / 2.0
    
    return get_spherical_domain(T, center, radius; bc_map = bc_map, is_periodic = is_per)
end