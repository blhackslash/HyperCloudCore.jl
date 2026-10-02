# --- Representation Parsing ---
function parse_representation(pde_conf::Dict)
    rep_sym = get(pde_conf, :representation, :conservative)::Symbol
    return _parse_representation(Val(rep_sym), pde_conf)
end

_parse_representation(::Val{:conservative}, pde_conf::Dict) = Conservative()
function _parse_representation(::Val{:primitive}, pde_conf::Dict)
    path_sym = get(pde_conf, :path, :mapped)::Symbol
    return Primitive(_parse_path(Val(path_sym)))
end
function _parse_representation(::Val{:lagrangian}, pde_conf::Dict)
    path_sym = get(pde_conf, :path, :mapped)::Symbol
    return Lagrangian(_parse_path(Val(path_sym)))
end
_parse_representation(rep_val::Val, pde_conf::Dict) = error("Unknown PDE representation: $(typeof(rep_val))")

_parse_path(::Val{:line}) = LinePath()
_parse_path(::Val{:mapped}) = MappedPath()
_parse_path(::Val{:naive}) = NaiveAveragePath()
_parse_path(path_val::Val) = error("Unknown PDE path: $(typeof(path_val))")

# --- Main Equation Builder ---
function build_equation(pde_conf::Dict, context::Dict)
    if !haskey(pde_conf, :name)
        error("PDE configuration must include a strictly typed :name Symbol (e.g., :linear, :burgers).")
    end
    
    eq_name = pde_conf[:name]::Symbol
    return build_equation(Val(eq_name), pde_conf, context)
end

# Generic fallback
build_equation(eq_name::Val, pde_conf::Dict, context::Dict) = error("PDE '$(typeof(eq_name))' is not implemented.")

"""
    analytical_solution(shared_params::ParamDict)

Reads the shared simulation parameters, constructs the corresponding InitialCondition
and continuous GeometricDomain, and returns a fast, standalone closure `exact_u(st)` 
that evaluates the exact analytical solution using a unified spacetime tensor.
"""
function analytical_solution(shared_params::ParamDict)
    T = get(shared_params, :real_type, Float64) 
    
    # Initialize the global build context
    context = Dict{Symbol, Any}()
    context[:Type] = T
    
    # 1. Extract Config Namespaces
    pde_conf  = extract_namespace(shared_params, :PDE)
    grid_conf = extract_namespace(shared_params, :Grid)
    ic_conf   = extract_namespace(shared_params, :IC)
    
    if !haskey(ic_conf, :name)
        error("Analytical Factory: Missing ':name' in the IC configuration namespace (e.g. :IC_name).")
    end
    
    # 2. Instantiate the PDE struct
    eq = build_equation(pde_conf, context)
    
    # Update context with robust dimension extraction
    context[:Equation] = eq
    context[:D] = get_D(eq)
    context[:M] = get_M(eq)
    
    # 3. Build the pure mathematical geometry (handles boundaries and periodicity)
    geom = build_domain(grid_conf, context)
    context[:Domain] = geom
    
    # 4. Instantiate the InitialCondition struct using the modular builder
    ic = build_ic(ic_conf, context)
    
    # 5. Dispatch to the correct pure mathematical closure 
    return analytic_closure(eq, ic, geom)
end

# ==============================================================================
# PDE-SPECIFIC CLOSURE GENERATORS
# ==============================================================================

# Generic fallback
analytic_closure(eq::HyperbolicPDE, ic::InitialCondition, params::ParamDict) = @warn "No Analytical Solution implemented for $(typeof(eq)) with $(typeof(ic))!"

include("linear_advection.jl")