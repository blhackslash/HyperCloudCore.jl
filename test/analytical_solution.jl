# ==============================================================================
# MAIN FACTORY FUNCTION
# ==============================================================================

"""
    generate_analytical_solution(shared_params::ParamDict)

Reads the shared simulation parameters, constructs the corresponding InitialCondition
struct, and returns a fast, standalone closure `exact_u(st)` that evaluates the 
exact analytical solution using a unified spacetime tensor.
"""
function analytical_solution(shared_params::ParamDict)
    # 1. Instantiate the PDE struct directly using the provided build_equation
    # We use Float64 as the standard type for analytical solutions
    eq, D, NM, vel_var = build_equation(shared_params, Float64) 
    
    init_func = get(shared_params, :init_func, nothing)
    init_params = get(shared_params, :init_params, nothing)
    
    isnothing(init_func) && error("Analytical Factory: Missing 'init_func' in shared_params")
    
    # 2. Instantiate the InitialCondition struct using your existing parser
    ic = getInitialCondition(init_func, init_params)
    
    # 3. Dispatch to the correct pure mathematical closure using the PDE type
    return _build_analytic_closure(eq, ic, shared_params)
end

# ==============================================================================
# PDE-SPECIFIC CLOSURE GENERATORS
# ==============================================================================

# Generic fallback
_build_analytic_closure(eq::HyperbolicPDE, ic::InitialCondition, params::ParamDict) = @warn "No Analytical Solution implemented for $(typeof(eq)) with $(typeof(ic))!"

# ---------------------------------------------------------
# 1. Linear Advection (N-Dimensional)
# ---------------------------------------------------------
function _build_analytic_closure(eq::LinearAdvection{D}, ic::InitialCondition, params::ParamDict) where {D}
    # Extract velocity vector from parameters (can also be extracted from eq.vel)
    vel_param = params[:PDE_params]
    vel = SVector{D, Float64}(ntuple(d -> Float64(vel_param[d][1]), Val(D)))
    
    is_per = params[:periodic]
    @assert is_per isa Bool "Analytical Solutions are only supported for full periodic or full non-periodic BCs for now!"
    # Extract domain boundaries
    mins_tup = get(params, :mins, ntuple(d->0.0, D))
    maxs_tup = get(params, :maxs, ntuple(d->1.0, D))
    
    mins = SVector{D, Float64}(mins_tup...)
    maxs = SVector{D, Float64}(maxs_tup...)
    
    return function exact_linear(st::SVector)
        t = st[end]
        pos = SVector{D, Float64}(ntuple(d -> st[d], Val(D))) 
        
        pos0 = pos - vel * t
        if is_per
            pos0 = mins .+ mod.(pos0 .- mins, maxs .- mins)
        end
        return ic(pos0)
    end
end
