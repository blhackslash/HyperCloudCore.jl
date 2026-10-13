# --- Main Scheme Builder ---
function build_scheme(conf::Dict, context::Dict)
    name = conf[:name]::Symbol
    return build_scheme(Val(name), conf, context)
end

build_scheme(name::Val, conf::Dict, context::Dict) = error("Unknown Scheme: $(typeof(name))")

function build_scheme(::Val{:MUSCL}, conf::Dict, context::Dict)
    T = context[:Type]::DataType
    D = context[:D]::Int
    M = context[:M]::Int
    
    flux = context[:Flux]
    limiter = context[:Limiter]
    
    order = conf[:order]::Int
    div_order = get(conf,:MLS_order,0)::Int 
    
    return MUSCL(T, D, M, order, limiter, flux; div_order=div_order)
end

function build_scheme(::Val{:Upwind}, conf::Dict, context::Dict)
    T = context[:Type]::DataType
    D = context[:D]::Int
    M = context[:M]::Int
    
    flux = context[:Flux]
    algType = conf[:upwind_alg_nd]::Symbol 

    order = conf[:order]::Int
    div_order = get(conf,:MLS_order,0)::Int 
    
    return UpwindDivergence(T, D, M, order, algType, flux; div_order=div_order)
end

function build_scheme(::Val{:Central}, conf::Dict, context::Dict)
    T = context[:Type]::DataType
    D = context[:D]::Int
    M = context[:M]::Int
    
    order = conf[:order]::Int
    div_order = get(conf,:MLS_order,0)::Int 
    
    return CentralDivergence(T, D, M, order; div_order=div_order)
end

function build_scheme(::Val{:WENO}, conf::Dict, context::Dict)
    T = context[:Type]::DataType
    D = context[:D]::Int
    M = context[:M]::Int
    
    order = conf[:order]::Int
    div_order = get(conf,:MLS_order,0)::Int 
    
    return WENO(T, D, M, order; div_order=div_order)
end

# --- Flux Builder ---
function build_flux(flux_conf::Dict, context::Dict)
    name = flux_conf[:name]::Symbol
    return build_flux(Val(name), flux_conf, context)
end

build_flux(name::Val, conf::Dict, context::Dict) = error("Unknown Flux: $(typeof(name))")
build_flux(::Val{:Rusanov}, conf::Dict, context::Dict) = RusanovFlux()
build_flux(::Val{:Upwind}, conf::Dict, context::Dict)  = UpwindFlux()