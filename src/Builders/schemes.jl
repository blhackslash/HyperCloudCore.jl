# --- Main Scheme Builder ---
"""
    build_scheme(scheme_conf, context) -> DivergenceInterpolator
    build_scheme(::Val{name}, scheme_conf, context)

`Scheme` namespace. Requires `real_type(context)`, `[:D]` and `[:M]`. Selector `:name`:

| `:name` | Scheme | Also reads |
|:--|:--|:--|
| `:MUSCL` | [`MUSCL`](@ref) | `context[:Flux]`, `context[:Limiter]`; allocates on [`storage_type`](@ref)`(context)` |
| `:Upwind` | [`UpwindDivergence`](@ref) | `context[:Flux]`; `:upwind_alg_nd` (`:Classic`, `:Tiwari` or `:Praveen`) |
| `:Central` | [`CentralDivergence`](@ref) | |
| `:WENO` | [`WENO`](@ref) | |

All take the consistency order `:order` (required) and the divergence order `:MLS_order` (the `div_order`
keyword, default 0: follow the particle's order).

Add a scheme with a method `build_scheme(::Val{:my_scheme}, scheme_conf, context)` returning a
[`DivergenceInterpolator`](@ref) (see [API Contracts](@ref)).
"""
function build_scheme end

function build_scheme(conf::Dict, context::Dict)
    name = conf[:name]::Symbol
    return build_scheme(Val(name), conf, context)
end

build_scheme(name::Val, conf::Dict, context::Dict) = error("Unknown Scheme: $(typeof(name))")

function build_scheme(::Val{:MUSCL}, conf::Dict, context::Dict)
    T = real_type(context)
    D = context[:D]::Int
    M = context[:M]::Int
    
    flux = context[:Flux]
    limiter = context[:Limiter]
    
    order = conf[:order]::Int
    div_order = get(conf,:MLS_order,0)::Int
    streamed = get(conf, :streamed, false)::Bool
    moment_correction = get(conf, :moment_correction, false)::Bool

    return MUSCL(T, D, M, order, limiter, flux; div_order = div_order, storage = storage_type(context), streamed = streamed,
                 moment_correction = moment_correction)
end

function build_scheme(::Val{:Upwind}, conf::Dict, context::Dict)
    T = real_type(context)
    D = context[:D]::Int
    M = context[:M]::Int
    
    flux = context[:Flux]
    algType = conf[:upwind_alg_nd]::Symbol 

    order = conf[:order]::Int
    div_order = get(conf,:MLS_order,0)::Int 
    
    return UpwindDivergence(T, D, M, order, algType, flux; div_order=div_order)
end

function build_scheme(::Val{:Central}, conf::Dict, context::Dict)
    T = real_type(context)
    D = context[:D]::Int
    M = context[:M]::Int
    
    order = conf[:order]::Int
    div_order = get(conf,:MLS_order,0)::Int 
    
    return CentralDivergence(T, D, M, order; div_order=div_order)
end

function build_scheme(::Val{:WENO}, conf::Dict, context::Dict)
    T = real_type(context)
    D = context[:D]::Int
    M = context[:M]::Int
    
    order = conf[:order]::Int
    div_order = get(conf,:MLS_order,0)::Int 
    
    return WENO(T, D, M, order; div_order=div_order)
end

# --- Flux Builder ---
"""
    build_flux(flux_conf, context) -> NumericalFluxFunction
    build_flux(::Val{name}, flux_conf, context)

`Flux` namespace. Selector `:name`: `:Rusanov` ([`RusanovFlux`](@ref)), `:AlignedRusanov` ([`AlignedRusanovFlux`](@ref)) or `:Upwind` ([`UpwindFlux`](@ref)).
"""
function build_flux end

function build_flux(flux_conf::Dict, context::Dict)
    name = flux_conf[:name]::Symbol
    return build_flux(Val(name), flux_conf, context)
end

build_flux(name::Val, conf::Dict, context::Dict) = error("Unknown Flux: $(typeof(name))")
build_flux(::Val{:Rusanov}, conf::Dict, context::Dict) = RusanovFlux()
build_flux(::Val{:AlignedRusanov}, conf::Dict, context::Dict) = AlignedRusanovFlux()
build_flux(::Val{:Upwind}, conf::Dict, context::Dict)  = UpwindFlux()