# --- Main Scheme Builder ---
function build_scheme(scheme_conf::Dict, context::Dict)
    name = scheme_conf[:name]::Symbol
    return build_scheme(Val(name), scheme_conf, context)
end

build_scheme(name::Val, conf::Dict, context::Dict) = error("Unknown Scheme: $(typeof(name))")

function build_scheme(::Val{:MUSCL}, conf::Dict, context::Dict)
    T = context[:Type]::DataType
    D = context[:D]::Int
    M = context[:M]::Int
    
    flux = context[:Flux]
    limiter = context[:Limiter]
    
    order = conf[:order]::Int
    div_order = conf[:MLS_order]::Int
    if div_order > order; div_order = order; end
    
    # 1. Generate the universal interpolator tuple
    interps = ntuple(Val(order)) do k
        k == 1 ? ConstantReconstruction() : Interpolator{D, k - 1, 1}()
    end
    
    # 2. Extract strictly-typed dimensions for pre-allocation
    max_degree = max(1, order - 1) 
    B_LEN_VAL = basis_length(Val(D), Val(max_degree))
    B_LEN = typeof(B_LEN_VAL).parameters[1] 
    
    # 3. Instantiate the pure struct
    return MUSCL{D, M, T, B_LEN, order, div_order, typeof(interps), typeof(limiter), typeof(flux)}(
        interps, limiter, flux, SVector{B_LEN, State{M, T}}[]
    )
end
function build_scheme(::Val{:Upwind}, conf::Dict, context::Dict)
    T = context[:Type]::DataType
    D = context[:D]::Int
    M = context[:M]::Int
    
    flux = context[:Flux]
    order = conf[:order]::Int
    algType = conf[:upwind_alg_nd]::Symbol 
    
    local alg_type
    if algType === :Classic
        alg_type = ClassicAlgorithm
    elseif algType === :Tiwari
        alg_type = TiwariAlgorithm
        @assert M == 1 "Tiwari Algorithm only supports Scalar Equations."
    elseif algType === :Praveen
        alg_type = PraveenAlgorithm 
        @assert order == 1 "Praveen only supports 1st order."
        @assert M == 1 "Praveen Algorithm only supports Scalar Equations."
    else
        error("Algorithm type $algType not fully configured for workspace selection.")
    end

    # FIX: Upwind directly evaluates the derivative, so degree = order
    B_LEN_VAL = basis_length(Val(D), Val(order))
    B_LEN = typeof(B_LEN_VAL).parameters[1] 

    interps = ntuple(Val(order)) do k
        Interpolator{D, k, 1}()
    end

    return UpwindDivergence{D, M, T, order, B_LEN, typeof(interps), alg_type, typeof(flux)}(
        interps, flux
    )
end

function build_scheme(::Val{:Central}, conf::Dict, context::Dict)
    T = context[:Type]::DataType
    D = context[:D]::Int
    M = context[:M]::Int
    order = conf[:order]::Int
    
    interps = ntuple(Val(order)) do k
        Interpolator{D, k, 1}()
    end
    
    return CentralDivergence{D, M, T, order, typeof(interps)}(interps)
end

function build_scheme(::Val{:WENO}, conf::Dict, context::Dict)
    T = context[:Type]::DataType
    D = context[:D]::Int
    M = context[:M]::Int
    order = conf[:order]::Int
    
    interps = ntuple(Val(order)) do k
        Interpolator{D, k, 1}()
    end
    
    return WENO{D, M, T, order, typeof(interps)}(interps)
end

# --- Flux Builder ---
function build_flux(flux_conf::Dict, context::Dict)
    name = flux_conf[:name]::Symbol
    return build_flux(Val(name), flux_conf, context)
end

build_flux(name::Val, conf::Dict, context::Dict) = error("Unknown Flux: $(typeof(name))")
build_flux(::Val{:Rusanov}, conf::Dict, context::Dict) = RusanovFlux()
build_flux(::Val{:Upwind}, conf::Dict, context::Dict)  = UpwindFlux()