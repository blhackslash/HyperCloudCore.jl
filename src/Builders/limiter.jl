# --- Limiter Builder ---
"""
    build_limiter(limiter_conf, context) -> AbstractSlopeLimiter
    build_limiter(::Val{name}, limiter_conf, context)

`Limiter` namespace. Selector `:name` (default `:none`): `:none` (`NoLimiter()`), `:minmod`,
`:superbee` (1-D only), `:VK` (Venkatakrishnan) or `:BJ` (Barth–Jespersen). Every limiter except `:none`
requires `:mode` (`:hard` limits all reconstruction coefficients, `:soft` only the first-order ones). See
[MUSCL Reconstruction and Limiting](@ref).
"""
function build_limiter end

function build_limiter(limiter_conf::Dict, context::Dict)
    name = get(limiter_conf,:name,:none)::Symbol
    return build_limiter(Val(name), limiter_conf, context)
end

build_limiter(name::Val, conf::Dict, context::Dict) = error("Unknown Limiter: $(typeof(name))")
build_limiter(::Val{:none}, conf::Dict, context::Dict) = NoLimiter()

function build_limiter(::Val{:minmod}, conf::Dict, context::Dict)
    return MinmodLimiter(conf[:mode]::Symbol)
end
function build_limiter(::Val{:superbee}, conf::Dict, context::Dict)
    return SuperbeeLimiter(conf[:mode]::Symbol)
end
function build_limiter(::Val{:VK}, conf::Dict, context::Dict)
    return VenkatakrishnanLimiter(conf[:mode]::Symbol)
end
function build_limiter(::Val{:BJ}, conf::Dict, context::Dict)
    return BarthJespersenLimiter(conf[:mode]::Symbol)
end
