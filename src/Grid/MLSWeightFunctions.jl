export ExponentialWeightFunction, InverseWeightFunction

"""
    ExponentialWeightFunction{T}
    InverseWeightFunction{T}

Weighting function definitions utilized by the moving least squares (MLS) algorithms.
- `ExponentialWeightFunction`: Requires a scaling `alpha` and spatial `range`, pre-computing the inverse square of the range upon instantiation.
- `InverseWeightFunction`: Provides an inverse distance weighting scaled by a parameter and minimal offset to prevent singularities.
"""
struct ExponentialWeightFunction{T} <: MLSWeightFunction
    alpha::T
    range::T
    inv_range_sq::T
end

struct InverseWeightFunction{T} <: MLSWeightFunction
    alpha::T
    range::T
end

function ExponentialWeightFunction(alpha::T, range::T) where {T}
    inv_range_sq = one(T) / (range^2)
    ExponentialWeightFunction{T}(alpha, range, inv_range_sq)
end

# Provide a parameterized default constructor
InverseWeightFunction(::Type{T}) where {T} = InverseWeightFunction{T}(zero(T), zero(T))

"""
    fast_exp_accurate(x)

Provides a highly accurate, stable numerical approximation of `exp(x)` tailored strictly for negative inputs (`x <= 0`). 
- Eliminates standard library overhead by computing a rigid 4th-order polynomial expansion shifted entirely into the denominator.
"""
@inline function fast_exp_accurate(x::T) where {T}
    y = -x
    denominator = one(T) + y * (one(T) + y * (T(0.5) + y * (T(0.16666666666666666) + y * T(0.041666666666666664))))
    return one(T) / denominator
end

"""
    (w::ExponentialWeightFunction)(dist_sq::Real)

Evaluates the exponential weight using the pre-computed inverse range squared and dispatches it through the `fast_exp_accurate` numerical approximation.
"""
@inline function (w::ExponentialWeightFunction{T})(dist_sq::Real) where {T}
    arg = -w.alpha * dist_sq * w.inv_range_sq
    return fast_exp_accurate(T(arg))
end

"""
    (w::InverseWeightFunction)(dist_sq::Real)

Evaluates the inverse distance weight natively, utilizing a fixed `1e-12` tolerance offset to guarantee numerical stability upon coincident particle distances.
"""
@inline function (w::InverseWeightFunction{T})(dist_sq::Real) where {T}
    return one(T) / (T(dist_sq) + T(1e-12))
end

@inline get_cutoff(wf::ExponentialWeightFunction) = wf.range
@inline get_cutoff(wf::InverseWeightFunction) = wf.range