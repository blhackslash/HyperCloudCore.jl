function ExponentialWeightFunction(alpha::T, range::T) where {T}
    inv_range_sq = one(T) / (range^2)
    ExponentialWeightFunction{T}(alpha, range, inv_range_sq)
end

# Provide a parameterized default constructor
InverseWeightFunction(::Type{T}) where {T} = InverseWeightFunction{T}(zero(T), zero(T))

"""
A fast, high-accuracy, and stable approximation of `exp(x)` for `x <= 0`.
"""
@inline function fast_exp_accurate(x::T) where {T}
    y = -x
    denominator = one(T) + y * (one(T) + y * (T(0.5) + y * (T(0.16666666666666666) + y * T(0.041666666666666664))))
    return one(T) / denominator
end

@inline function (w::ExponentialWeightFunction{T})(dist_sq::Real) where {T}
    arg = -w.alpha * dist_sq * w.inv_range_sq
    return fast_exp_accurate(T(arg))
end

@inline function (w::InverseWeightFunction{T})(dist_sq::Real) where {T}
    return one(T) / (T(dist_sq) + T(1e-12))
end