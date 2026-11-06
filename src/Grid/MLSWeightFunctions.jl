export ExponentialWeightFunction, InverseWeightFunction

"""
    ExponentialWeightFunction{T} <: MLSWeightFunction
    ExponentialWeightFunction(alpha::T, range::T)
    (w::ExponentialWeightFunction)(dist_sq::Real) -> T

An exponential weighting function for Moving Least Squares (MLS) algorithms. It pre-computes the inverse square of the spatial `range` upon instantiation to optimize repeated distance evaluations.

# Constructors
    
    ExponentialWeightFunction(alpha::T, range::T)

- `alpha`: Scaling parameter controlling the decay rate.
- `range`: The spatial cutoff radius.

# Callable / Functor

Evaluates the weight for a given squared distance `dist_sq` (the neighbour search passes `r²`), approximately
`exp(-alpha r² / range²)`: a positive, monotone kernel through `fast_exp_accurate`, which is close to the exponential
only for small arguments (0.4 % off at the cutoff with `alpha = 1`, 23 % with `alpha = 3`). That is harmless for an
MLS weight, but the kernel is this approximation, not the exact Gaussian.
"""
struct ExponentialWeightFunction{T} <: MLSWeightFunction
    alpha::T
    range::T
    inv_range_sq::T
end

"""
    InverseWeightFunction{T} <: MLSWeightFunction
    InverseWeightFunction(alpha::T, range::T)
    (w::InverseWeightFunction)(dist_sq::Real) -> T

An inverse distance weighting function for Moving Least Squares (MLS) algorithms. 

# Constructors

    InverseWeightFunction(alpha::T, range::T)

- `alpha`: stored for a uniform interface with `ExponentialWeightFunction`; the inverse weight does not use it.
- `range`: The spatial cutoff radius.

# Callable / Functor

Evaluates the weight for a given squared distance `dist_sq`. It uses a fixed `1e-12` tolerance offset in the denominator to prevent singularities and guarantee numerical stability upon coincident particle distances.
"""
struct InverseWeightFunction{T} <: MLSWeightFunction
    alpha::T
    range::T
end
InverseWeightFunction(alpha::T, range::T) where {T <: AbstractFloat} = InverseWeightFunction{T}(alpha, range)
InverseWeightFunction(alpha::Real, range::Real) = InverseWeightFunction(promote(float(alpha), float(range))...)

function ExponentialWeightFunction(alpha::T, range::T) where {T <: AbstractFloat}
    inv_range_sq = one(T) / (range^2)
    ExponentialWeightFunction{T}(alpha, range, inv_range_sq)
end
# Mixed argument types (e.g. a Float64 alpha with a Float32 range) are promoted to a common float type (integers to Float64)
ExponentialWeightFunction(alpha::Real, range::Real) = ExponentialWeightFunction(promote(float(alpha), float(range))...)

"""
    fast_exp_accurate(x)

`1 / P₄(-x)` with `P₄` the 4th-order Taylor polynomial of `exp`, for `x <= 0`: positive and monotone, exact to
`O(x⁵)` near 0 and increasingly off for larger `|x|` (0.4 % at `x = -1`, 23 % at `x = -3`).
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