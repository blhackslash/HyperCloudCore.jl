# MLS Weight Functions

Weight functions dictate the spatial decay of influence between neighboring particles within the Moving Least Squares (MLS) interpolator. They are essential for computing local polynomial approximations and managing the interaction radius of the meshfree topology.

## Built-In Weight Functions

The solver provides two heavily optimized weighting strategies:

*   **`ExponentialWeightFunction`**: An exponential weighting function parameterized by a scaling factor `alpha` and a spatial cutoff `range`. 
    *   **Optimization**: Upon instantiation, it pre-computes the inverse square of the spatial range to optimize repeated distance evaluations. 
    *   **Logic**: When evaluating a squared distance, it bypasses the standard library `exp()` function, instead dispatching through a highly accurate, stable 4th-order polynomial approximation of `exp(x)` tailored strictly for negative inputs (`x <= 0`). This shifts the expansion entirely into the denominator to eliminate standard library overhead.
*   **`InverseWeightFunction`**: An inverse distance weighting function, also parameterized by `alpha` and a spatial `range`.
    *   **Logic**: Evaluates the weight natively for a given squared distance, utilizing a fixed `1e-12` tolerance offset in the denominator. This prevents singularities and guarantees numerical stability upon coincident particle distances.

## Custom Weight Functions (API Contract)

To introduce a new weighting scheme, you must define a struct that subtypes the abstract `MLSWeightFunction` type and implement two specific methods to satisfy the internal API contract.

### 1. The Functor (Distance Evaluation)
Your struct must be callable and accept a **squared distance** (`dist_sq::Real`) as its only argument, returning the computed scalar weight.

```julia
@inline function (w::YourCustomWeightFunction{T})(dist_sq::Real) where {T}
    # Custom decay logic here using dist_sq
    return computed_weight
end
```

### 2. The Cutoff Radius Accessor
The spatial hashing and neighbor search algorithms require strict knowledge of the maximum interaction radius. You must overload the `get_cutoff` method to return your weight function's defined spatial range.

```julia
@inline get_cutoff(wf::YourCustomWeightFunction) = wf.range
```

## Documentation

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["Grid/MLSWeightFunctions.jl"]
Private = false
```