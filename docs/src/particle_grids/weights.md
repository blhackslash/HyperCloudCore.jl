# MLS Weight Functions

A weight function sets how strongly a neighbour influences the MLS fit at a particle, and its cutoff radius defines the neighbourhood. Weights are evaluated once per neighbour pair during the neighbour search (and again by `refresh_geometry!` on moving grids) and stored in `pg.neighbor.weights`.

## Built-in weights

**`ExponentialWeightFunction(alpha, range)`** approximates

```math
w(r) \approx \exp\!\left(-\alpha\, \frac{r^2}{R^2}\right), \qquad R = \texttt{range}.
```

The inverse range squared is precomputed. The exponential is evaluated as the reciprocal of a fourth-order Taylor polynomial of $e^{y}$, $y = \alpha r^2/R^2 \ge 0$. This is positive, monotonically decreasing and cheap, but it decays more slowly than the true exponential for large $\alpha r^2/R^2$.

**`InverseWeightFunction(alpha, range)`** evaluates $w(r) = 1/(r^2 + 10^{-12})$. The offset keeps coincident particles finite. `alpha` is stored but not used by the current implementation.

For both, `range` is the absolute cutoff radius $R$. A common choice is $R \approx 2.5$–$3.5\,\Delta x$. Higher-order schemes need more neighbours: a degree-$p$ fit in $D$ dimensions needs at least $\binom{p+D}{D} - 1$ neighbours, and schemes drop to a lower degree when a neighbourhood is too small.

## Writing your own

Subtype `MLSWeightFunction` and provide two methods:

```julia
struct MyWeight{T} <: MLSWeightFunction
    range::T
end

# Called with the SQUARED distance
@inline (w::MyWeight{T})(dist_sq::Real) where {T} = ...

# Cutoff radius used for the neighbour search and bins
@inline HyperCloudCore.get_cutoff(w::MyWeight) = w.range
```

The weight must be positive within the cutoff. It does not need to vanish at the cutoff, since pairs beyond `R` are never listed. On moving grids, `refresh_geometry!` sets the weight of pairs that drift beyond `R` to exactly zero.

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["MLSWeightFunctions.jl"]
Private = false
```
