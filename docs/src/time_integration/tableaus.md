# Butcher Tableaus

Every tableau is an `IMEXButcherTableau{T}` holding an explicit part $(\tilde A, \tilde b, \tilde c)$ = `(a_t, b_t, c_t)` and an implicit part $(A, b, c)$ = `(a, b, c)`. The constructor checks that all components have consistent sizes, that `a_t` is strictly lower triangular, and that `a` is lower triangular. Stage times are taken from `c_t`.

`RKButcherTableau(a, b, c)` wraps a classical explicit tableau into this form with `a = 0`, `b = 0`.

## Available tableaus

| Constructor | Stages | Explicit part SSP | Notes |
|---|---|---|---|
| `RK1_Euler_Tableau(T)` | 1 | yes | Forward Euler, order 1 |
| `RK2_SSP_Tableau(T)` | 2 | yes | Heun / SSP-RK2, order 2 |
| `RK2_Ralston_Tableau(T)` | 2 | no | Order 2, minimal truncation error |
| `RK3_SSP_Tableau(T)` | 3 | yes | Shu–Osher SSP-RK3, order 3 |
| `RK4_Classical_Tableau(T)` | 4 | no | Classical RK4, order 4 |
| `RK4_SSP_Tableau(T)` | 5 | yes | Spiteri–Ruuth SSPRK(5,4), order 4, SSP coefficient 1.508 |
| `IMEX_Euler_Tableau(T)` | 2 | yes | Forward/backward Euler, order 1 |
| `IMEX_ARS222_Tableau(T[, γ])` | 3 | no | Ascher–Ruuth–Spiteri (2,2,2), order 2, first stage explicit, L-stable implicit part |
| `IMEX_ARS233_Tableau(T[, γ])` | 3 | no | Ascher–Ruuth–Spiteri (2,3,3), order 3 |
| `IMEX_SSP2332_Tableau(T)` | 3 | yes | Pareschi–Russo SSP2(3,3,2), order 2, L-stable implicit part |
| `IMEX_PRSSP3_Tableau(T)` | 3 | yes | Pareschi–Russo SSP3(3,3,2), order 2 (SSP-RK3 explicit part, L-stable implicit part) |

The IMEX orders are those of the pair, i.e. including the coupling conditions between the explicit and the implicit part (the test suite checks them, and the observed order through the stepper). The explicit part of `IMEX_PRSSP3_Tableau` alone is third order, but a three-stage pair with these weights is at most second order; Pareschi and Russo's third-order scheme SSP3(4,3,3) needs four stages. Up to version 0.7.9, `IMEX_ARS222_Tableau` dropped the explicit first stage and `IMEX_PRSSP3_Tableau` had an inconsistent implicit row: both were first order on coupled problems.

## Choosing a tableau with MOOD

For explicit SSP tableaus (positive [`ssp_coefficient`](@ref): Euler, SSP-RK2, Ralston with $C = 1/2$, SSP-RK3, SSPRK(5,4)) the stepper checks MOOD admissibility on the forward Euler substeps $Y_{k-1} + (\Delta t/C) K_{k-1}$ of the canonical Shu–Osher form. Every stage is a convex combination of these substeps and $U^n$, so admissible substeps give an admissible step (under the scheme's own CFL condition, scaled by $C$) for every SSP method (see [The stage-wise fixed point](@ref)). For the non-SSP tableaus (RK4, the ARS pairs) the stepper checks the stage candidates; the checks still catch oscillations in practice, but there is no such guarantee.

Recommended choices:

- **Explicit, with discontinuities:** `RK3_SSP_Tableau` (or `RK2_SSP_Tableau` for second-order spatial schemes); `RK4_SSP_Tableau` for fourth- and fifth-order spatial schemes with MOOD, where it keeps the SSP guarantee at order 4 (five stages, but a 1.5 times larger SSP step than RK3).
- **Explicit, smooth problems, convergence studies at order 4+:** `RK4_Classical_Tableau` or `RK4_SSP_Tableau`.
- **Stiff relaxation sources:** `IMEX_SSP2332_Tableau` or `IMEX_PRSSP3_Tableau`.

## Defining your own

```julia
# Explicit
bt = RKButcherTableau(A::Matrix{T}, b::Vector{T}, c::Vector{T})

# IMEX
bt = IMEXButcherTableau(a, a_t, c, c_t, b, b_t)
```

Only diagonally implicit tableaus are supported: the stepper solves one local implicit relation per stage via `implicit_solve`.

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["ButcherTableaus.jl"]
```
