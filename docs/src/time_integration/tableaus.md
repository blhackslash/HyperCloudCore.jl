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
| `IMEX_ARS222_Tableau(T[, γ])` | 2 | no | Ascher–Ruuth–Spiteri (2,2,2) |
| `IMEX_ARS233_Tableau(T[, γ])` | 3 | no | Ascher–Ruuth–Spiteri (2,3,3) |
| `IMEX_SSP2332_Tableau(T)` | 3 | yes | Pareschi–Russo SSP2(3,3,2), L-stable implicit part |
| `IMEX_PRSSP3_Tableau(T)` | 3 | yes | Pareschi–Russo type, SSP-RK3 explicit part |

## Choosing a tableau with MOOD

The stepper checks MOOD admissibility after every stage and treats each stage as a forward-Euler substep. For SSP tableaus this is backed by theory: every stage is a convex combination of Euler steps, so admissible substeps give an admissible step (under the scheme's own CFL condition). For the non-SSP tableaus (Ralston, RK4, the ARS pairs) the stepper still runs, and the per-stage checks still catch oscillations in practice, but there is no such guarantee.

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
