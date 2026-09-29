```@meta
CurrentModule = HyperCloudCore
```

# Base API
The base API contains all the methods you need to define your hyperbolic system. If you don't plan on developing custom solvers, this is everything you need to set to start solving your problem.

```@docs
flux
velocity
cons2prim
prim2cons
update_size!
update_content!
max_eigenvalue
evaluate_source
implicit_solve
pre_solve_update!
```
