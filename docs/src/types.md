```@meta
CurrentModule = HyperCloudCore
```
# Static Arrays Aliases
The package heavily uses aliases for frequently used physical quantities which are also exported for use in custom logic:
```@docs
Space
State
Flux
Velocity
```

# Abstract Types
If you want to create your own methods within this package, you can find the list of all abstract types you can create custom structs for here:

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["Types.jl"]
```