```@meta
CurrentModule = HyperCloudCore
```

# Timestepper
Currently there are two types of time stepping methods implemented: explicit Runge-Kutta (RK) and implicit-explicit (IMEX) methods. They both need their respective Butcher tableau to specify the concrete scheme. The general structs are defined here:

```@docs
RKButcherTableau
GeneralRKTimeStepper
IMEXButcherTableau
GeneralIMEXTimeStepper
```
# Inbuilt Butcher Tableaus
To choose the specific timestepper, you have to either implement your own Butcher tableau or use one of these inbuilt one:

```@docs
RK1_Euler_Tableau
RK2_Ralston_Tableau
RK3_SSP_Tableau
RK4_Classical_Tableau
IMEX_Euler_Tableau
IMEX_ARS233_Tableau
IMEX_ARS222_Tableau
IMEX_PRSSP3_Tableau
IMEX_SSP2332_Tableau
```
