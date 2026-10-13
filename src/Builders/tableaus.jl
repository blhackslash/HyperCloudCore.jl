"""
    build_tableau(time_conf, context) -> IMEXButcherTableau

Selector `:stepper`: `:Euler`, `:RK2` (SSP), `:RK2Ralston`, `:RK3` (SSP), `:RK4`, `:ARS222`,
`:ARS233`, `:PRSSP3`, `:SSP332`, `:IMEXEuler`. Explicit tableaus are returned as
`IMEXButcherTableau`s with a zero implicit part.
"""
function build_tableau(time_conf::Dict, context::Dict)
    name = time_conf[:stepper]::Symbol
    return build_tableau(Val(name), time_conf, context)
end

build_tableau(name::Val, conf::Dict, ctx::Dict) = error("Unknown Time Stepper Tableau: $(typeof(name))")

# Explicit RK Tableaus
build_tableau(::Val{:Euler}, conf::Dict, ctx::Dict) = RK1_Euler_Tableau(real_type(ctx))
build_tableau(::Val{:RK2}, conf::Dict, ctx::Dict)   = RK2_SSP_Tableau(real_type(ctx))
build_tableau(::Val{:RK2Ralston}, conf::Dict, ctx::Dict) = RK2_Ralston_Tableau(real_type(ctx))
build_tableau(::Val{:RK3}, conf::Dict, ctx::Dict)   = RK3_SSP_Tableau(real_type(ctx))
build_tableau(::Val{:RK4}, conf::Dict, ctx::Dict)   = RK4_Classical_Tableau(real_type(ctx))

# IMEX Tableaus
build_tableau(::Val{:ARS233}, conf::Dict, ctx::Dict) = IMEX_ARS233_Tableau(real_type(ctx))
build_tableau(::Val{:PRSSP3}, conf::Dict, ctx::Dict) = IMEX_PRSSP3_Tableau(real_type(ctx))
build_tableau(::Val{:ARS222}, conf::Dict, ctx::Dict) = IMEX_ARS222_Tableau(real_type(ctx))
build_tableau(::Val{:SSP332}, conf::Dict, ctx::Dict) = IMEX_SSP2332_Tableau(real_type(ctx))
build_tableau(::Val{:IMEXEuler}, conf::Dict, ctx::Dict) = IMEX_Euler_Tableau(real_type(ctx))

