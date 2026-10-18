# --- MOOD Criteria & Strategies ---
"""
    build_mood_criterion(::Val{name}, mood_conf, context) -> MOODCriterion

The criterion of [`build_mood`](@ref) for `:criterion = name`: `:none` ([`NoMOOD`](@ref)), `:only`
([`OnlyMOOD`](@ref)), `:U1` ([`MOODu1`](@ref)) or `:U2` ([`MOODu2`](@ref)). `:U1` and `:U2` require
`:delta_relax`, which is multiplied by the cell volume `prod(context[:Grid].meta.dx)`, so they need the grid.
"""
function build_mood_criterion end

build_mood_criterion(name::Val, conf::Dict, context::Dict) = error("Unknown MOOD criterion: $(typeof(name))")
build_mood_criterion(::Val{:none}, conf::Dict, context::Dict) = NoMOOD()
build_mood_criterion(::Val{:only}, conf::Dict, context::Dict) = OnlyMOOD()

function build_mood_criterion(::Val{:U1}, conf::Dict, context::Dict)
    T = real_type(context)
    pg = context[:Grid]
    
    vol_dx = prod(pg.meta.dx)
    
    # Strictly require the delta relax coefficient if U1/U2 is requested
    delta_relax = vol_dx * T(conf[:delta_relax]) 
    
    return MOODu1(delta_relax)
end

function build_mood_criterion(::Val{:U2}, conf::Dict, context::Dict)
    T = real_type(context)
    pg = context[:Grid]
    
    vol_dx = prod(pg.meta.dx)
    delta_relax = vol_dx * T(conf[:delta_relax]) 
    
    return MOODu2(delta_relax)
end

"""
    build_mood_strategy(::Val{name}, mood_conf, context) -> MOODStrategy

The strategy of [`build_mood`](@ref) for `:strategy = name`: `:EPD0` ([`EPD0`](@ref)), `:SEPD0`
([`StrictEPD0`](@ref)), `:EPD1` ([`EPD1`](@ref)) or `:EPD2` ([`EPD2`](@ref)).
"""
function build_mood_strategy end

build_mood_strategy(name::Val, conf::Dict, context::Dict) = error("Unknown MOOD strategy: $(typeof(name))")
build_mood_strategy(::Val{:EPD0}, conf::Dict, context::Dict)   = EPD0()
build_mood_strategy(::Val{:SEPD0}, conf::Dict, context::Dict)  = StrictEPD0()
build_mood_strategy(::Val{:EPD1}, conf::Dict, context::Dict)   = EPD1()
build_mood_strategy(::Val{:EPD2}, conf::Dict, context::Dict)   = EPD2()

# --- MOOD Builder ---
"""
    build_mood(mood_conf, context) -> MOOD

`MOOD` namespace. `:criterion` and `:strategy` (both default `:none`) select
[`build_mood_criterion`](@ref) and [`build_mood_strategy`](@ref); with both `:none` MOOD is disabled
(`MOOD()`), otherwise both have to be set. See
[Multi-Dimensional Optimal Order Detection (MOOD)](@ref).
"""
function build_mood end

function build_mood(mood_conf::Dict, context::Dict)
    crit_sym = get(mood_conf, :criterion, :none)::Symbol
    strat_sym = get(mood_conf, :strategy, :none)::Symbol

    if crit_sym === :none && strat_sym === :none
        return MOOD() 
    end
    
    criterion = build_mood_criterion(Val(crit_sym), mood_conf, context)
    strategy  = build_mood_strategy(Val(strat_sym), mood_conf, context)
    
    return MOOD(strategy, criterion)
end