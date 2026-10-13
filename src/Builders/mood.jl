# --- MOOD Criteria & Strategies ---
build_mood_criterion(name::Val, conf::Dict, context::Dict) = error("Unknown MOOD criterion: $(typeof(name))")
build_mood_criterion(::Val{:none}, conf::Dict, context::Dict) = NoMOOD()
build_mood_criterion(::Val{:only}, conf::Dict, context::Dict) = OnlyMOOD()

function build_mood_criterion(::Val{:U1}, conf::Dict, context::Dict)
    T = context[:Type]::DataType
    pg = context[:Grid]
    
    vol_dx = prod(pg.meta.dx)
    
    # Strictly require the delta relax coefficient if U1/U2 is requested
    delta_relax = vol_dx * T(conf[:delta_relax]) 
    
    return MOODu1(delta_relax)
end

function build_mood_criterion(::Val{:U2}, conf::Dict, context::Dict)
    T = context[:Type]::DataType
    pg = context[:Grid]
    
    vol_dx = prod(pg.meta.dx)
    delta_relax = vol_dx * T(conf[:delta_relax]) 
    
    return MOODu2(delta_relax)
end

build_mood_strategy(name::Val, conf::Dict, context::Dict) = error("Unknown MOOD strategy: $(typeof(name))")
build_mood_strategy(::Val{:EPD0}, conf::Dict, context::Dict)   = EPD0()
build_mood_strategy(::Val{:SEPD0}, conf::Dict, context::Dict)  = StrictEPD0()
build_mood_strategy(::Val{:EPD1}, conf::Dict, context::Dict)   = EPD1()
build_mood_strategy(::Val{:EPD2}, conf::Dict, context::Dict)   = EPD2()

# --- MOOD Builder ---
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