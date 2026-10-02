struct Box{D, M} <: InitialCondition
    u_bg::State{M}
    u_box::State{M}
    mins::Space{D}
    maxs::Space{D}
end

(ic::Box)(pos::Space{D}) where {D} = all(ic.mins .<= pos .<= ic.maxs) ? ic.u_box : ic.u_bg

function build_ic(::Val{:box}, conf::Dict, ctx::Dict)
    T, D, M = ctx[:Type]::DataType, ctx[:D]::Int, ctx[:M]::Int
    
    u_bg = State{M, T}(Tuple(T.(conf[:u_bg])))
    u_box = State{M, T}(Tuple(T.(conf[:u_box])))
    mins = Space{D, T}(Tuple(T.(conf[:mins])))
    maxs = Space{D, T}(Tuple(T.(conf[:maxs])))
    
    return Box(u_bg, u_box, mins, maxs)
end