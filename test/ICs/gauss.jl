struct Gauss{D, M} <: SmoothInitialCondition
    a::State{M}
    b::Space{D}
    width::Float64
end

(ic::Gauss)(pos::Space{D}) where {D} = ic.a * exp(-sum(abs2, pos - ic.b) / ic.width^2)

function build_ic(::Val{:gauss}, conf::Dict, ctx::Dict)
    T, D, M = ctx[:Type]::DataType, ctx[:D]::Int, ctx[:M]::Int
    
    a = State{M, T}(Tuple(T.(conf[:a])))
    b = Space{D, T}(Tuple(T.(conf[:b])))
    width = T(conf[:width])
    
    return Gauss(a, b, width)
end