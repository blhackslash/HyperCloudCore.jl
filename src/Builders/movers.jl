# --- Grid Mover Builder ---
"""
    build_mover(grid_conf, context) -> GridMover
    build_mover(::Val{name}, grid_conf, context)

`Grid` namespace. Selector `:mover` (default `:none`):

| `:mover` | Mover | Keys |
|:--|:--|:--|
| `:none` | [`NoGridMover`](@ref) | |
| `:lagrangian` | [`LagrangianMover`](@ref) | |
| `:scaled_lagrangian` | [`ScaledLagrangianMover`](@ref) | `:mover_alpha` |
| `:sinusoidal` | [`PrescribedMover`](@ref) with [`SinusoidalVelocity`](@ref) | `:mover_amplitude` (needs `context[:Domain]`) |

The `:sinusoidal` velocity, ``w_d(x) = A \\sin(2\\pi (x_d - \\mathrm{min}_d) / L_d)``, is periodic on the domain.
Called by [`build_particle_grid`](@ref).
"""
function build_mover end

function build_mover(grid_conf::Dict, context::Dict)
    name = get(grid_conf, :mover, :none)::Symbol
    return build_mover(Val(name), grid_conf, context)
end

build_mover(name::Val, conf::Dict, context::Dict) = error("Unknown grid mover: $(typeof(name))")
build_mover(::Val{:none},       conf::Dict, context::Dict) = NoGridMover()
build_mover(::Val{:lagrangian}, conf::Dict, context::Dict) = LagrangianMover()

function build_mover(::Val{:scaled_lagrangian}, conf::Dict, context::Dict)
    T = real_type(context)
    return ScaledLagrangianMover(T(conf[:mover_alpha]))
end

"""
    SinusoidalVelocity{D, T}(A, mins, Ls)

`w_d(x) = A sin(2π (x_d - mins_d) / Ls_d)`: the velocity field of the `:sinusoidal` mover. A bits-type callable
(a closure capturing `T` or `D` as values would not be isbits and could not enter a GPU kernel).
"""
struct SinusoidalVelocity{D, T}
    A::T
    mins::Space{D, T}
    Ls::Space{D, T}
end
@inline function (v::SinusoidalVelocity{D, T})(x, t) where {D, T}
    two_pi = T(2π)
    return Space{D, T}(ntuple(d -> v.A * sin(two_pi * (x[d] - v.mins[d]) / v.Ls[d]), Val(D)))
end

function build_mover(::Val{:sinusoidal}, conf::Dict, context::Dict)
    T    = real_type(context)
    D    = context[:D]::Int
    geom = context[:Domain]
    A    = T(conf[:mover_amplitude])
    mins = Space{D, T}(Tuple(geom.mins))
    Ls   = Space{D, T}(Tuple(geom.maxs .- geom.mins))
    return PrescribedMover(SinusoidalVelocity{D, T}(A, mins, Ls))
end
