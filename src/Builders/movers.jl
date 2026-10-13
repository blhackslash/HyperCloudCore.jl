# --- Grid Mover Builder ---
# Grid namespace keys:
#   :mover           => :none | :lagrangian | :scaled_lagrangian | :sinusoidal   (default :none)
#   :mover_alpha     => α for :scaled_lagrangian
#   :mover_amplitude => A for :sinusoidal, w_d(x) = A * sin(2π (x_d - min_d) / L_d)  (periodic-safe)
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

function build_mover(::Val{:sinusoidal}, conf::Dict, context::Dict)
    T    = real_type(context)
    D    = context[:D]::Int
    geom = context[:Domain]
    A    = T(conf[:mover_amplitude])
    mins = geom.mins
    Ls   = geom.maxs .- geom.mins
    two_pi = T(2π)
    f = (x, t) -> Space{D, T}(ntuple(d -> A * sin(two_pi * (x[d] - mins[d]) / Ls[d]), Val(D)))
    return PrescribedMover(f)
end
