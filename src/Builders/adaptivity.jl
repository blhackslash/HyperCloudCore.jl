# --- Particle adaptivity builder ---
"""
    build_adaptivity(adapt_conf, context) -> Union{ParticleAdaptivity, Nothing}

`Adapt` namespace. `nothing` unless `:enabled` is `true`; then a [`ParticleAdaptivity`](@ref) with the
optional keys `:max_gap` (1.6), `:min_gap` (0.5), `:clearance` (0.7), `:samples_per_dx` (2) and
`:exclude_reduced_order` (`true`), see [Particle insertion and merging](@ref).
"""
function build_adaptivity end

function build_adaptivity(conf::Dict, context::Dict)
    get(conf, :enabled, false) || return nothing
    T = real_type(context)
    return ParticleAdaptivity(T;
        max_gap               = get(conf, :max_gap, 1.6),
        min_gap               = get(conf, :min_gap, 0.5),
        clearance             = get(conf, :clearance, 0.7),
        samples_per_dx        = get(conf, :samples_per_dx, 2),
        exclude_reduced_order = get(conf, :exclude_reduced_order, true))
end
