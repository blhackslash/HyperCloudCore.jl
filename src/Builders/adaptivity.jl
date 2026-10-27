# --- Particle adaptivity builder ---
"""
    build_adaptivity(adapt_conf, context) -> Union{ParticleAdaptivity, Nothing}

`Adapt` namespace. `nothing` unless `:enabled` is `true`; then a [`ParticleAdaptivity`](@ref) with the
optional keys `:max_gap` (1.6), `:min_gap` (0.5), `:clearance` (0.7), `:samples_per_dx` (2),
`:exclude_reduced_order` (`true`) and `:max_events` (unlimited), see [Particle insertion and merging](@ref).
On a moving grid (`context[:Grid]`) the adaptivity re-estimates the volumes with
`context[:VolumeEstimator]` (set by [`build_particle_grid`](@ref); `ShepardVolumes()` if absent, as in
[`build_timestepper`](@ref)), the estimator the time stepper uses, so the remap's conservation survives the
next volume update.
"""
function build_adaptivity end

function build_adaptivity(conf::Dict, context::Dict)
    get(conf, :enabled, false) || return nothing
    T = real_type(context)
    pg = get(context, :Grid, nothing)
    moving = pg isa ParticleGrid && _is_moving(pg)
    return ParticleAdaptivity(T;
        max_gap               = get(conf, :max_gap, 1.6),
        min_gap               = get(conf, :min_gap, 0.5),
        clearance             = get(conf, :clearance, 0.7),
        samples_per_dx        = get(conf, :samples_per_dx, 2),
        exclude_reduced_order = get(conf, :exclude_reduced_order, true),
        max_events            = get(conf, :max_events, typemax(Int)),
        volume_estimator      = moving ? get(context, :VolumeEstimator, ShepardVolumes()) : nothing)   # the stepper's default
end
