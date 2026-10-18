# --- Volume estimator ---
"""
    build_volume_estimator(grid_conf, context) -> VolumeEstimator
    build_volume_estimator(::Val{name}, grid_conf, context)

`Grid` namespace. Selector `:volumes` (default `:shepard`): `:shepard` ([`ShepardVolumes`](@ref), optional
`:volumes_sweeps`, default 1) or `:mls` ([`MLSVolumes`](@ref), optional `:volumes_degree`, default 2, and
`:volumes_samples_per_dx`, default 2). Called by [`build_particle_grid`](@ref).
"""
function build_volume_estimator end

function build_volume_estimator(grid_conf::Dict, context::Dict)
    name = get(grid_conf, :volumes, :shepard)::Symbol
    return build_volume_estimator(Val(name), grid_conf, context)
end

build_volume_estimator(name::Val, conf::Dict, context::Dict) = error("Unknown volume estimator: $(typeof(name))")
build_volume_estimator(::Val{:shepard}, conf::Dict, context::Dict) = ShepardVolumes(sweeps = get(conf, :volumes_sweeps, 1))
build_volume_estimator(::Val{:mls}, conf::Dict, context::Dict) =
    MLSVolumes(get(conf, :volumes_degree, 2); samples_per_dx = get(conf, :volumes_samples_per_dx, 2))
