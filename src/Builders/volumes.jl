# --- Volume estimator (Grid namespace: :volumes => :shepard | :mls, :volumes_degree, :volumes_samples_per_dx) ---
function build_volume_estimator(grid_conf::Dict, context::Dict)
    name = get(grid_conf, :volumes, :shepard)::Symbol
    return build_volume_estimator(Val(name), grid_conf, context)
end

build_volume_estimator(name::Val, conf::Dict, context::Dict) = error("Unknown volume estimator: $(typeof(name))")
build_volume_estimator(::Val{:shepard}, conf::Dict, context::Dict) = ShepardVolumes(sweeps = get(conf, :volumes_sweeps, 1))
build_volume_estimator(::Val{:mls}, conf::Dict, context::Dict) =
    MLSVolumes(get(conf, :volumes_degree, 2); samples_per_dx = get(conf, :volumes_samples_per_dx, 2))
