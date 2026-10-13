# --- Particle adaptivity builder (Adapt namespace) ---
# :Adapt_enabled => true, optional :Adapt_max_gap, :Adapt_min_gap, :Adapt_clearance,
# :Adapt_samples_per_dx, :Adapt_exclude_reduced_order
function build_adaptivity(conf::Dict, context::Dict)
    get(conf, :enabled, false) || return nothing
    T = context[:Type]::DataType
    return ParticleAdaptivity(T;
        max_gap               = get(conf, :max_gap, 1.6),
        min_gap               = get(conf, :min_gap, 0.5),
        clearance             = get(conf, :clearance, 0.7),
        samples_per_dx        = get(conf, :samples_per_dx, 2),
        exclude_reduced_order = get(conf, :exclude_reduced_order, true))
end
