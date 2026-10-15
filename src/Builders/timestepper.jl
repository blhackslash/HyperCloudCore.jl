"""
    build_timestepper(context; time_conf = Dict()) -> GeneralRKTimeStepper

Requires `context[:Equation]`, `[:Scheme]`, `[:MOOD]`, `[:Tableau]`, `[:ExplicitSources]`,
`[:ImplicitSources]`. Optional `Time` keys: `reset_orders_per_stage`, `full_search_per_stage`;
the volume estimator comes from `context[:VolumeEstimator]` if set (otherwise the stepper default),
the stage buffers live on `storage_type(context)`.
"""
function build_timestepper(context::Dict; time_conf::Dict = Dict{Symbol, Any}())
    kwargs = (; reset_orders_per_stage = get(time_conf, :reset_orders_per_stage, false)::Bool,
                full_search_per_stage  = get(time_conf, :full_search_per_stage, false)::Bool,
                storage                = storage_type(context))
    if haskey(context, :VolumeEstimator)
        kwargs = (; kwargs..., volume_estimator = context[:VolumeEstimator])
    end
    return GeneralRKTimeStepper(
        context[:Equation], context[:Scheme], context[:MOOD],
        context[:ExplicitSources], context[:ImplicitSources], context[:Tableau]::IMEXButcherTableau;
        kwargs...)
end
