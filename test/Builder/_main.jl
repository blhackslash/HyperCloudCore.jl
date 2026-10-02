function build_particle_grid(grid_conf::Dict, context::Dict)
    # 1. Pull required dependencies from the context
    T = context[:Type]::DataType
    D = context[:D]::Int
    M = context[:M]::Int
    geom = context[:Domain]
    
    geom_mins = Tuple(geom.mins)
    geom_maxs = Tuple(geom.maxs)
    
    # 2. Dynamically extract bounds from the resolved geometry
    Ns = grid_conf[:Ns]::Tuple
    rf_tuple = grid_conf[:randomness_factor]::Tuple
    
    dxs = ntuple(d -> (T(geom_maxs[d]) - T(geom_mins[d])) / Int(Ns[d]), Val(D))
    max_dx = maximum(dxs)
    nominal_dx = dxs
    randomness = ntuple(d -> T(rf_tuple[d]) * dxs[d], Val(D))
    
    # 3. Save max_dx into the context so the weight builder (and others) can access it
    context[:max_dx] = max_dx
    
    # 4. Instantiate the MLS Weight Function using the unified API
    weight_conf = context[:WeightConf]::Dict
    weight_func = build_weights(weight_conf, context)
    
    # Strict SEED extraction
    rng = MersenneTwister(grid_conf[:SEED]::Int)

    # 5. Construct the Particle Grid
    pg = ParticleGrid(
        geom, nominal_dx, weight_func, M;
        randomness = randomness,
        rng = rng,
    )
    
    return pg
end

include("limiter.jl")
include("mood.jl")
include("schemes.jl")
include("weights.jl")
include("kinetic.jl")
include("domains.jl")
include("timestepper.jl")