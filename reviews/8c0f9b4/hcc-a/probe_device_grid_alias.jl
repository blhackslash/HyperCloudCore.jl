# device_grid(Array, host_grid): "writes through either grid are seen by both until one of them grows a buffer"?
using Test, HyperCloudCore, StaticArrays
const HC = HyperCloudCore
geom = get_rectangular_domain(Float64, (0.0,), (1.0,); is_periodic = true)
pg = ParticleGrid(geom, (0.1,), ExponentialWeightFunction(1.0, 0.25), 1)
dg = HC.device_grid(Array, pg)
HC.ensure_capacity!(dg, 4 * length(dg.rhos))                     # dg grows every buffer
dg.rhos[1] = State{1, Float64}(42.0)
println("after growth of dg: same Vector object = ", parent(dg.rhos) === parent(pg.rhos), ", pg sees the write = ", pg.rhos[1][1] == 42.0,
        ", length(pg.rhos) = ", length(pg.rhos))
