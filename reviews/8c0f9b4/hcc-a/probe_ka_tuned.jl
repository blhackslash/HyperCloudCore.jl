# KABackend work-group fallback: hot-path cost with a non-empty cache, concurrent first launches, error matching.
using Test, HyperCloudCore, StaticArrays
const HC = HyperCloudCore
small!(i, out) = (out[i] = 2i; nothing)
wide!(i, out, k) = (v = SVector{4096, Float64}(ntuple(j -> Float64(i + j), Val(4096))); out[i] = v[mod1(i * k, 4096)]; nothing)
wide2!(i, out, k) = (v = SVector{4096, Float64}(ntuple(j -> Float64(i - j), Val(4096))); out[i] = v[mod1(i * k, 4096)]; nothing)
ka = KABackend(HC.KA.CPU())
out = zeros(Int, 1000)
foreach_particle(small!, ka, 1000, (out,))                      # compile
t_empty = minimum(@elapsed(foreach_particle(small!, ka, 1000, (out,))) for _ in 1:200)
a_empty = @allocated foreach_particle(small!, ka, 1000, (out,))
outf = zeros(1000)
foreach_particle(wide!, ka, 1000, (outf, 7))                   # fills the cache
@test !isempty(HC._TUNED_WORKGROUP)
t_full = minimum(@elapsed(foreach_particle(small!, ka, 1000, (out,))) for _ in 1:200)
a_full = @allocated foreach_particle(small!, ka, 1000, (out,))
println("small kernel launch: min ", round(1e6t_empty; digits = 1), " µs / ", a_empty, " B with an empty cache, ",
        round(1e6t_full; digits = 1), " µs / ", a_full, " B with a non-empty cache")
t_look = minimum(@elapsed(HC._needs_tuned_workgroup((ka.backend, 256, typeof(small!), Tuple{Vector{Int}}))) for _ in 1:1000)
println("cache lookup alone: ", round(1e9t_look; digits = 0), " ns")

# concurrent first launches of a new wide kernel from several tasks
outs = [zeros(1000) for _ in 1:4]
foreach_particle(wide2!, ka, 10, (zeros(10), 3))   # compile once outside (PoCL compile is not the subject)
empty!(HC._TUNED_WORKGROUP)
ts = [Threads.@spawn foreach_particle(wide2!, ka, 1000, (outs[k], 3)) for k in 1:4]
foreach(wait, ts)
@test all(o -> o == [Float64(i - mod1(3i, 4096)) for i in 1:1000], outs)
@test count(k -> k[3] === typeof(wide2!), HC._TUNED_WORKGROUP) == 1
# a body that throws an unrelated ArgumentError is not swallowed
bad!(i, out) = (i == 1 && throw(ArgumentError("Work-group size inside the body")); nothing)
