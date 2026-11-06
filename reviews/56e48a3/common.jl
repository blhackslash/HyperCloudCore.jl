using HyperCloudCore, StaticArrays, LinearAlgebra, Random
const HCC = HyperCloudCore
struct Adv{D, T} <: HyperbolicPDE{D, 1, T}
    a::SVector{D, T}
end
HCC.flux(eq::Adv{D, T}, U::State{1, T}) where {D, T} = Flux{D, 1, T}(ntuple(d -> State{1, T}(eq.a[d] * U[1]), Val(D)))
HCC.velocity(eq::Adv{D, T}, U::State{1, T}, d::Int) where {D, T} = SMatrix{1, 1, T}(eq.a[d])
HCC.max_eigenvalue(eq::Adv{D, T}, U::State{1, T}, d::Int) where {D, T} = abs(eq.a[d])
# admissible = non-negative (for the PAD)
HCC.is_admissible(eq::Adv, U::SVector) = all(isfinite, U) && U[1] >= 0

function grid1d(N; T = Float64, range = 2.5, jitter = 0.0, seed = 1)
    dx = T(2) / N
    geom = get_rectangular_domain(T, (-1.0,), (1.0,); is_periodic = true)
    ParticleGrid(geom, (dx,), ExponentialWeightFunction(one(T), T(range) * dx), 1;
                 randomness = (T(jitter) * dx,), rng = MersenneTwister(seed))
end
function grid2d(n; T = Float64, range = 2.5, jitter = 0.0, seed = 1)
    dx = T(1) / n
    geom = get_rectangular_domain(T, (0.0, 0.0), (1.0, 1.0); is_periodic = true)
    ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(one(T), T(range) * dx), 1;
                 randomness = (T(jitter) * dx, T(jitter) * dx), rng = MersenneTwister(seed))
end
setic!(pg, f) = (T = eltype(pg.rhos[1]); for i in 1:pg.meta.N; pg.rhos[i] = State{1, T}(f(pg.core.positions[i])); end; pg)
