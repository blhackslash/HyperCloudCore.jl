# Grid kernels (neighbour search, cell list, BCs, CFL, Shepard volumes, adaptivity pre-check) on the KA CPU
# backend (PoCL) in 2-D and 3-D, Float64 and Float32, against Polyester; and the slot-major layout on host arrays.
using Test, HyperCloudCore, StaticArrays, LinearAlgebra, Random
const HC = HyperCloudCore
import HyperCloudCore: flux, max_eigenvalue, velocity
struct Adv{D, M, T} <: HyperbolicPDE{D, M, T}; c::SVector{D, T}; end
LinearAdvection(c::NTuple{D}, ::Type{T}) where {D, T} = Adv{D, 1, T}(SVector{D, T}(c))
flux(eq::Adv{D, 1, T}, U::State) where {D, T} = Flux{D, 1, T}(ntuple(d -> eq.c[d] * U, Val(D)))
max_eigenvalue(eq::Adv, U::State, d::Int) = abs(eq.c[d])
velocity(eq::Adv{D, 1, T}, U::State, d::Int) where {D, T} = SMatrix{1, 1, T, 1}(eq.c[d])

function setup(D, T; per = false)
    bc = Dict{Int, AbstractBoundaryCondition}(t => OutflowBC() for t in (D == 2 ? (1, 2, 3, 4) : (1,)))
    geom = get_rectangular_domain(T, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D); is_periodic = per, bc_map = per ? Dict{Int, AbstractBoundaryCondition}() : bc)
    dx = D == 3 ? T(1 / 8) : T(1 / 20)
    pg = ParticleGrid(geom, ntuple(_ -> dx, D), ExponentialWeightFunction(one(T), T(2.5) * dx), 1;
                      randomness = ntuple(_ -> T(0.2) * dx, D), rng = MersenneTwister(3))
    for i in 1:pg.meta.N
        x = pg.core.positions[i]
        pg.rhos[i] = State{1, T}(exp(-sum(abs2, x .- T(0.4)) / T(0.05)))
    end
    return pg
end

function passes(pg, eq)
    pg.neighbor(pg)
    nbsets = [sort(collect(pg.neighbor.indices[k] for k in pg.neighbor.ranges[i])) for i in 1:pg.meta.N]
    dt = get_time_step(pg, eq, CentralDivergence(eltype(pg.meta.dx), length(pg.meta.dx), 1, 2); cached_weights = false)
    buf = copy(pg.rhos)
    for (tag, bc) in pg.geometry.bc_map
        bc(pg, buf, tag, nothing, eq, 0.0)
    end
    update_volumes!(pg, ShepardVolumes())
    cand = HC._has_adapt_candidates(pg, ParticleAdaptivity(eltype(pg.meta.dx); max_gap = 1.1))
    return nbsets, dt, copy(view(buf, 1:pg.meta.N)), copy(view(pg.core.volumes, 1:pg.meta.N)), cand
end

@testset "KA CPU grid kernels D=$D $T per=$per" for D in (2, 3), T in (Float64, Float32), per in (false, true)
    eq = LinearAdvection(ntuple(d -> T(1 + 0.5d), D), T)
    set_backend!(PolyesterBackend())
    ref = passes(setup(D, T; per), eq)
    try
        set_backend!(KABackend(HC.KA.CPU(); workgroupsize = 64))
        dev = passes(setup(D, T; per), eq)
        @test dev[1] == ref[1]
        @test isapprox(dev[2], ref[2]; rtol = 10eps(T))
        @test all(isapprox.(first.(dev[3]), first.(ref[3]); rtol = 10eps(T)))
        @test isapprox(dev[4], ref[4]; rtol = 100eps(T))
        @test dev[5] == ref[5]
    finally
        set_backend!(PolyesterBackend())
    end
end

@testset "slot-major neighbour search on host arrays D=$D" for D in (1, 2, 3)
    pg = setup(D, Float64)
    N  = pg.meta.N
    nd = HC.with_slot_layout(pg.neighbor, N, HC.SlotMajorSlots)
    sm = ParticleGrid(pg.meta, pg.geometry, pg.domain, pg.core, pg.shared, nd, pg.reorder, pg.bins, pg.mover, pg.rhos, pg.curvatures)
    rows(g) = [sort(collect(zip((g.neighbor.indices[k] for k in g.neighbor.ranges[i]), (g.neighbor.weights[k] for k in g.neighbor.ranges[i])))) for i in 1:N]
    r0 = rows(pg)
    @test rows(sm) == r0                               # conversion
    sm.neighbor(sm)                                    # a fresh search in the slot-major layout
    @test eltype(sm.neighbor.ranges) === HC.SlotMajorSlots
    @test rows(sm) == r0
    @test all(i -> step(sm.neighbor.ranges[i]) == N, 1:N)
    @test HC._slots_needed(sm) == N * sm.meta.max_nb
end
