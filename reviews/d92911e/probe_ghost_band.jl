# b094db1: per-axis ghost layers. On an unjittered anisotropic lattice every interior particle must see the same
# neighbour count as a bulk particle (complete stencils at walls and corners), in 2-D and 3-D, Float64 and Float32,
# with mixed periodicity; and the number of layers per axis should be ceil(R / dx_d) (Float32 round-off guard).
include(joinpath(@__DIR__, "header.jl"))

# reference count: lattice offsets k·dx with |k·dx| ≤ R (computed in Float64, ties excluded by the choice of R)
function ref_count(dx, R)
    D = length(dx)
    K = ntuple(d -> floor(Int, R / dx[d]) + 1, D)
    c = 0
    for I in CartesianIndices(ntuple(d -> -K[d]:K[d], D))
        all(==(0), Tuple(I)) && continue
        sum(abs2, ntuple(d -> I[d] * dx[d], D)) <= R^2 && (c += 1)
    end
    return c
end

cases = [
    ((0.0, 0.0), (2.0, 1.0), (0.02, 0.01), 2.37, (false, false)),
    ((0.0, 0.0), (2.0, 1.0), (0.1, 0.05), 2.37, (false, false)),
    ((0.0, 0.0), (1.0, 1.0), (0.05, 0.0125), 2.37, (false, false)),    # ratio 4
    ((0.0, 0.0), (1.0, 1.0), (0.05, 0.02), 2.37, (false, false)),      # ratio 2.5
    ((0.0, 0.0), (1.0, 1.0), (0.05, 0.02), 2.37, (true, false)),
    ((0.0, 0.0), (1.0, 1.0), (0.02, 0.05), 2.37, (false, true)),
    ((0.0, 0.0, 0.0), (1.0, 0.5, 0.5), (0.1, 0.05, 0.025), 2.37, (false, false, false)),
    ((0.0, 0.0, 0.0), (1.0, 0.5, 0.5), (0.1, 0.05, 0.025), 2.37, (true, false, false)),
]
for T in (Float64, Float32), (mins, maxs, dx, f, per) in cases
    @testset "$T dx=$dx per=$per" begin
        D    = length(dx)
        geom = get_rectangular_domain(T, mins, maxs; is_periodic = per)
        dxT  = T.(dx)
        R    = T(f) * maximum(dxT)
        pg   = ParticleGrid(geom, dxT, ExponentialWeightFunction(one(T), R), 1)
        nref = ref_count(Float64.(dxT), Float64(R))
        counts = [length(pg.neighbor.ranges[i]) for i in 1:pg.meta.N if !pg.core.is_boundary[i]]
        bad = count(!=(nref), counts)
        layers = ntuple(d -> per[d] ? 0 : round(Int, (geom.mins[d] - pg.domain.canvas_mins[d]) / dxT[d]), D)
        want   = ntuple(d -> per[d] ? 0 : ceil(Int, Float64(R) / Float64(dxT[d]) - 1e-6), D)
        println("$T dx=$dx per=$per: interior $(length(counts)), reference count $nref, interior with a different count: $bad; layers $layers (ceil(R/dx_d) = $want)")
        @test bad == 0
        @test all(layers .>= want)
    end
end

# The guard `(1 - 4 eps(Float64))` in Float32: layers for R/dx_d that should be an exact integer
for (dx, f) in (((0.02f0, 0.01f0), 2.5f0), ((0.03f0, 0.01f0), 2.0f0), ((0.1f0, 0.025f0), 2.5f0), ((0.07f0, 0.01f0), 3.0f0))
    geom = get_rectangular_domain(Float32, (0.0, 0.0), (1.0, 1.0))
    R    = f * maximum(dx)
    cd   = HC.ComputationalDomain(geom, dx, R / maximum(dx))
    layers = ntuple(d -> round(Int, (geom.mins[d] - cd.canvas_mins[d]) / dx[d]), 2)
    exact  = ntuple(d -> Float64(f) * Float64(maximum(dx)) / Float64(dx[d]), 2)
    println("Float32 dx=$dx R=$(f)·max dx: layers $layers, R/dx_d = $exact, (R/maxdx)*(maxdx/dx) in Float32 = ",
            ntuple(d -> (R / maximum(dx)) * (maximum(dx) / dx[d]), 2))
end
