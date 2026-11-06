# Ghost band on a non-periodic axis whose spacing is smaller than max_dx: is it at least R thick?
using Test, HyperCloudCore, StaticArrays
@testset "anisotropic ghost band covers the support radius" begin
    for (dx, Ns_note) in (((0.02, 0.01), "[0,2]x[0,1], Ns=(100,100)"), ((0.1, 0.05), "dx=(0.1,0.05)"))
        geom = get_rectangular_domain(Float64, (0.0, 0.0), (2.0, 1.0))
        R  = 2.5 * maximum(dx)
        pg = ParticleGrid(geom, dx, ExponentialWeightFunction(1.0, R), 1)
        cd = pg.domain
        ymin_ghost = minimum(pg.core.positions[i][2] for i in 1:pg.meta.N if pg.core.is_boundary[i])
        xmin_ghost = minimum(pg.core.positions[i][1] for i in 1:pg.meta.N if pg.core.is_boundary[i])
        println(Ns_note, ": R = ", R, ", ghost band below y=0 reaches ", ymin_ghost, ", left of x=0 reaches ", xmin_ghost,
                ", canvas_mins = ", cd.canvas_mins)
        # neighbour counts: wall-adjacent interior particle (x mid, y = 0) vs centre particle
        pos = pg.core.positions
        nbcount(i) = length(pg.neighbor.ranges[i])
        imid = argmin(i -> pg.core.is_boundary[i] ? Inf : sum(abs2, pos[i] - SVector(1.0, 0.5)), 1:pg.meta.N)
        iwal = argmin(i -> pg.core.is_boundary[i] ? Inf : sum(abs2, pos[i] - SVector(1.0, 0.0)), 1:pg.meta.N)
        iwlx = argmin(i -> pg.core.is_boundary[i] ? Inf : sum(abs2, pos[i] - SVector(0.0, 0.5)), 1:pg.meta.N)
        println("  neighbours: centre ", nbcount(imid), ", bottom wall ", nbcount(iwal), ", left wall ", nbcount(iwlx))
        @test ymin_ghost <= -R + 1e-12          # band should reach R below the wall
        @test nbcount(iwal) == nbcount(imid)
    end
end
