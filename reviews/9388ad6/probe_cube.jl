using HyperCloudCore, Logging
for (n, rf) in ((16, 2.5), (16, 5.5), (32, 2.5))
    g = get_rectangular_domain(Float64, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    h = 1 / n
    pg = with_logger(() -> ParticleGrid(g, (h, h, h), ExponentialWeightFunction(1.0, rf * h), 1), NullLogger())
    gh = [i for i in 1:pg.meta.N if pg.core.is_boundary[i]]
    out(p) = [d for d in 1:3 if p[d] < 0 || p[d] > 1]
    single = [i for i in gh if length(out(pg.core.positions[i])) == 1]
    wrong  = [i for i in single if (pg.core.tags[i] + 1) ÷ 2 != out(pg.core.positions[i])[1]]
    # strictly inside the face's footprint by more than one spacing, i.e. not on an edge column
    wrong_far = [i for i in wrong if (p = pg.core.positions[i]; minimum(d -> min(p[d], 1 - p[d]), filter(d -> d != out(p)[1], 1:3)) > h / 2)]
    e = isempty(wrong_far) ? "" : " e.g. $(Tuple(pg.core.positions[wrong_far[1]])) -> tag $(pg.core.tags[wrong_far[1]])"
    println("cube n=$n R=$(rf)h: ghosts $(length(gh)), outside one face only $(length(single)), tagged with another face $(length(wrong)) (of these off the edge lines $(length(wrong_far)))$e")
end
