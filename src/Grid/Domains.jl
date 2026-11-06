export GeometricDomain, get_rectangular_domain, get_spherical_domain

# =========================================================================
# 1. THE PURE GEOMETRY LAYER
# =========================================================================
"""
    GeometricDomain{GEO, D, T, F_Interior, F_Tag}

A pure mathematical abstraction representing the physical shape and boundary regions of the problem space. 

# Fields
- `label::GEO`: The geometric identifier (e.g., `Val(:rectangular)` or `Val(:spherical)`).
- `mins::Space{D, T}`: The minimal bounding box coordinates.
- `maxs::Space{D, T}`: The maximal bounding box coordinates.
- `is_periodic::SVector{D, Bool}`: Periodic boundary flags for each spatial dimension.
- `is_interior::F_Interior`: A function evaluating whether a given coordinate resides strictly within the geometry.
- `get_tag::F_Tag`: A function generating integer boundary tags based on spatial location.
- `bc_map::Dict{Int, AbstractBoundaryCondition}`: A mapping of integer tags to specific boundary condition evaluators.
"""
struct GeometricDomain{GEO, D, T, F_Interior, F_Tag}
    label::GEO
    mins::Space{D, T}
    maxs::Space{D, T}
    is_periodic::SVector{D, Bool}
    is_interior::F_Interior
    get_tag::F_Tag
    bc_map::Dict{Int, AbstractBoundaryCondition}
end

"""
    get_rectangular_domain(::Type{T}, mins, maxs; bc_map)

Generates a rectangular `GeometricDomain` instance. 
- Constructs an interior evaluation function enforcing strict Cartesian bounding box constraints.
- Automatically handles edge tagging in 2D by assigning 1 to Left, 2 to Right, 3 to Bottom, and 4 to Top based on nearest-distance evaluations
  (the edges of a periodic axis are ignored, so a channel periodic in x has only the tags 3 and 4).
- In 1-D and 3-D every boundary point has tag 1 (both walls of a 1-D interval, all faces of a box). The
  `ParticleGrid` constructor warns when `bc_map` names a tag no boundary particle carries, or when boundary
  particles carry a tag `bc_map` does not cover (they then keep their state, like a Dirichlet condition).
"""
function get_rectangular_domain(
    ::Type{T}, 
    mins::NTuple{D, Real}, 
    maxs::NTuple{D, Real};
    bc_map::Dict{Int, AbstractBoundaryCondition} = Dict{Int, AbstractBoundaryCondition}(),
    is_periodic::Union{Bool, NTuple{D, Bool}} = false
) where {D, T}
    
    mins_f = Space{D, T}(mins...)
    maxs_f = Space{D, T}(maxs...)

    is_per_svec = isa(is_periodic, Bool) ? SVector{D, Bool}(ntuple(_ -> is_periodic, Val(D))) : SVector{D, Bool}(is_periodic)
    
    is_interior_func = (pos) -> begin
        for d in 1:D
            if pos[d] < mins_f[d] || pos[d] > maxs_f[d]
                return false
            end
        end
        return true
    end

    tag_func = (pos) -> begin
        if is_interior_func(pos); return 0; end
        
        if D == 2
            # The edges of a periodic axis are no boundary: ignore them (a channel periodic in x used to tag the
            # ghosts near the seam 1/2, which no bc_map entry covered)
            d_left   = is_per_svec[1] ? T(Inf) : abs(pos[1] - mins_f[1])
            d_right  = is_per_svec[1] ? T(Inf) : abs(pos[1] - maxs_f[1])
            d_bottom = is_per_svec[2] ? T(Inf) : abs(pos[2] - mins_f[2])
            d_top    = is_per_svec[2] ? T(Inf) : abs(pos[2] - maxs_f[2])
            
            min_d = min(d_left, d_right, d_bottom, d_top)
            
            if min_d == d_left; return 1
            elseif min_d == d_right; return 2
            elseif min_d == d_bottom; return 3
            else; return 4
            end
        else
            return 1
        end
    end

    return GeometricDomain(Val(:rectangular),mins_f, maxs_f, is_per_svec, is_interior_func, tag_func, bc_map)
end

"""
    get_spherical_domain(::Type{T}, center, radius; bc_map)

Generates a spherical (or circular) `GeometricDomain` instance.
- Employs a squared-distance interior evaluation against the defined center point and radius.
- Assigns a uniform boundary tag of 1 to all exterior points.
"""
function get_spherical_domain(
    ::Type{T}, 
    center::NTuple{D, Real}, 
    radius::Real;
    bc_map::Dict{Int, AbstractBoundaryCondition} = Dict{Int, AbstractBoundaryCondition}(),
    is_periodic::Union{Bool, NTuple{D, Bool}} = false
) where {D, T}

    is_per_svec = isa(is_periodic, Bool) ? SVector{D, Bool}(ntuple(_ -> is_periodic, Val(D))) : SVector{D, Bool}(is_periodic)
    
    c_svec = Space{D, T}(center...)
    r_T = T(radius)
    
    mins_f = c_svec .- r_T
    maxs_f = c_svec .+ r_T
    
    is_interior_func = (pos) -> sum(abs2, pos - c_svec) <= r_T^2
    tag_func = (pos) -> is_interior_func(pos) ? 0 : 1 
    
    return GeometricDomain(Val(:spherical), mins_f, maxs_f, is_per_svec, is_interior_func, tag_func, bc_map)
end

"""
    ComputationalDomain{D, T}
    ComputationalDomain(geom::GeometricDomain, nominal_dx, interp_range_factor)

A numerical wrapper surrounding the `GeometricDomain` that manages computational padding, canvas expansion, and periodic limits.

# Constructors

    ComputationalDomain(geom, nominal_dx, interp_range_factor)

Constructs the domain by extending the baseline mathematical boundaries out by a specified ghost layer width (dictated by `interp_range_factor`), unless periodicity is explicitly activated. It calculates and stores inverse lengths to assist with rapid periodic distance computations downstream.

# Fields
- `canvas_mins::Space{D, T}`: Expanded minimal coordinates including ghost layers.
- `canvas_maxs::Space{D, T}`: Expanded maximal coordinates including ghost layers.
- `is_periodic::SVector{D, Bool}`: Dimensional periodicity flags.
- `L::Space{D, T}`: The physical lengths of the domain.
- `L_inv::Space{D, T}`: Pre-computed inverse lengths.
- `L_wrap::Space{D, T}`: Pre-computed wrap lengths for periodic distance evaluation.
- `invL_wrap::Space{D, T}`: Pre-computed inverse wrap lengths.
"""
struct ComputationalDomain{D, T}
    canvas_mins::Space{D, T}
    canvas_maxs::Space{D, T}
    is_periodic::SVector{D, Bool}
    L::Space{D, T}
    L_inv::Space{D, T}
    L_wrap::Space{D, T}
    invL_wrap::Space{D, T}
end

function ComputationalDomain(
    geom::GeometricDomain{GEO, D, T, FI, FT}, 
    nominal_dx::NTuple{D, Real}, 
    interp_range_factor::Real
) where {D, T, FI, FT, GEO}
    
    is_per_svec = geom.is_periodic
    dxs_f = Space{D, T}(nominal_dx...)
    # Ghost layers are decided per axis: a periodic axis needs none, every other axis needs a band
    # of `interp_range_factor` spacings (mixed periodicity used to drop all ghost layers).
    # `interp_range_factor` is R / max(dx): an axis with a smaller spacing needs more layers to be R thick
    # (it used to get the same number, i.e. a band thinner than R on the finer axis of an anisotropic grid).
    # The product is formed in `T` and pulled down by a few ulps of `T`, so an exact ratio (e.g. 2.5 · 2 = 5)
    # does not get an extra layer from round-off.
    max_dx  = maximum(dxs_f)
    range_f = T(interp_range_factor)
    N_ghost = ntuple(Val(D)) do d
        dxs_f[d] == max_dx ? ceil(Int, interp_range_factor) :
                             ceil(Int, range_f * (max_dx / dxs_f[d]) * (1 - 4 * eps(T)))
    end

    canvas_mins = Space{D, T}(ntuple(d -> is_per_svec[d] ? geom.mins[d] : geom.mins[d] - N_ghost[d] * dxs_f[d], Val(D)))
    canvas_maxs = Space{D, T}(ntuple(d -> is_per_svec[d] ? geom.maxs[d] : geom.maxs[d] + N_ghost[d] * dxs_f[d], Val(D)))
    
    L_physical = geom.maxs - geom.mins
    L_wrap = Space{D, T}(ntuple(d -> is_per_svec[d] ? L_physical[d] : zero(T), Val(D)))
    invL_wrap = Space{D, T}(ntuple(d -> is_per_svec[d] ? one(T) / L_physical[d] : zero(T), Val(D)))
    L_inv = one(T) ./ max.(L_physical, T(1e-12))

    return ComputationalDomain{D, T}(
        canvas_mins, canvas_maxs, is_per_svec, L_physical, L_inv, L_wrap, invL_wrap
    )
end

# =========================================================================
# 3. UNIVERSAL NARROW-BAND POINT GENERATOR
# =========================================================================
# Lattice index `j` (point `canvas_min + j dx`) of the wall at `w`, or -1 if the wall falls between two lattice
# points (the extent is not a multiple of `dx`). The ratio carries the round-off of `w`, `canvas_min` and `dx`
# (a few ulps of the ratio); a quarter spacing caps the tolerance, so a wall is never assigned to a point that is
# not its own.
function _wall_index(w::T, canvas_min::T, dx::T) where {T}
    r = (w - canvas_min) / dx
    k = round(Int, r)
    return abs(r - k) <= min(T(0.25), T(0.01) + 8 * eps(T) * abs(r)) ? k : -1
end

"""
    get_points(cd::ComputationalDomain, geom::GeometricDomain; nominal_dx, interp_range_factor, randomness, rng)

Generates a universal narrow-band particle grid utilizing the spatial canvas defined by the computational domain.

# Details
- Generates a baseline Cartesian point cloud separated by `nominal_dx`, applying optional random positional offsets.
- Sorts generated points into interior candidates and ghost candidates using the underlying geometry's `is_interior` functor.
- Filters exterior ghosts aggressively using a cutoff distance dictated by the `interp_range_factor`, yielding an optimized narrow-band boundary layer rather than filling the entire canvas.
- Outputs the filtered particle positions alongside their respective boundary booleans, tags, volumes, and effective spacing.
"""
function get_points(
    cd::ComputationalDomain{D, T},
    geom::GeometricDomain{GEO, D, T, FI, FT};
    nominal_dx::NTuple{D, Real}, 
    interp_range_factor::Real,
    randomness::Tuple = ntuple(i -> zero(T), D),
    rng = Random.default_rng(),
    kwargs... 
) where {D, T, FI, FT, GEO}
    
    dxs_f = Space{D, T}(nominal_dx...)
    rand_f = Space{D, T}(randomness...)
    
    # Points per axis. On a periodic axis the end point coincides with the start point after
    # wrapping, so it is not generated (otherwise the seam particle would exist twice).
    Ns_total = ntuple(Val(D)) do d
        n = max(1, round(Int, (cd.canvas_maxs[d] - cd.canvas_mins[d]) / dxs_f[d]))
        cd.is_periodic[d] ? n : n + 1
    end
    
    inner_points = Space{D, T}[]
    ghost_candidates = Space{D, T}[]
    
    # The lattice points of a non-periodic wall are put exactly on the wall: `canvas_mins + (I - 1) dx` misses
    # `maxs` by a few ulps for many spacings (e.g. [-1, 1] with dx = 2/30), and the closed-interval classification
    # then made that wall point a ghost while the one on `mins` stayed interior. The wall point is chosen by its
    # index (`_wall_index`), so exactly one lattice point per wall moves and no neighbour of it can land on the
    # wall as well (a tolerance on the distance did that at very fine Float32 resolutions).
    wall_lo = ntuple(d -> cd.is_periodic[d] ? -1 : _wall_index(geom.mins[d], cd.canvas_mins[d], dxs_f[d]), Val(D))
    wall_hi = ntuple(d -> cd.is_periodic[d] ? -1 : _wall_index(geom.maxs[d], cd.canvas_mins[d], dxs_f[d]), Val(D))
    for I in CartesianIndices(Ns_total)
        pos_tuple = ntuple(Val(D)) do d
            j = I[d] - 1
            x = j == wall_lo[d] ? geom.mins[d] : j == wall_hi[d] ? geom.maxs[d] : cd.canvas_mins[d] + j * dxs_f[d]
            x += rand_f[d] * (rand(rng, T) * 2 - 1)
            # A jittered point that left a periodic axis belongs to the other end: classifying it
            # before wrapping made it a (frozen) boundary particle at the seam.
            cd.is_periodic[d] ? _wrap_periodic(x, geom.mins[d], cd.L_wrap[d]) : x
        end
        
        pos = Space{D, T}(pos_tuple)
        if geom.is_interior(pos)
            push!(inner_points, pos)
        else
            push!(ghost_candidates, pos)
        end
    end
    
    max_dx = maximum(dxs_f)
    cutoff_dist_sq = (T(interp_range_factor) * max_dx + max_dx)^2
    surviving_ghosts = _narrow_band(inner_points, ghost_candidates, cutoff_dist_sq)

    positions = vcat(inner_points, surviving_ghosts)
    N = length(positions)
    
    is_boundary = zeros(Bool, N)
    tags = zeros(Int, N)
    volumes = fill(prod(dxs_f), N)
    
    for i in 1:N
        pos = positions[i]
        tag = geom.get_tag(pos)
        tags[i] = tag
        is_boundary[i] = (tag != 0)
    end
    
    return positions, is_boundary, tags, volumes, Tuple(dxs_f)
end

# Boundary particles whose tag has no condition keep their state (the stage predictor copies `pg.rhos`), and a
# condition whose tag no particle carries does nothing: both are silent, so warn (host side, at grid set-up)
# when a non-empty `bc_map` and the tags of the boundary particles do not match. An empty map (no conditions,
# every boundary particle frozen) is taken as intended. Each mismatch is reported once per session.
const _WARNED_BC_TAGS = Set{Tuple{Vector{Int}, Vector{Int}}}()
const _WARNED_BC_TAGS_LOCK = ReentrantLock()          # grids may be built from several tasks
function _check_bc_tags(geom::GeometricDomain, tags::AbstractVector{Int}, is_boundary::AbstractVector{Bool})
    isempty(geom.bc_map) && return nothing
    carried = sort!(unique(tags[i] for i in eachindex(tags) if is_boundary[i]))
    mapped  = sort!(collect(keys(geom.bc_map)))
    unused, uncovered = setdiff(mapped, carried), setdiff(carried, mapped)
    (isempty(unused) && isempty(uncovered)) && return nothing
    key = (mapped, carried)
    first_time = lock(_WARNED_BC_TAGS_LOCK) do
        key in _WARNED_BC_TAGS ? false : (push!(_WARNED_BC_TAGS, key); true)
    end
    first_time || return nothing
    @warn "bc_map and boundary tags do not match: " *
          (isempty(unused) ? "" : "no boundary particle carries tag(s) $(unused), so their condition is never applied; ") *
          (isempty(uncovered) ? "" : "boundary particles with tag(s) $(uncovered) have no condition and keep their state; ") *
          "the boundary particles carry tag(s) $(carried) (1-D and 3-D boxes tag every boundary particle 1, see get_rectangular_domain)."
    return nothing
end

@inline _wrap_periodic(x::T, lo, L) where {T} = (y = T(lo) + mod(x - T(lo), T(L)); y >= T(lo) + T(L) ? T(lo) : y)

# The ghost candidates within `sqrt(cutoff_sq)` of an interior point, in their original order. Interior
# points are hashed into cells of that width, so only the 3^D surrounding cells are searched and the
# cost is linear in the number of points (the pairwise check was quadratic and dominated 3-D set-ups).
function _narrow_band(inner::Vector{Space{D, T}}, candidates::Vector{Space{D, T}}, cutoff_sq::T) where {D, T}
    survivors = Space{D, T}[]
    (isempty(inner) || isempty(candidates)) && return survivors
    h = sqrt(cutoff_sq)
    origin = reduce((a, b) -> min.(a, b), inner)
    cell_of(p) = ntuple(d -> floor(Int, (p[d] - origin[d]) / h), Val(D))
    cells = Dict{NTuple{D, Int}, Vector{Int}}()
    for (i, p) in enumerate(inner)
        push!(get!(() -> Int[], cells, cell_of(p)), i)
    end
    window = CartesianIndices(ntuple(_ -> -1:1, Val(D)))
    for g in candidates
        c = cell_of(g)
        found = false
        for off in window
            lst = get(cells, ntuple(d -> c[d] + off[d], Val(D)), nothing)
            lst === nothing && continue
            for i in lst
                if sum(abs2, g - inner[i]) <= cutoff_sq
                    found = true
                    break
                end
            end
            found && break
        end
        found && push!(survivors, g)
    end
    return survivors
end