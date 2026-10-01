export GeometricDomain, get_rectangular_domain, get_spherical_domain

# =========================================================================
# 1. THE PURE GEOMETRY LAYER
# =========================================================================
"""
    GeometricDomain{GEO, D, T, F_Interior, F_Tag}

A pure mathematical abstraction representing the physical shape of the problem space.

# Fields
- `label::GEO`: The geometric identifier.
- `mins::Space{D, T}`, `maxs::Space{D, T}`: The minimal and maximal bounding box coordinates.
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
- Automatically handles edge tagging in 2D by assigning 1 to Left, 2 to Right, 3 to Bottom, and 4 to Top based on nearest-distance evaluations.
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
            d_left   = abs(pos[1] - mins_f[1])
            d_right  = abs(pos[1] - maxs_f[1])
            d_bottom = abs(pos[2] - mins_f[2])
            d_top    = abs(pos[2] - maxs_f[2])
            
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

# =========================================================================
# 2. THE NUMERICAL WRAPPER (COMPUTATIONAL DOMAIN)
# =========================================================================
struct ComputationalDomain{D, T}
    canvas_mins::Space{D, T}
    canvas_maxs::Space{D, T}
    is_periodic::SVector{D, Bool}
    L::Space{D, T}
    L_inv::Space{D, T}
    L_wrap::Space{D, T}
    invL_wrap::Space{D, T}
end

"""
    ComputationalDomain{D, T}

A numerical wrapper surrounding the `GeometricDomain` that manages computational padding, canvas expansion, and periodic limits.

# Details
- Extends the baseline mathematical boundaries out by a specified ghost layer width unless periodicity is explicitly activated.
- Calculates and stores inverse lengths (`L_inv`, `invL_wrap`) to assist with rapid periodic distance computations downstream.
"""
function ComputationalDomain(
    geom::GeometricDomain{GEO, D, T, FI, FT}, 
    nominal_dx::NTuple{D, Real}, 
    interp_range_factor::Real
) where {D, T, FI, FT, GEO}
    
    is_per_svec = geom.is_periodic
    dxs_f = Space{D, T}(nominal_dx...)
    N_ghost = any(is_per_svec) ? 0 : ceil(Int, interp_range_factor)
    
    canvas_mins = Space{D, T}(ntuple(d -> is_per_svec[d] ? geom.mins[d] : geom.mins[d] - N_ghost * dxs_f[d], Val(D)))
    canvas_maxs = Space{D, T}(ntuple(d -> is_per_svec[d] ? geom.maxs[d] : geom.maxs[d] + N_ghost * dxs_f[d], Val(D)))
    
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
    
    Ns_total = ntuple(Val(D)) do d
        max(1, round(Int, (cd.canvas_maxs[d] - cd.canvas_mins[d]) / dxs_f[d])) + 1
    end
    
    inner_points = Space{D, T}[]
    ghost_candidates = Space{D, T}[]
    
    for I in CartesianIndices(Ns_total)
        pos_tuple = ntuple(Val(D)) do d
            cd.canvas_mins[d] + (I[d] - 1) * dxs_f[d] + rand_f[d] * (rand(rng, T) * 2 - 1)
        end
        
        pos = Space{D, T}(pos_tuple)
        if geom.is_interior(pos)
            push!(inner_points, pos)
        else
            push!(ghost_candidates, pos)
        end
    end
    
    surviving_ghosts = Space{D, T}[]
    max_dx = maximum(dxs_f)
    cutoff_dist_sq = (T(interp_range_factor) * max_dx + max_dx)^2
    
    for ghost in ghost_candidates
        for inner in inner_points
            if sum(abs2, ghost - inner) <= cutoff_dist_sq
                push!(surviving_ghosts, ghost)
                break 
            end
        end
    end
    
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