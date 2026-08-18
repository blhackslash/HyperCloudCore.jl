function RectangularDomain(
    ::Type{T}, 
    interior_mins::NTuple{D, Real}, 
    interior_maxs::NTuple{D, Real}, 
    nominal_dx::NTuple{D, Real};
    is_periodic_input::Union{Bool, NTuple{D, Bool}} = false,
    interp_range_factor::Real = 2.0,
    bc_map::Dict{Int, AbstractBoundaryCondition} = Dict{Int, AbstractBoundaryCondition}(),
    tag_func = nothing
) where {D, T}
    
    is_per_svec = isa(is_periodic_input, Bool) ? SVector{D, Bool}(ntuple(_ -> is_periodic_input, Val(D))) : SVector{D, Bool}(is_periodic_input)
    
    mins_f = Space{D, T}(interior_mins...)
    maxs_f = Space{D, T}(interior_maxs...)
    dxs_f  = Space{D, T}(nominal_dx...)
    
    N_ghost = ceil(Int, interp_range_factor)
    canvas_mins = Space{D, T}(ntuple(d -> is_per_svec[d] ? mins_f[d] : mins_f[d] - N_ghost * dxs_f[d], Val(D)))
    canvas_maxs = Space{D, T}(ntuple(d -> is_per_svec[d] ? maxs_f[d] : maxs_f[d] + N_ghost * dxs_f[d], Val(D)))
    
    L_physical = maxs_f - mins_f
    L_wrap = Space{D, T}(ntuple(d -> is_per_svec[d] ? L_physical[d] : zero(T), Val(D)))
    invL_wrap = Space{D, T}(ntuple(d -> is_per_svec[d] ? one(T) / L_physical[d] : zero(T), Val(D)))
    L_inv = one(T) ./ max.(L_physical, T(1e-12))

    # --- NEW: is_interior logic ---
    is_interior_func = (pos) -> begin
        for d in 1:D
            # Periodic axes don't have a strict physical boundary to cross
            if !is_per_svec[d]
                if pos[d] < mins_f[d] || pos[d] > maxs_f[d]
                    return false
                end
            end
        end
        return true
    end

    actual_tag_func = if isnothing(tag_func)
        eps_tol = maximum(dxs_f) * T(1e-3)
        (pos) -> begin
            # We can now reuse is_interior_func to simplify the tagger!
            if !is_interior_func(pos)
                if D == 2
                    if pos[1] < mins_f[1] - eps_tol; return 1
                    elseif pos[1] > maxs_f[1] + eps_tol; return 2
                    elseif pos[2] < mins_f[2] - eps_tol; return 3
                    elseif pos[2] > maxs_f[2] + eps_tol; return 4
                    else; return 5
                    end
                else
                    return 1
                end
            else
                return 0 
            end
        end
    else
        tag_func
    end

    is_valid_func = (pos) -> true 

    return Domain{Val{:rectangular}, D, T, typeof(is_valid_func), typeof(is_interior_func), typeof(actual_tag_func)}(
        canvas_mins, canvas_maxs, mins_f, maxs_f, 
        is_per_svec, L_physical, L_inv, L_wrap, invL_wrap, 
        is_valid_func, is_interior_func, actual_tag_func, bc_map
    )
end

function get_points(
    domain::Domain{Val{:rectangular}, D, T},
    Ns_interior::NTuple{D, Integer};
    interp_range_factor::Real = 2.0,
    randomness::Tuple = ntuple(i -> zero(T), D),
    rng = Random.default_rng()
) where {D, T}
    
    is_per_svec = domain.is_periodic
    any_periodic = any(is_per_svec)
    N_ghost = any_periodic ? 0 : ceil(Int, interp_range_factor)
    
    mins_f = domain.interior_mins
    maxs_f = domain.interior_maxs
    rand_f = Space{D, T}(randomness...)
    
    # Recalculate exact dx based on the integer Ns_interior provided
    if any_periodic
        Ns_total = Ns_interior
        dxs_f = (maxs_f .- mins_f) ./ max.(Space{D, T}(Ns_interior...), T(1.0))
    else
        Ns_total = Ns_interior .+ 2 * N_ghost
        dxs_f = (maxs_f .- mins_f) ./ max.(Space{D, T}((Ns_interior .- 1)...), T(1.0))
    end
    
    N = prod(Ns_total)
    positions = Vector{Space{D, T}}(undef, N)
    is_boundary = zeros(Bool, N)
    tags = Vector{Int}(undef, N)
    
    for (i, I) in enumerate(CartesianIndices(Ns_total))
        pos_tuple = ntuple(Val(D)) do d
            idx = I[d]
            if is_per_svec[d]
                return mins_f[d] + dxs_f[d] * (idx - T(0.5)) + rand_f[d] * (rand(rng, T) * 2 - 1)
            else
                if idx <= N_ghost
                    return mins_f[d] - (N_ghost - idx + 1) * dxs_f[d]
                elseif idx > Ns_interior[d] + N_ghost
                    return maxs_f[d] + (idx - (Ns_interior[d] + N_ghost)) * dxs_f[d]
                else
                    base = Ns_interior[d] == 1 ? (mins_f[d] + maxs_f[d]) / T(2.0) : mins_f[d] + (idx - N_ghost - 1) * dxs_f[d]
                    return base + rand_f[d] * (rand(rng, T) * 2 - 1)
                end
            end
        end
        
        pos_svec = Space{D, T}(pos_tuple)
        positions[i] = pos_svec
        
        # Use the domain's embedded tagging logic
        tags[i] = domain.get_tag(pos_svec)
        is_boundary[i] = tags[i] != 0
    end
    
    volumes = fill(prod(dxs_f), N)
    
    return positions, is_boundary, tags, volumes, Tuple(dxs_f)
end