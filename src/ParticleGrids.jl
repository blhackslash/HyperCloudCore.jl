module ParticleGrids

export ParticleGrid, ParticleGrid1D, ParticleGrid2D, getPeriodicDistance, saveGrid, plotDensity, 
       animateDensity, getTimeStep, findLocalExtrema, updateVoxelInformation!, gridToLinearIndex, linearIndexToGrid, 
       findneighboringVoxels, updateNeighbors!, getEuclideanDistance, logMOODEvents!, findLocalExtremaAbs, sort_1d_particles!,
       determineVolumes!, getDistance, apply_boundary_conditions!, set_df!, getNBSlice, reorder_particles_for_locality!,
       manage_particles!, sort_particles!

using Random
using LinearAlgebra
using CellListMap
using StaticArrays
using Base.Threads # For Atomic operations
using ProgressMeter
using ..SimSettings
using ..HyperbolicPDEs
using ..MLSWeightFunctions

export get_positions, get_weights, get_xdistance, get_ydistance, get_neighbors
export ParticleGrid, GridMetadata, SharedBuffers, NeighborData, ReorderData, ManagementData, ParticleGridCore, createParticleGrid

# 1D Intercept
@inline get_positions(pg::ParticleGrid{1}) = reinterpret(Float64, pg.core.positions)
# 2D Normal Access
@inline get_positions(pg::ParticleGrid{2}) = pg.core.positions

# Column-Major Views for Neighbors
@inline get_weights(pg::ParticleGrid)   = @inbounds view(pg.neighbor.data, :, 1)
@inline get_xdistance(pg::ParticleGrid) = @inbounds view(pg.neighbor.data, :, 2)
@inline get_ydistance(pg::ParticleGrid) = @inbounds view(pg.neighbor.data, :, 3)
@inline get_neighbors(pg::ParticleGrid) = pg.neighbor.indices

# --- Aliases for convenience ---
const ParticleGrid1D{M, S, WF} = ParticleGrid{1, M, S, WF}
const ParticleGrid2D{M, S, WF} = ParticleGrid{2, M, S, WF}

function createParticleGrid(
    ::Val{1}, xmin::Real, xmax::Real, N_interior::Integer, bc::Symbol,
    interp_range_factor::Real;
    M::Int = 1, randomness::Real = 0.0, rng = Random.default_rng(), merge_factor = 0.2,
    weight_func = exponentialWeightFunction(1.,1.)
)
    N_ghost::Int = bc == :periodic ? 0 : ceil(Int, interp_range_factor)
    if bc == :periodic
        @assert N_ghost == 0 "Periodic grids do not use ghost cells."
        N = N_interior
        dx = (xmax - xmin) / max(N_interior, 1.0)
    else
        @assert N_ghost >= 0 "N_ghost must be non-negative."
        N = N_interior + 2 * N_ghost
        dx = (xmax - xmin) / max(N_interior - 1, 1.0)
    end

    xmin_tot = xmin - N_ghost * dx
    xmax_tot = xmax + N_ghost * dx
    h = dx * interp_range_factor
    regular = (randomness == 0.0)

    positions = Vector{SVector{1, Float64}}(undef, N)
    is_boundary = zeros(Bool,N)

    if bc == :periodic
        for i in 1:N_interior
            positions[i] = SVector(xmin + dx*(i-0.5) + randomness*(rand(rng, Float64)*2 - 1))
        end
    else
        for i in 1:N_ghost
            positions[i] = SVector(xmin - (N_ghost - i + 1) * dx)
            is_boundary[i] = true
        end
        for i in 1:N_interior
            base_pos = (N_interior == 1) ? (xmin+xmax)/2.0 : xmin + (i-1) * dx
            positions[N_ghost + i] = SVector(base_pos + randomness*(rand(rng, Float64)*2 - 1))
        end
        for i in 1:N_ghost
            positions[N_ghost + N_interior + i] = SVector(xmax + i * dx)
            is_boundary[N_ghost + N_interior + i] = true
        end
    end

    meta = GridMetadata{1}(
        N, N_interior, N_ghost, SVector(xmin_tot), SVector(xmax_tot), 
        h, SVector(dx), regular, bc, Float64(interp_range_factor), 0
    )

    core = ParticleGridCore{1}(positions, is_boundary, zeros(Int, N))
    shared = SharedBuffers{1, M}(zeros(N, M), similar(positions), zeros(Bool,N), zeros(Int, N))

    # Initialize ranges array
    neighbor = NeighborData{1, Nothing, typeof(weight_func)}(
        nothing, weight_func, fill(1:0, N + 1), Int[], 
        Matrix{Float64}(undef, 0, 2), 
        [Atomic{Int}(0) for _ in 1:N], [Atomic{Int}(0) for _ in 1:N]
    )

    permutation = collect(1:N)
    reorder = ReorderData{1}(permutation, copy(permutation), zeros(Int, N), zeros(Bool,N))

    min_nb = floor(Int, interp_range_factor)
    R = dx * interp_range_factor
    voxels = LocalVoxels(min_nb, R)
    manage = ManagementData{1}(zeros(Bool,N), Int[], SVector{D, Float64}[], NTuple{M, Float64}[], voxels)

    pg = ParticleGrid{1, M, Nothing, typeof(weight_func)}(
        meta, core, shared, neighbor, reorder, manage,
        zeros(N, M), zeros(Bool,N,M), zeros(N, M) 
    )

    pg.reorder(pg)
    pg.neighbor(pg)
    
    return pg
end

function createParticleGrid(
    ::Val{2}, xmin::Real, xmax::Real, ymin::Real, ymax::Real, 
    Nx_interior::Int, Ny_interior::Int, bc::Symbol, interp_range_factor::Real;
    M::Int = 1, randomness::NTuple{2, Float64} = (0.0, 0.0), rng = Random.default_rng(), 
    weight_func = exponentialWeightFunction(1.,1.)
)
    xmin_f, xmax_f = Float64(xmin), Float64(xmax)
    ymin_f, ymax_f = Float64(ymin), Float64(ymax)
    range_factor_f = Float64(interp_range_factor)
    rand_x, rand_y = Float64(randomness[1]), Float64(randomness[2])

    N_ghost::Int = bc == :periodic ? 0 : ceil(Int, range_factor_f)
    
    if bc == :periodic
        @assert N_ghost == 0 "Periodic grids do not use ghost cells."
        Nx_total, Ny_total = Nx_interior, Ny_interior
        dx_nominal = (xmax_f - xmin_f) / Nx_interior
        dy_nominal = (ymax_f - ymin_f) / Ny_interior
    else
        @assert N_ghost >= 0 "N_ghost must be non-negative."
        Nx_total = Nx_interior + 2*N_ghost
        Ny_total = Ny_interior + 2*N_ghost
        dx_nominal = (xmax_f - xmin_f) / max(Nx_interior - 1, 1.0)
        dy_nominal = (ymax_f - ymin_f) / max(Ny_interior - 1, 1.0)
    end
    
    N = Nx_total * Ny_total
    interp_range = range_factor_f < 1e-10 ? max(dx_nominal, dy_nominal) : range_factor_f * max(dx_nominal, dy_nominal)

    positions = Vector{SVector{2, Float64}}(undef, N)
    is_boundary = zeros(Bool,N)
    
    function _build_grid(sys)
        meta = GridMetadata{2}(
            N, Nx_interior * Ny_interior, N - (Nx_interior * Ny_interior), 
            SVector{2, Float64}(xmin_f, ymin_f), SVector{2, Float64}(xmax_f, ymax_f), 
            interp_range, SVector{2, Float64}(dx_nominal, dy_nominal), 
            (randomness == (0.0, 0.0)), bc, range_factor_f, 0
        )

        core = ParticleGridCore{2}(positions, is_boundary, zeros(Float64, N))
        shared = SharedBuffers{2, M}(zeros(N, M), similar(positions), zeros(Bool,N), zeros(Int, N))

        # Initialize ranges array
        neighbors = NeighborData{2, typeof(sys), typeof(weight_func)}(
            sys, weight_func, fill(1:0, N + 1), Int[], 
            Matrix{Float64}(undef, 0, 3),
            [Atomic{Int}(0) for _ in 1:N], [Atomic{Int}(0) for _ in 1:N]
        )

        reorder = ReorderData{2}(collect(1:N), collect(1:N), zeros(Int, N), zeros(Bool,N))
        manage = ManagementData{2}(zeros(Bool,N), Int[])

        return ParticleGrid{2, M, typeof(sys), typeof(weight_func)}(
            meta, core, shared, neighbors, reorder, manage,
            zeros(N, M), zeros(Bool,N,M), zeros(N, M)
        )
    end
    local pg
    if bc == :periodic
        for i in 1:Nx_total, j in 1:Ny_total
            index = (i - 1) * Ny_total + j
            posX = xmin_f + dx_nominal*(i-0.5) + rand_x*(rand(rng, Float64)*2 - 1)
            posY = ymin_f + dy_nominal*(j-0.5) + rand_y*(rand(rng, Float64)*2 - 1)
            positions[index] = SVector{2, Float64}(posX, posY)
        end
        unit_cell = SVector{2, Float64}(xmax_f - xmin_f, ymax_f - ymin_f)
        system = InPlaceNeighborList(x=positions, cutoff=interp_range, unitcell=unit_cell, parallel=true)
        pg =  _build_grid(system)
    else
        for i in 1:Nx_total, j in 1:Ny_total
            index = (i - 1) * Ny_total + j
            is_interior = (N_ghost < i <= Nx_interior + N_ghost) && (N_ghost < j <= Ny_interior + N_ghost)
            
            posX = i <= N_ghost ? xmin_f - (N_ghost-i+1)*dx_nominal : (i > Nx_interior+N_ghost ? xmax_f+(i-(Nx_interior+N_ghost))*dx_nominal : xmin_f+(i-N_ghost-1)*dx_nominal + rand_x*(rand(rng,Float64)*2-1))
            posY = j <= N_ghost ? ymin_f - (N_ghost-j+1)*dy_nominal : (j > Ny_interior+N_ghost ? ymax_f+(j-(Ny_interior+N_ghost))*dy_nominal : ymin_f+(j-N_ghost-1)*dy_nominal + rand_y*(rand(rng,Float64)*2-1))
            
            positions[index] = SVector{2, Float64}(posX, posY)
            is_boundary[index] = !is_interior
        end
        
        system = InPlaceNeighborList(x=positions, cutoff=interp_range, parallel=true)
        pg = _build_grid(system)
    end
    pg.neighbor(pg)
    pg.reorder(pg)
    pg.neighbor(pg)
    return pg
end

# Extractor simply returns the cached UnitRange
@inline function getNBSlice(pg::ParticleGrid, p_idx::Int)
    return pg.neighbor.ranges[p_idx]
end

sort_particles!(pg::ParticleGrid) = pg.reorder(pg) 

function (rd::ReorderData{D})(pg::ParticleGrid{D, M, S, WF}) where {D, M, S, WF}
    N = pg.meta.N
    visited = rd.seen_buffer
    fill!(visited, false)
    perm_idx = 0
    queue = Int[]
    neighbor_buffer = Int[]

    for i in 1:N 
        if !visited[i]
            start_node = i
            visited[start_node] = true
            resize!(queue, 0)
            push!(queue, start_node)

            while !isempty(queue)
                current_node = popfirst!(queue)
                perm_idx += 1
                rd.permutation[perm_idx] = current_node

                resize!(neighbor_buffer, 0)
                nb_slice = pg.neighbor.ranges[current_node]
                
                if !isempty(nb_slice)
                    @inbounds for k in nb_slice
                        nb_idx = pg.neighbor.indices[k]
                        if !visited[nb_idx]
                            visited[nb_idx] = true 
                            push!(neighbor_buffer, nb_idx)
                        end
                    end
                end
                
                # Sort neighbors by their degree (length of their UnitRange)
                sort!(neighbor_buffer, by = idx -> length(pg.neighbor.ranges[idx]))
                append!(queue, neighbor_buffer)
            end
        end
    end
    
    reverse!(rd.permutation)

    copyto!(pg.shared.pos_buffer, pg.core.positions)
    copyto!(pg.shared.bit_buffer, pg.core.is_boundary)
    copyto!(pg.shared.rho_buffer, pg.rhos)

    Threads.@threads for i in 1:N
        src_idx = rd.permutation[i]
        pg.core.positions[i]   = pg.shared.pos_buffer[src_idx]
        pg.core.is_boundary[i] = pg.shared.bit_buffer[src_idx]
        for m in 1:M
            pg.rhos[i, m] = pg.shared.rho_buffer[src_idx, m]
        end
    end

    Threads.@threads for i in 1:N
        rd.inv_permutation[rd.permutation[i]] = i
    end
    return nothing
end

function _build_connectivity_graph!(pg::ParticleGrid2D, system)
    N = pg.meta.N
    
    @threads for i in 1:N
        pg.neighbor.atomic_counts[i][] = 0
    end

    map_pairwise!(
        (xi, xj, i, j, d2, null) -> begin
            atomic_add!(pg.neighbor.atomic_counts[i], 1)
            atomic_add!(pg.neighbor.atomic_counts[j], 1)
            null
        end,
        0, system.box, system.cl; parallel = true
    )

    current_ptr = 1
    for i in 1:N
        num_nb = pg.neighbor.atomic_counts[i][]
        pg.neighbor.ranges[i] = current_ptr:(current_ptr + num_nb - 1)
        current_ptr += num_nb
    end
    total_neighbors = current_ptr - 1

    if length(pg.neighbor.indices) < total_neighbors
        resize!(pg.neighbor.indices, total_neighbors)
    end
    
    @threads for i in 1:N
        pg.neighbor.atomic_offsets[i][] = 0
    end
    
    atomic_offsets = pg.neighbor.atomic_offsets

    map_pairwise!(
        (xi, xj, i, j, d2, null) -> begin
            offset_i = atomic_add!(atomic_offsets[i], 1)
            write_idx_i = pg.neighbor.ranges[i].start + offset_i
            pg.neighbor.indices[write_idx_i] = j

            offset_j = atomic_add!(atomic_offsets[j], 1)
            write_idx_j = pg.neighbor.ranges[j].start + offset_j
            pg.neighbor.indices[write_idx_j] = i
            null
        end,
        0, system.box, system.cl; parallel = true
    )
    return nothing
end

updateNeighbors!(pg::ParticleGrid) = pg.neighbor(pg)

function (nd::NeighborData{1, S, WF})(pg::ParticleGrid{1, M, S, WF}) where {M, S, WF}
    N = pg.meta.N
    maxDist = pg.meta.range_factor * pg.meta.dx[1]
    weightFunc = nd.weight_func
    
    max_nb = 0
    total_neighbors = 0
    for i in 1:N
        num_nb = length(_find_neighbors_1d(pg, i, maxDist))
        pg.neighbor.ranges[i] = (total_neighbors + 1):(total_neighbors + num_nb)
        total_neighbors += num_nb
        max_nb = max(max_nb, num_nb)
    end
    pg.neighbor.ranges[N+1] = (total_neighbors + 1):total_neighbors
    pg.meta.max_nb = max_nb
    
    current_capacity = length(pg.neighbor.indices)
    if total_neighbors > current_capacity
        new_capacity = ceil(Int, total_neighbors * 1.25)
        resize!(pg.neighbor.indices, new_capacity)
        pg.neighbor.data = Matrix{Float64}(undef, new_capacity, 2)
    end

    offset_counts = zeros(Int, N) 
    for i in 1:N
        neighbor_list = _find_neighbors_1d(pg, i, maxDist)
        for j in neighbor_list
            offset = offset_counts[i]
            write_idx = pg.neighbor.ranges[i].start + offset
            
            dist_x = getDistance(pg, i, j) 
            d2 = dist_x^2

            pg.neighbor.indices[write_idx] = j
            pg.neighbor.data[write_idx, 1] = weightFunc(d2)
            pg.neighbor.data[write_idx, 2] = dist_x
            
            offset_counts[i] += 1
        end
    end
    determineVolumes!(pg) 
    return nothing
end

function getDistance(pg::ParticleGrid1D, i::Integer, j::Integer)
    dist = get_positions(pg)[j] - get_positions(pg)[i]
    if pg.meta.bc == :periodic
        domain_size = pg.meta.maxs[1] - pg.meta.mins[1]
        dist -= round(dist / domain_size) * domain_size
    end
    return dist
end

function _find_neighbors_1d(pg::ParticleGrid1D, i::Int, maxDist::Float64)
    N = pg.meta.N
    positions = get_positions(pg)
    pos_i = positions[i]
    
    neighbor_list = Vector{Int}()
    sizehint!(neighbor_list, 2 * ceil(Int, maxDist / pg.meta.dx[1]) + 2)

    if pg.meta.bc == :periodic
        for j_offset in 1:div(N, 2)
            j = mod1(i - j_offset, N)
            dist = abs(getDistance(pg, i, j))
            if dist <= maxDist; push!(neighbor_list, j)
            else; break; end
        end
        for j_offset in 1:div(N, 2)
            j = mod1(i + j_offset, N)
            dist = abs(getDistance(pg, i, j))
            if dist <= maxDist; push!(neighbor_list, j)
            else; break; end
        end
    else 
        for j in (i-1):-1:1
            if abs(positions[j] - pos_i) <= maxDist; push!(neighbor_list, j)
            else; break; end
        end
        for j in (i+1):N
            if abs(positions[j] - pos_i) <= maxDist; push!(neighbor_list, j)
            else; break; end
        end
    end
    return neighbor_list
end

function (nd::NeighborData{D, S, WF})(pg::ParticleGrid{D, M, S, WF}) where {D, M, S, WF}
    system = nd.system
    weightFunc = nd.weight_func    

    CellListMap.update!(system, pg.core.positions)

    @inbounds for i in 1:pg.meta.N; nd.atomic_counts[i][] = 0; end
    atomic_counts = nd.atomic_counts

    map_pairwise!(
        (xi, xj, i, j, d2, null) -> begin
            atomic_add!(atomic_counts[i], 1)
            atomic_add!(atomic_counts[j], 1)
            null
        end,
        0, system.box, system.cl; parallel = true
    )

    max_so_far = 0 
    current_ptr = 1
    @inbounds for i in 1:pg.meta.N
        count = atomic_counts[i][]
        pg.neighbor.ranges[i] = current_ptr:(current_ptr + count - 1)
        current_ptr += count
        if count > max_so_far
            max_so_far = count
        end
    end
    pg.neighbor.ranges[pg.meta.N + 1] = current_ptr:(current_ptr - 1)
    pg.meta.max_nb = max_so_far

    total_neighbors = current_ptr - 1
    current_capacity = length(pg.neighbor.indices)
    if total_neighbors > current_capacity
        new_capacity = ceil(Int, total_neighbors * 1.25)
        resize!(pg.neighbor.indices, new_capacity)
        pg.neighbor.data = Matrix{Float64}(undef, new_capacity, D+1)
    end
    
    @inbounds for i in 1:pg.meta.N; nd.atomic_offsets[i][] = 0; end
    atomic_offsets = nd.atomic_offsets
    
    map_pairwise!(
        (xi, xj, i, j, d2, null) -> begin
            dist = xj - xi 
            if pg.meta.bc == :periodic
                domainSize = pg.meta.maxs - pg.meta.mins
                dist -= round.(dist ./ domainSize) .* domainSize
            end
            weight = weightFunc(d2)

            offset_i = atomic_add!(atomic_offsets[i], 1)
            write_idx_i = pg.neighbor.ranges[i].start + offset_i
            pg.neighbor.indices[write_idx_i] = j
            pg.neighbor.data[write_idx_i, 1] = weight
            for d in 1:D; pg.neighbor.data[write_idx_i, 1 + d] = dist[d]; end

            offset_j = atomic_add!(atomic_offsets[j], 1) 
            write_idx_j = pg.neighbor.ranges[j].start + offset_j
            pg.neighbor.indices[write_idx_j] = i
            pg.neighbor.data[write_idx_j, 1] = weight
            for d in 1:D; pg.neighbor.data[write_idx_j, 1 + d] = -dist[d]; end
            
            null
        end,
        0, system.box, system.cl; parallel = true
    )
    return nothing
end

reorder_particles!(pg::ParticleGrid) = pg.reorder(pg)

function (rd::ReorderData{1})(pg::ParticleGrid{1, M, S, WF}) where {M, S, WF}
    N = pg.meta.N
    range = (N + 1):length(pg.core.positions)
    p = [sortperm(pg.core.positions[1:N]); collect(range)]

    if issorted(p); return nothing; end

    Base.permute!(pg.core.positions, p)
    Base.permute!(pg.core.is_boundary, p)
    pg.rhos .= pg.rhos[p, :]
    pg.curvatures .= pg.curvatures[p, :]
    pg.mood_events .= pg.mood_events[p, :]
    return nothing
end

# --- 1D BCs (Matrix) ---
function apply_boundary_conditions!(pg::ParticleGrid{1}, rhos_buffer::AbstractMatrix)
    bc = pg.meta.bc
    if bc == :periodic; return; end
    if bc == :fixed_dirichlet
        @inbounds for i in 1:pg.meta.N
            if pg.core.is_boundary[i]
                rhos_buffer[i, :] .= pg.rhos[i, :]
            end
        end
    elseif bc == :outflow
        first_int = findfirst(==(false), pg.core.is_boundary)
        last_int  = findlast(==(false), pg.core.is_boundary)
        if isnothing(first_int) || isnothing(last_int); return; end
        
        val_left = rhos_buffer[first_int, :]
        val_right = rhos_buffer[last_int, :]
        for i in 1:(first_int-1); rhos_buffer[i, :] .= val_left; end
        for i in (last_int+1):pg.meta.N; rhos_buffer[i, :] .= val_right; end
    end
    return nothing
end

# --- 1D BCs (Vector) ---
function apply_boundary_conditions!(pg::ParticleGrid{1}, rhos_buffer::AbstractVector)
    bc = pg.meta.bc
    if bc == :periodic; return; end
    if bc == :fixed_dirichlet
        @inbounds for i in 1:pg.meta.N
            if pg.core.is_boundary[i]
                rhos_buffer[i] = pg.rhos[i, 1]
            end
        end
    elseif bc == :outflow
        first_int = findfirst(==(false), pg.core.is_boundary)
        last_int  = findlast(==(false), pg.core.is_boundary)
        if isnothing(first_int) || isnothing(last_int); return; end
        
        val_left = rhos_buffer[first_int]
        val_right = rhos_buffer[last_int]
        rhos_buffer[1:(first_int-1)] .= val_left
        rhos_buffer[(last_int+1):end] .= val_right
    end
    return nothing
end

# --- 2D BCs (Matrix) ---
function apply_boundary_conditions!(pg::ParticleGrid{2}, rhos_buffer::AbstractMatrix)
    bc = pg.meta.bc
    if bc == :periodic; return; end
    if bc == :fixed_dirichlet
        @inbounds for i in 1:pg.meta.N
            if pg.core.is_boundary[i]
                rhos_buffer[i, :] .= pg.rhos[i, :]
            end
        end
    end
    return nothing
end

# --- 2D BCs (Vector) ---
function apply_boundary_conditions!(pg::ParticleGrid{2}, rhos_buffer::AbstractVector)
    bc = pg.meta.bc
    if bc == :periodic; return; end
    if bc == :fixed_dirichlet
        @inbounds for i in 1:pg.meta.N
            if pg.core.is_boundary[i]
                rhos_buffer[i] = pg.rhos[i, 1]
            end
        end
    end
    return nothing
end

function determineVolumes!(pg::ParticleGrid1D)
    N = pg.meta.N
    if N == 0; return; end
    positions = get_positions(pg)
    volumes = pg.core.volumes
    
    if pg.meta.bc == :periodic
        for i in 1:N
            prev_idx = mod1(i - 1, N)
            next_idx = mod1(i + 1, N)
            deltaPosL = abs(getDistance(pg, i, prev_idx))
            deltaPosR = abs(getDistance(pg, i, next_idx))
            volumes[i] = (deltaPosL + deltaPosR) / 2.0
        end
    else
        for i in 1:N
            if pg.core.is_boundary[i]; continue end
            volumes[i] = (positions[i+1] - positions[i-1]) / 2.0
        end
    end
    return
end

function findLocalExtrema(rho_i::Float64, nb_slice::UnitRange{Int}, neighbor_fs::AbstractVector{Float64})::Tuple{Float64, Float64}
    minU = maxU = rho_i
    @inbounds for k in nb_slice 
        rho_j = neighbor_fs[k]
        minU = min(minU, rho_j)
        maxU = max(maxU, rho_j)
    end
    return (minU, maxU)
end

function findLocalExtremaAbs(curve_i::Float64, nb_slice::UnitRange{Int}, neighbor_indices_full::Vector{Int}, curveVec::AbstractVector{Float64})::Tuple{Float64, Float64, Float64, Float64}
    mini = maxi = curve_i
    minAbs = maxAbs = abs(curve_i)
    
    @inbounds for k in nb_slice
        j = neighbor_indices_full[k]
        curve_j = curveVec[j]
        abs_curve_j = abs(curve_j)
        mini = min(mini, curve_j); maxi = max(maxi, curve_j)
        minAbs = min(minAbs, abs_curve_j); maxAbs = max(maxAbs, abs_curve_j)
    end
    return (mini, maxi, minAbs, maxAbs)
end

function findLocalExtremaAbs(
    curve_xx_i::Float64, curve_yy_i::Float64, nb_slice::UnitRange{Int}, neighbor_indices_full::Vector{Int}, 
    curve_xx_Vec::AbstractVector{Float64}, curve_yy_Vec::AbstractVector{Float64}
)::NTuple{8, Float64}
    mini1 = maxi1 = curve_xx_i
    mini2 = maxi2 = curve_yy_i
    minAbs1 = maxAbs1 = abs(curve_xx_i)
    minAbs2 = maxAbs2 = abs(curve_yy_i)
    
    @inbounds for k in nb_slice
        j = neighbor_indices_full[k] 
        curve_xx_j = curve_xx_Vec[j]; curve_yy_j = curve_yy_Vec[j]
        abs_curve_xx_j = abs(curve_xx_j); abs_curve_yy_j = abs(curve_yy_j)

        mini1 = min(mini1, curve_xx_j); maxi1 = max(maxi1, curve_xx_j)
        minAbs1 = min(minAbs1, abs_curve_xx_j); maxAbs1 = max(maxAbs1, abs_curve_xx_j)
        
        mini2 = min(mini2, curve_yy_j); maxi2 = max(maxi2, curve_yy_j)
        minAbs2 = min(minAbs2, abs_curve_yy_j); maxAbs2 = max(maxAbs2, abs_curve_yy_j)
    end
    return (mini1, maxi1, minAbs1, maxAbs1, mini2, maxi2, minAbs2, maxAbs2)
end

function getTimeStep(pg::ParticleGrid{1}, eq)
    dtMax = Inf
    vel = velocity(eq, 0.0)

    for i in 1:pg.meta.N
        if pg.core.is_boundary[i]; continue; end
        
        nb_slice = pg.neighbor.ranges[i]
        if isempty(nb_slice); continue; end

        num = 0.0; denum = 0.0
        w_vec = get_weights(pg)
        dx_vec = get_xdistance(pg)
        
        @inbounds for k in nb_slice
            w  = w_vec[k]; dx = dx_vec[k]
            if ((vel >= 0.0) && (dx <= 0.0)) || ((vel <= 0.0) && (dx >= 0.0))
                num += w * dx
                denum += w * dx * dx
            end
        end

        if abs(vel * num) > 1e-14
            dtMax = min(-denum / (vel * num), dtMax)
        end
    end
    return dtMax
end

function getTimeStep(pg::ParticleGrid{2}, eq)
    dtMax = Inf
    vel = eq.vel

    for i in 1:pg.meta.N
        if pg.core.is_boundary[i]; continue; end
        
        nb_slice = pg.neighbor.ranges[i]
        if isempty(nb_slice); continue; end
        
        w_vec = get_weights(pg); dx_vec = get_xdistance(pg); dy_vec = get_ydistance(pg)
        
        A11 = 0.0; A12 = 0.0; A22 = 0.0
        @inbounds for k in nb_slice
            w  = w_vec[k]; dx = dx_vec[k]; dy = dy_vec[k]
            A11 += w * dx * dx
            A12 += w * dx * dy
            A22 += w * dy * dy
        end

        D = A11 * A22 - (A12^2)
        if abs(D) < 1e-14; continue; end

        sumCij = 0.0
        @inbounds for k in nb_slice
            w  = w_vec[k]; dx = dx_vec[k]; dy = dy_vec[k]
            coeff_x = (A22 * w * dx - A12 * w * dy) / D
            coeff_y = (A11 * w * dy - A12 * w * dx) / D
            
            angle = atan(dy, dx)
            n_x, n_y = cos(angle), sin(angle)
            s_x, s_y = -n_y, n_x
            
            alfaBar = n_x * coeff_x + n_y * coeff_y
            betaBar = s_x * coeff_x + s_y * coeff_y
            
            dot_vel_n = vel[1] * n_x + vel[2] * n_y
            dot_vel_s = vel[1] * s_x + vel[2] * s_y
            
            bracketMinus = dot_vel_n > 0.0 ? 0.0 : dot_vel_n
            bracketMinus2 = betaBar * dot_vel_s > 0.0 ? 0.0 : betaBar * dot_vel_s
            
            sumCij -= alfaBar * bracketMinus + bracketMinus2
        end
        
        if abs(sumCij) > 1e-14
            dtMax = min(1 / (2 * sumCij), dtMax)
        end
    end
    return dtMax
end

include("ParticleManagement.jl")

end  # module ParticleGrids