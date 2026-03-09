using Test
using StaticArrays
using Base.Threads: Atomic
using Meshfree4ScalarEq.ParticleGrids
using Meshfree4ScalarEq.HyperbolicPDEs
# using CellListMap # Ensure this is available in your environment

# =========================================================================
# 1. Mocks & Stubs (To run independently of your physics modules)
# =========================================================================

# Mock Weight Function
exponentialWeightFunction(alpha, beta) = (d2) -> exp(-alpha * sqrt(d2) / beta)

#velocity(eq::LinearAdvection{1}, t) = eq.vel[1]

# Fallback dummy for _find_neighbors_1d if it's not loaded in your test env
function _find_neighbors_1d(pg, i, maxDist)
    neighbors = Int[]
    pos_i = pg.core.positions[i][1]
    for j in 1:pg.meta.N
        if i != j
            dist = abs(pg.core.positions[j][1] - pos_i)
            # Simple periodic wrap for mock
            if pg.meta.bc == :periodic
                L = pg.meta.maxs[1] - pg.meta.mins[1]
                dist = min(dist, L - dist)
            end
            if dist <= maxDist
                push!(neighbors, j)
            end
        end
    end
    return neighbors
end

# Fallback dummy for determineVolumes! if not loaded
#determineVolumes!(pg) = nothing

# =========================================================================
# 2. Test Suite
# =========================================================================

@testset "ParticleGrid System Tests" begin

    @testset "1D ParticleGrid Initialization & Property Forwarding" begin
        # Create a 1D periodic grid with 2 system variables (M=2)
        pg1 = createParticleGrid(Val(1), 0.0, 1.0, 10, :periodic, 1.5; M=2)
        
        # Test Property Forwarding
        @test pg1.N == 10
        @test length(pg1.positions) == 10
        @test pg1.positions[1] isa Float64
        
        # Test matrix allocations
        @test size(pg1.rhos) == (10, 2)
        @test size(pg1.core.neighbor_data) == (0, 2) # Initially empty
        
        # Test Boundaries
        @test sum(pg1.is_boundary) == 0 # Periodic means no boundary particles
    end

    @testset "1D Functors: Neighbors, Sorting, and Timestep" begin
        pg1 = createParticleGrid(Val(1), 0.0, 1.0, 20, :outflow, 1.5; M=1)
        
        # 1. Test Neighbor Update
        updateNeighbors!(pg1)
        @test pg1.max_nb > 0
        @test size(pg1.core.neighbor_data, 2) == 2 # Row 1=weight, Row 2=dx
        @test length(pg1.core.neighbor_indices) > 0
        
        # 2. Test Sorting
        # Artificially scramble positions to test the sort functor
        pg1.positions[1] = 99.0
        sort_particles!(pg1)
        @test issorted([p[1] for p in pg1.positions])
        
        # 3. Test Timestep Calculation
        eq = LinearAdvection((1.0,))
        updateNeighbors!(pg1) # Rebuild graph after sorting!
        dt = getTimeStep(pg1, eq)
        @test dt > 0.0
        @test dt != Inf
        
        # 4. Test Boundary Conditions (1D Outflow)
        rho_buffer = rand(pg1.N)
        apply_boundary_conditions!(pg1, rho_buffer)
        # For outflow, the ghost cells should match the first/last interior cells
        interior_start = findfirst(==(false), pg1.is_boundary)
        @test rho_buffer[1] == rho_buffer[interior_start]

    end

    # NOTE: The 2D tests assume CellListMap is loaded and functional. 
    # If CellListMap throws errors in a pure test environment, ensure it is imported.
    @testset "2D ParticleGrid Initialization & Functors" begin
        # Create a 2D fixed dirichlet grid
        pg2 = createParticleGrid(Val(2), 0.0, 1.0, 0.0, 1.0, 5, 5, :fixed_dirichlet, 1.5; M=3)
        
        @test pg2.N > 25 # 25 interior + ghost cells
        @test pg2.positions[1] isa SVector{2, Float64}
        @test size(pg2.rhos) == (pg2.N, 3) # M=3 variables
        
        # 1. Test Neighbor Update
        updateNeighbors!(pg2)
        @test pg2.max_nb > 0
        @test size(pg2.core.neighbor_data, 2) == 3 # Row 1=weight, Row 2=dx, Row 3=dy
        
        # 2. Test Sorting (RCM Reordering)
        # RCM should generate a valid permutation containing all indices 1:N
        sort_particles!(pg2)
        @test sort(pg2.reorder.permutation) == collect(1:pg2.N)
        
        # 3. Test Timestep Calculation
        eq2 = LinearAdvection((1.0, -1.0))
        updateNeighbors!(pg2) # Rebuild after reorder
        dt = getTimeStep(pg2, eq2)
        @test dt > 0.0
        
        # 4. Test Boundary Conditions (2D Dirichlet)
        # Assign a distinct value to the grid's persistent rhos
        fill!(pg2.rhos, 5.0) 
        rho_buffer = zeros(pg2.N, 3)
        apply_boundary_conditions!(pg2, rho_buffer)
        
        # Find a boundary particle and ensure the buffer received the Dirichlet value
        boundary_idx = findfirst(==(true), pg2.is_boundary)
        if !isnothing(boundary_idx)
            @test rho_buffer[boundary_idx, 1] == 5.0
        end
    end
end
pg1 = createParticleGrid(Val(1), 0.0, 1.0, 10, :periodic, 1.5; M=2)
pg2 = createParticleGrid(Val(2), 0.0, 1.0, 0.0, 1.0, 5, 5, :fixed_dirichlet, 1.5; M=3)
function test_access(pg)
    return pg.positions
end

using InteractiveUtils # Required for @code_warntype in some environments

function test_grid_access(pg)
    # Test a direct property
    a = pg.rhos
    # Test a forwarded property (Meta)
    b = pg.N
    # Test a forwarded property (Core)
    c = pg.positions
    # Test our special view
    d = pg.neighbor_xdistance
    return a, b, c, d
end

# Call it once to compile
test_grid_access(pg1)

# Now ask the compiler what it sees
@code_warntype test_grid_access(pg1)

@code_typed test_access(pg2)


