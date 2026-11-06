using HyperCloudCore
using StaticArrays

# =========================================================================
# 1. DEFINE THE PHYSICS (CUSTOM PDE)
# =========================================================================
struct SimpleAdvection{D, M, T} <: HyperbolicPDE{D, M, T,}
    vel::SVector{D, T}
end

@inline HyperCloudCore.flux(eq::SimpleAdvection{D, M, T}, U::State{M, T}) where {D, M, T} = 
    Flux{D, M, T}(ntuple(d -> eq.vel[d] * U, Val(D)))

@inline HyperCloudCore.max_eigenvalue(eq::SimpleAdvection, U::State, d::Int) = 
    abs(eq.vel[d])

@inline HyperCloudCore.velocity(eq::SimpleAdvection{D, M, T}, U::State{M, T}, d::Int) where {D, M, T} = 
    SMatrix{M, M, T}(eq.vel[d])

eq = SimpleAdvection{2, 1, Float64}(SVector(1.0, 1.0))

# =========================================================================
# 2. DEFINE THE GEOMETRY & BOUNDARIES
# =========================================================================
bc_map = Dict{Int, AbstractBoundaryCondition}(
    1 => FixedDirichlet(), # Left boundary
    2 => OutflowBC(),      # Right boundary
    3 => FixedDirichlet(), # Bottom boundary
    4 => OutflowBC()       # Top boundary
)

domain = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); bc_map=bc_map)

# =========================================================================
# 3. GENERATE THE PARTICLE GRID & INITIALIZE
# =========================================================================
nominal_dx = (0.025, 0.025)

# The grid now strictly requires a fully instantiated weight function 
# containing the exact physical cutoff radius
interp_range_factor = 2.5
cutoff_radius = interp_range_factor * maximum(nominal_dx)
weight_func = ExponentialWeightFunction(1.0, cutoff_radius) # alpha = 1.0

# Build grid (Domain, nominal_dx, weight_func, Number of Equations M)
pg = ParticleGrid(domain, nominal_dx, weight_func, 1)

positions = HyperCloudCore.get_positions(pg)
x_coords = [pos[1] for pos in positions]
y_coords = [pos[2] for pos in positions]

# Initialize with Gaussian pulse centered at (0.25, 0.25)
for i in 1:pg.meta.N
    pos = positions[i]
    r2 = (pos[1] - 0.25)^2 + (pos[2] - 0.25)^2
    pg.rhos[i] = State{1, Float64}((exp(-100.0 * r2),))
end

# Cache initial state for plotting
rho_initial = [pg.rhos[i][1] for i in 1:pg.meta.N]

# =========================================================================
# 4. CONFIGURE NUMERICS
# =========================================================================
# Upwind strictly requires (T, D, M, order, algType, flux)
main_grad = UpwindDivergence(Float64, 2, 1, 1, :Classic, UpwindFlux())

# The TimeStepper now universally orchestrates MOOD (NoMOOD by default)
mood = MOOD() 

# Utilizing a simple Forward Euler explicit tableau
tableau = RK1_Euler_Tableau(Float64)
time_stepper = GeneralRKTimeStepper(eq, main_grad, mood, tableau)

# =========================================================================
# 5. EXECUTE MAIN INTEGRATION LOOP
# =========================================================================
t = 0.0
t_end = 0.5
cfl = 0.4 # Stability factor for explicit Euler

println("Starting simulation...")
while t < t_end
    dt = cfl * get_time_step(pg, eq, main_grad)
    dt = min(dt, t_end - t)
    
    time_stepper(eq, pg, t, dt)
    global t += dt
end
println("Simulation complete.")

# Extract final state
rho_final = [pg.rhos[i][1] for i in 1:pg.meta.N]

# =========================================================================
println("N = ", pg.meta.N, " max rho final = ", maximum(rho_final), " min = ", minimum(rho_final), " t = ", t)
