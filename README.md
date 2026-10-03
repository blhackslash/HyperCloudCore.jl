# HyperCloudCore.jl

[![Build Status](https://github.com/blhackslash/HyperCloudCore.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/blhackslash/HyperCloudCore.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/blhackslash/HyperCloudCore.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/blhackslash/HyperCloudCore.jl)
[![Stable Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://blhackslash.github.io/HyperCloudCore.jl/dev/)

**HyperCloudCore.jl** is a high-performance, physics-agnostic meshless solver backend designed for the numerical integration of hyperbolic Partial Differential Equations (PDEs) in Julia. 

By operating entirely on unstructured point clouds, the framework completely bypasses traditional mesh-generation bottlenecks. It achieves this using **Generalized Finite Differences (GFD)** driven by **Moving Least Squares (MLS)** approximations to construct highly accurate spatial operators directly on arbitrary node distributions.

Built with a zero-overhead, trait-based API, `HyperCloudCore` provides a strict but flexible mathematical engine. Users simply define their governing physical equations, fluxes, and source terms, while the backend autonomously handles the complex spatial reconstruction, stabilization, and time integration.

### Core Capabilities
* **Meshless Spatial Discretization:** High-order spatial gradients via MLS, supporting advanced reconstruction schemes like **MUSCL** (stabilized by Multi-Dimensional Optimal Order Detection, or **MOOD**) and meshless **WENO**.
* **Advanced Time Integration:** A robust suite of time-steppers including explicit Strong Stability Preserving Runge-Kutta (**SSP-RK**) methods and **IMEX** (Implicit-Explicit) tableaus for resolving stiff operator interactions.
* **Unified Source Term API:** Zero-cost tuple unrolling for stacking multiple explicit (e.g., gravity, body forces) and implicit (e.g., stiff kinetic relaxation) source terms natively into the sub-stages of the RK/IMEX tableaus.
* **Dimensional & Physical Generality:** A completely decoupled architecture allowing it to solve anything from simple 1D scalar advection to 3D non-linear gas dynamics (Euler) and kinetic relaxation systems without altering the core solver logic.

## Quick Start Guide: Step-by-Step Implementation
This guide details the complete configuration and execution of a numerical simulation using the HyperCloudCore framework. Because HyperCloudCore is a physics-agnostic mathematical engine, we will demonstrate how to define a custom governing equation from scratch, discretize it on a two-dimensional rectangular domain, and march the solution forward in time.

### 1. Define the Governing Equation
The mathematical system must be defined by creating a custom structure that subtypes the abstract `HyperbolicPDE` type. To satisfy the core engine's API contract, we explicitly extend three fundamental methods: `flux`, `max_eigenvalue`, and `velocity`. In this example, we formulate a simple 2D linear advection equation for a scalar variable (M=1) with a constant advection velocity of $(1.0, 1.0)$ along the domain diagonal.

### 2. Formulate the Computational Domain
We instantiate a standard rectangular geometry and map discrete boundary tags to appropriate physical boundary conditions.

### 3. Initialize the Particle Grid and State
The continuous geometry is seamlessly discretized into a meshfree point cloud based on a nominal spatial resolution, entirely eliminating the need for unstructured mesh generation.

### 4. Configure Spatial and Temporal Discretizations
We define a spatial divergence interpolator and pass it to an explicit Runge-Kutta time integrator to advance the system.

### 5. Create Time Loop
We create a simple time loop to integrate our linear advection equation up to time $t_{max}=.5$ using the functor of our Runge-Kutta time integrator.

### 6. Visualize Results
Finally, we plot the integrated results compared to the initial condition as a simple scatter.

---

### Complete Example Code

```julia
using HyperCloudCore
using StaticArrays
using Plots

# =========================================================================
# 1. DEFINE THE PHYSICS (CUSTOM PDE)
# =========================================================================
struct SimpleAdvection{D, M, T} <: HyperbolicPDE{D, M, T, Conservative}
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
# 6. PLOT INITIAL VS FINAL STATE
# =========================================================================
# Plot t = 0.0
p1 = scatter(
    x_coords, y_coords,
    zcolor=rho_initial,
    markersize=3.5,
    markerstrokewidth=0,
    aspect_ratio=:equal,
    xlims=(0, 1), ylims=(0, 1),
    title="Initial Condition (t = 0.0)",
    xlabel="x", ylabel="y",
    colorbar=true,
    color=:viridis
)

# Plot t = 0.5
p2 = scatter(
    x_coords, y_coords,
    zcolor=rho_final,
    markersize=3.5,
    markerstrokewidth=0,
    aspect_ratio=:equal,
    xlims=(0, 1), ylims=(0, 1),
    title="Solution at t = $(t_end)",
    xlabel="x", ylabel="y",
    colorbar=true,
    color=:viridis
)

# Combine into a side-by-side comparison and save
fig = plot(p1, p2, layout=(1, 2), size=(900, 400))
savefig(fig, "advection_comparison.png")
display(fig)
```

---

Portions of this codebase and documentation were drafted with the assistance of large language models (LLMs). All code has been human-reviewed, verified, and tested. If you notice any inaccuracies or unexpected behavior, please open an issue.
