# Quick Start Guide

This guide details the complete configuration and execution of a numerical simulation using the HyperCloud framework. As a canonical explicit example, we will discretize and solve the linear advection equation on a two-dimensional rectangular domain.

## Step-by-Step Implementation

### 1. Define the Governing Equation
The mathematical system must be defined by instantiating an object that subtypes the abstract `HyperbolicPDE` type[span_0](start_span)[span_0](end_span). For this tutorial, we employ the in-built `LinearAdvection` model[span_1](start_span)[span_1](end_span). This structure rigorously conforms to the internal PDE API by defining explicit methods for the `flux`, `max_eigenvalue`, and `velocity` calculations[span_2](start_span)[span_2](end_span). We initialize a 2D scalar system (M=1) with a constant advection velocity of $(1.0, 1.0)$ along the domain diagonal[span_3](start_span)[span_3](end_span).

### 2. Formulate the Computational Domain
We instantiate a standard rectangular geometry and map discrete boundary tags to appropriate boundary conditions.

### 3. Initialize the Particle Grid and State
The continuous geometry is subsequently discretized into a meshfree point cloud based on a nominal spatial resolution.

### 4. Configure Spatial and Temporal Discretizations
We define a spatial divergence operator and pass it to an explicit Runge-Kutta time integrator to march the solution forward in time.

---

### Complete Example Code

```julia
using HyperCloud

# =========================================================================
# 1. DEFINE THE PHYSICS
# =========================================================================
# Configure the LinearAdvection PDE with velocity v_x = 1.0, v_y = 1.0[span_4](start_span)[span_4](end_span).
# The input is a Tuple containing the state vectors for each dimension[span_5](start_span)[span_5](end_span).
velocities = ((1.0,), (1.0,)) 
eq = LinearAdvection(velocities) 

# =========================================================================
# 2. DEFINE THE GEOMETRY & BOUNDARIES
# =========================================================================
# Map integer geometric boundary tags to explicit boundary condition functors.
bc_map = Dict{Int, AbstractBoundaryCondition}(
    1 => FixedDirichlet(), # Left boundary
    2 => OutflowBC(),      # Right boundary
    3 => FixedDirichlet(), # Bottom boundary
    4 => OutflowBC()       # Top boundary
)

# Instantiate a 2D unit square domain [0,1] x [0,1]
domain = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); bc_map=bc_map)

# =========================================================================
# 3. GENERATE THE PARTICLE GRID
# =========================================================================
# Discretize the domain with a nominal spacing and interaction radius multiplier
nominal_dx = (0.025, 0.025)
interp_range_factor = 2.5
pg = ParticleGrid(domain, nominal_dx, interp_range_factor; M=1)

# Initialize the state vector (M=1) with a Gaussian pulse centered at (0.25, 0.25)
for i in 1:pg.meta.N
    pos = get_positions(pg)[i]
    r2 = (pos[1] - 0.25)^2 + (pos[2] - 0.25)^2
    pg.rhos[i] = State{1, Float64}((exp(-100.0 * r2),))
end

# =========================================================================
# 4. CONFIGURE NUMERICS (SPATIAL & TEMPORAL)
# =========================================================================
# Initialize a 1st-order upwind spatial interpolator 
main_grad = UpwindDivergence(Float64, 2, 1, 1; flux=UpwindFlux(), algType="Classic")

# Initialize a 3rd-order Strong Stability Preserving (SSP) Runge-Kutta time stepper
tableau = RK3_SSP_Tableau(Float64)
time_stepper = GeneralRKTimeStepper(eq, main_grad, tableau)

# =========================================================================
# 5. EXECUTE MAIN INTEGRATION LOOP
# =========================================================================
t = 0.0
t_end = 0.5

println("Starting simulation...")
while t < t_end
    # Extract the maximum stable step size based on the exact geometric CFL
    dt = getTimeStep(pg, eq, main_grad)
    
    # Bound the final step size to precisely hit t_end
    dt = min(dt, t_end - t)
    
    # Advance the solution via the time stepper functor
    time_stepper(eq, pg, t, dt)
    
    global t += dt
    println("Integrated to t = $(round(t, digits=4)) \vert{} dt =$(round(dt, digits=6))")
end
println("Simulation complete.")
```