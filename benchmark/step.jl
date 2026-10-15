# Throughput of one time step: 2-D scalar advection, MUSCL3 with the Venkatakrishnan limiter, MOOD
# (EPD1 / u1), RK3, on a uniform periodic cloud. Prints milliseconds per step.
#
#   BENCH_N=250 BENCH_T=Float32 BENCH_DEVICE=gpu BENCH_MOVING=1 julia -t 8 --project=test/gpu benchmark/step.jl
#
# BENCH_N:      particles per axis (default 250 → 62 500 particles)
# BENCH_T:      Float64 (default) or Float32
# BENCH_DEVICE: cpu (default: the Polyester backend with the launched threads) or gpu (AMDGPU, ROCBackend)
# BENCH_MOVING: 1 runs a prescribed grid motion (neighbour search every step), default 0
# BENCH_STEPS:  timed steps (default 30), after 3 warm-up steps
#
# The `gpu` device needs an environment with AMDGPU.jl (test/gpu). The printed checksum is Σ ρ over the
# particles: the same for every backend of one element type (round-off aside).

using HyperCloudCore, StaticArrays, LinearAlgebra, Printf
import HyperCloudCore: flux, max_eigenvalue, velocity

T      = Dict("Float64" => Float64, "Float32" => Float32)[get(ENV, "BENCH_T", "Float64")]
n      = parse(Int, get(ENV, "BENCH_N", "250"))
device = get(ENV, "BENCH_DEVICE", "cpu")
moving = get(ENV, "BENCH_MOVING", "0") == "1"
steps  = parse(Int, get(ENV, "BENCH_STEPS", "30"))

if device == "gpu"
    using AMDGPU
    AMDGPU.allowscalar(false)
    const Storage = ROCArray
else
    const Storage = Array
end
set_backend!(PolyesterBackend())                                 # grids are built on the host

struct Adv2D{T} <: HyperbolicPDE{2, 1, T}; c::SVector{2, T}; end
@inline flux(eq::Adv2D{T}, U::State{1, T}) where {T} = Flux{2, 1, T}((eq.c[1] * U, eq.c[2] * U))
@inline max_eigenvalue(eq::Adv2D, U::State, d::Int) = abs(eq.c[d])
@inline max_eigenvalue(eq::Adv2D{T}, U::State, d::Int, shift::Real) where {T} = abs(eq.c[d] - T(shift))
@inline velocity(eq::Adv2D{T}, U::State, d::Int) where {T} = SMatrix{1, 1, T}(eq.c[d])

dx    = one(T) / n
geom  = get_rectangular_domain(T, (0.0, 0.0), (1.0, 1.0); is_periodic = (true, true))
wf    = ExponentialWeightFunction(one(T), T(2.5) * dx)
mover = moving ? PrescribedMover((x, t) -> Space{2, T}(T(0.2) * sin(2 * T(π) * x[2]), T(0.2) * cos(2 * T(π) * x[1]))) : NoGridMover()
pg    = ParticleGrid(geom, (dx, dx), wf, 1; mover = mover)
for i in 1:pg.meta.N
    x = pg.core.positions[i]
    pg.rhos[i] = State{1, T}(exp(-((x[1] - T(0.5))^2 + (x[2] - T(0.5))^2) / T(0.02)))
end
Storage === Array || (pg = HyperCloudCore.device_grid(Storage, pg); set_backend!(ROCBackend()))

eq     = Adv2D{T}(SVector{2, T}(1, 0.5))
interp = MUSCL(T, 2, 1, 3, VenkatakrishnanLimiter(:hard), RusanovFlux(); storage = Storage)
ts     = GeneralRKTimeStepper(eq, interp, MOOD(EPD1(), MOODu1(T(1e-6))), RK3_SSP_Tableau(T); storage = Storage)
dt     = T(0.3) * get_time_step(pg, eq, interp)

for k in 1:3; ts(eq, pg, k * dt, dt); end                     # warm-up (compilation)
t = @elapsed for k in 1:steps; ts(eq, pg, k * dt, dt); end
checksum = sum(first, Array(pg.rhos)[1:pg.meta.N])
@printf("%-7s %-5s N=%-8d moving=%d threads=%d  ms/step = %8.2f  checksum = %.10g\n",
        T, device, pg.meta.N, moving, Threads.nthreads(), 1000t / steps, checksum)
