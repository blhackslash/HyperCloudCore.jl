# Throughput of one time step: linear advection (a diagonal system of BENCH_M equations), MUSCL3, RK3,
# on a uniform periodic cloud in BENCH_D dimensions. Prints milliseconds per step.
#
#   BENCH_N=250 BENCH_D=2 BENCH_M=1 BENCH_T=Float32 BENCH_DEVICE=gpu BENCH_STAB=mood julia -t 8 --project=test/gpu benchmark/step.jl
#
# BENCH_N:      particles per axis (default 250); the cloud has BENCH_N^BENCH_D particles
# BENCH_D:      1, 2 (default) or 3 space dimensions
# BENCH_M:      number of equations (default 1): advection speeds differ per equation (diagonal system)
# BENCH_T:      Float64 (default) or Float32
# BENCH_DEVICE: cpu (default: the Polyester backend with the launched threads) or gpu (AMDGPU, ROCBackend)
# BENCH_STAB:   none (unlimited, no MOOD), limiter (Venkatakrishnan), mood (MOOD EPD1/u1), both (default)
# BENCH_MOVING: 1 runs a prescribed grid motion (neighbour search every step), default 0
# BENCH_STEPS:  timed steps (default 30), after 3 warm-up steps
#
# The `gpu` device needs an environment with AMDGPU.jl (test/gpu). The printed checksum is Σ ρ₁ over the
# particles: the same for every backend of one element type (round-off aside).

using HyperCloudCore, StaticArrays, LinearAlgebra, Printf
import HyperCloudCore: flux, max_eigenvalue, velocity

const T      = Dict("Float64" => Float64, "Float32" => Float32)[get(ENV, "BENCH_T", "Float64")]
const n      = parse(Int, get(ENV, "BENCH_N", "250"))
const D      = parse(Int, get(ENV, "BENCH_D", "2"))
const M      = parse(Int, get(ENV, "BENCH_M", "1"))
const device = get(ENV, "BENCH_DEVICE", "cpu")
const stab   = get(ENV, "BENCH_STAB", "both")
const moving = get(ENV, "BENCH_MOVING", "0") == "1"
const steps  = parse(Int, get(ENV, "BENCH_STEPS", "30"))

if device == "gpu"
    using AMDGPU
    AMDGPU.allowscalar(false)
    const Storage = ROCArray
else
    const Storage = Array
end
set_backend!(PolyesterBackend())                                 # grids are built on the host

# Diagonal linear system: equation m is advected with speed c[d][m] along axis d.
struct DiagAdvection{D, M, T, L} <: HyperbolicPDE{D, M, T}
    c::SVector{D, SVector{M, T}}
    amax::SVector{D, T}
    jac::SVector{D, SMatrix{M, M, T, L}}
end
function DiagAdvection{D, M, T}(c::SVector{D, SVector{M, T}}) where {D, M, T}
    amax = SVector{D, T}(ntuple(d -> maximum(abs, c[d]), Val(D)))
    jac  = SVector{D, SMatrix{M, M, T, M * M}}(ntuple(d -> SMatrix{M, M, T}(Diagonal(c[d])), Val(D)))
    return DiagAdvection{D, M, T, M * M}(c, amax, jac)
end
@inline flux(eq::DiagAdvection{D, M, T}, U::State{M, T}) where {D, M, T} = Flux{D, M, T}(ntuple(d -> eq.c[d] .* U, Val(D)))
@inline max_eigenvalue(eq::DiagAdvection, U::State, d::Int) = eq.amax[d]
@inline max_eigenvalue(eq::DiagAdvection{D, M, T}, U::State, d::Int, shift::Real) where {D, M, T} = maximum(abs, eq.c[d] .- T(shift))
@inline velocity(eq::DiagAdvection, U::State, d::Int) = eq.jac[d]

dx    = one(T) / n
geom  = get_rectangular_domain(T, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D); is_periodic = ntuple(_ -> true, D))
wf    = ExponentialWeightFunction(one(T), T(2.5) * dx)
mover = moving ? PrescribedMover((x, t) -> Space{D, T}(ntuple(d -> T(0.2) * sin(2 * T(π) * x[mod1(d + 1, D)]), Val(D)))) : NoGridMover()
pg    = ParticleGrid(geom, ntuple(_ -> dx, D), wf, M; mover = mover)
for i in 1:pg.meta.N
    x = pg.core.positions[i]
    g = exp(-sum(abs2, x .- T(0.5)) / T(0.02))
    pg.rhos[i] = State{M, T}(ntuple(m -> g * (1 + T(0.1) * (m - 1)), Val(M)))
end
Storage === Array || (pg = HyperCloudCore.device_grid(Storage, pg); set_backend!(ROCBackend()))

speeds  = SVector{D, SVector{M, T}}(ntuple(d -> SVector{M, T}(ntuple(m -> T(1 - 0.5 * (d - 1)) * (1 + T(0.2) * (m - 1)), Val(M))), Val(D)))
eq      = DiagAdvection{D, M, T}(speeds)
limiter = stab in ("limiter", "both") ? VenkatakrishnanLimiter(:hard) : NoLimiter()
mood    = stab in ("mood", "both") ? MOOD(EPD1(), MOODu1(T(1e-6))) : MOOD()
interp  = MUSCL(T, D, M, 3, limiter, RusanovFlux(); storage = Storage)
ts      = GeneralRKTimeStepper(eq, interp, mood, RK3_SSP_Tableau(T); storage = Storage)
dt      = T(0.3) * get_time_step(pg, eq, interp)

for k in 1:3; ts(eq, pg, k * dt, dt); end                     # warm-up (compilation)
t = @elapsed for k in 1:steps; ts(eq, pg, k * dt, dt); end
checksum = sum(first, Array(pg.rhos)[1:pg.meta.N])
@printf("%-7s %-3s D=%d M=%d N=%-8d stab=%-7s moving=%d threads=%d  ms/step = %8.2f  checksum = %.10g\n",
        T, device, D, M, pg.meta.N, stab, moving, Threads.nthreads(), 1000t / steps, checksum)
