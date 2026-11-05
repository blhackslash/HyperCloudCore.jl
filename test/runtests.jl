using Test
using HyperCloudCore
import HyperCloudCore: flux, velocity, max_eigenvalue, prim2cons, cons2prim, fluid_velocity,
                       implicit_solve, evaluate_source, pre_solve_update!,
                       math_min, math_max, basis_length, param2vel,
                       update_size!, update_content!, _extract_order, _order_reach, _divergence_degree,
                       has_cached_gradients, grid_velocity
using PDEStudioCore
using StaticArrays
using LinearAlgebra
using Random

include("ICs/_main.jl")
include("PDEs/_main.jl")
include("SimulationFunctions/_main.jl")
include("Builder/kinetic.jl")        # physics-side builder (kinetic relaxation) stays with the tests

include("kinetic_source.jl")
include("time_integration.jl")

PDEStudioCore.set_target_module!(@__MODULE__)

# Device tests: HYPERCLOUD_BACKEND=amdgpu adds the AMDGPU passes (run in the test/gpu environment, which
# has AMDGPU.jl; see test/gpu/Project.toml).
const GPU_BACKEND = get(ENV, "HYPERCLOUD_BACKEND", "")
if GPU_BACKEND == "amdgpu"
    using AMDGPU
    AMDGPU.allowscalar(false)
elseif GPU_BACKEND != ""
    error("HYPERCLOUD_BACKEND=$(GPU_BACKEND) is not supported (amdgpu)")
end

# -------------------------------------------------------------------------
# Shared test helpers
# -------------------------------------------------------------------------

"Uniform 1-D periodic grid on [-1, 1] with N particles (no randomness)."
function make_pg_1d(N::Int; mover = NoGridMover(), range = 2.5, T = Float64)
    dx   = T(2) / N
    geom = get_rectangular_domain(T, (-1.0,), (1.0,); is_periodic = true)
    wf   = ExponentialWeightFunction(one(T), T(range) * dx)
    return ParticleGrid(geom, (dx,), wf, 1; mover = mover)
end

"Scalar 1-D linear advection with speed `c`."
make_advection_1d(c = 1.0) = LinearAdvection(((c,),), Float64)

"Set a smooth initial condition `f(x)` on a scalar grid."
function set_ic!(pg, f)
    for i in 1:pg.meta.N
        pg.rhos[i] = State{1, Float64}(f(pg.core.positions[i][1]))
    end
    return pg
end

interior_indices(pg) = findall(i -> !pg.core.is_boundary[i], 1:pg.meta.N)

# -------------------------------------------------------------------------
# Test groups
# -------------------------------------------------------------------------
# The suite is dominated by compilation: every new combination of scheme, MOOD and tableau types compiles the
# time stepper anew (~10-20 s, about three times that on the CI runners), and compilation is single threaded.
# The groups are independent and run in separate processes in parallel:
#   HYPERCLOUDCORE_TEST_JOBS=n     at most n processes (default: the CPU threads, at most the number of groups
#                                  and one per 2.5 GiB of free memory); 1 runs every group in this process, one
#                                  after the other.
#   HYPERCLOUDCORE_TEST_GROUP=name runs one group in this process (what the parallel runner starts).
const TEST_GROUPS = [
    "core"       => ["unit_core.jl"],
    "backends-a" => ["unit_backends.jl"],
    "backends-b" => ["unit_backends.jl"],
    "schemes"    => ["unit_schemes.jl"],
    "pipeline"   => ["pipeline_tests.jl"],
]
const TEST_GROUP = get(ENV, "HYPERCLOUDCORE_TEST_GROUP", "")
TEST_GROUP == "" || TEST_GROUP in first.(TEST_GROUPS) ||
    error("HYPERCLOUDCORE_TEST_GROUP=$(TEST_GROUP) is not a test group ($(join(first.(TEST_GROUPS), ", ")))")

# The execution-backend parity cases are split over the two `backends` groups (all of them when run in one process).
_backend_cases(cases) = TEST_GROUP == "backends-a" ? cases[1:5] : TEST_GROUP == "backends-b" ? cases[6:end] : cases

function run_group(name; label = name)
    @testset "$label" begin
        for file in Dict(TEST_GROUPS)[name]
            include(file)
        end
    end
end

# Runs the groups in child processes (same julia flags, e.g. --check-bounds and coverage, and the same project) and
# prints each group's output when it finishes: in full if it failed, else its test summary.
function run_groups_in_parallel(jobs)
    names = first.(TEST_GROUPS)
    threads = max(1, Threads.nthreads() ÷ jobs)
    # The backend groups compare Polyester, Threads and KA runs: with one thread `@batch` runs inline and nothing
    # runs concurrently, so they get at least two (with -t 8 and 5 groups: 2 + 2 + 1 + 1 + 1 threads)
    group_threads(name) = startswith(name, "backends") ? max(2, threads) : threads
    cmd(name) = addenv(`$(Base.julia_cmd()) --project=$(Base.active_project()) --threads=$(group_threads(name)) $(@__FILE__)`,
                       "HYPERCLOUDCORE_TEST_GROUP" => name)
    logs = Dict(name => tempname() for name in names)
    queue, running, failed = copy(names), Dict{String, Tuple{Base.Process, IOStream, Float64}}(), String[]
    println("Running $(length(names)) test groups in up to $jobs processes ($threads thread(s) each, ",
            "$(group_threads("backends-a")) for the backend groups)")
    while !isempty(queue) || !isempty(running)
        while length(running) < jobs && !isempty(queue)
            name = popfirst!(queue)
            io = open(logs[name], "w")
            running[name] = (run(pipeline(cmd(name); stdout = io, stderr = io); wait = false), io, time())
        end
        sleep(0.5)
        for (name, (proc, io, t0)) in collect(running)
            process_exited(proc) || continue
            close(io)
            delete!(running, name)
            out = read(logs[name], String)
            ok = success(proc)
            ok || push!(failed, name)
            println("\n", "="^30, " test group `$name`: ", ok ? "passed" : "FAILED", " in $(round(time() - t0; digits = 1)) s")
            i = findlast("Test Summary:", out)
            print(ok && i !== nothing ? out[first(i):end] : out)
        end
    end
    @testset "HyperCloudCore" begin
        @testset "$name" for name in names
            @test !(name in failed)
        end
    end
end

const TEST_JOBS = parse(Int, get(ENV, "HYPERCLOUDCORE_TEST_JOBS",
    string(clamp(min(Sys.CPU_THREADS, Int(Sys.free_memory() ÷ (5 * 2^29))), 1, length(TEST_GROUPS)))))

set_stat_preset!("hyperbolic")
set_save_path!(mktempdir())
if TEST_GROUP != ""
    run_group(TEST_GROUP)
    println("test group `$(TEST_GROUP)`: peak memory $(Sys.maxrss() ÷ 2^20) MiB")
elseif TEST_JOBS <= 1 || GPU_BACKEND != ""
    @testset "HyperCloudCore" begin
        for name in unique(first.(TEST_GROUPS))
            name == "backends-b" && continue            # `backends-a` runs every case in one process
            run_group(name; label = name == "backends-a" ? "backends" : name)
        end
        GPU_BACKEND == "amdgpu" && include("gpu_tests.jl")
    end
else
    run_groups_in_parallel(TEST_JOBS)
end
