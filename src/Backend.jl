# =========================================================================
# EXECUTION BACKENDS
#
# Every parallel per-particle loop of the package is written as a *kernel body*
# `body(i, args...)` that only touches the arrays and structs it is given, and is launched with
# `foreach_particle(body, backend, N, args)`. The backend decides how the loop runs:
#
#   PolyesterBackend()   Polyester `@batch per=core`        (default; fastest on the CPU)
#   ThreadsBackend()     Base.Threads `@threads :dynamic`    (no spin-waiting workers)
#   KABackend(b)         a KernelAbstractions.jl kernel on `b` (`CPU()`, or a GPU backend such as
#                        `CUDABackend()` / `ROCBackend()` from the vendor package)
#
# Bodies therefore have to obey the GPU kernel rules: no allocation, no dynamic dispatch, no
# `Dict` access, nothing captured from the enclosing method's static parameters (see also the
# Polyester rules at `@smart_parallel`). Serial parts (prefix sums, the MOOD decrement, sorting,
# boundary donor searches) stay on the host for now; see docs/src/gpu_roadmap.md.
# =========================================================================
using KernelAbstractions
using Adapt
const KA = KernelAbstractions

export ExecutionBackend, PolyesterBackend, ThreadsBackend, KABackend
export set_backend!, get_backend, foreach_particle

"Supertype of the ways a per-particle loop can be executed (see `Backend.jl`)."
abstract type ExecutionBackend end

"Polyester `@batch per=core` loops: the default CPU path."
struct PolyesterBackend <: ExecutionBackend end

"`Threads.@threads :dynamic` loops (no spin-waiting workers; handy next to other threaded code)."
struct ThreadsBackend <: ExecutionBackend end

"""
    KABackend(backend::KernelAbstractions.Backend; workgroupsize = 256)

Runs every per-particle loop as a KernelAbstractions kernel on `backend`: `CPU()` for testing the
kernels on the host, or a device backend from CUDA.jl / AMDGPU.jl / Metal.jl / oneAPI.jl. All
arrays reached by the kernels must live on that device (see docs/src/gpu_roadmap.md).
"""
struct KABackend{B <: KA.Backend} <: ExecutionBackend
    backend::B
    workgroupsize::Int
end
KABackend(backend::KA.Backend; workgroupsize::Int = 256) = KABackend(backend, workgroupsize)

const _BACKEND = Ref{ExecutionBackend}(PolyesterBackend())

"The execution backend of the per-particle loops (default `PolyesterBackend()`)."
get_backend() = _BACKEND[]

"""
    set_backend!(backend)

Selects how the per-particle loops run: an `ExecutionBackend`, or a `KernelAbstractions.Backend`
(wrapped in a `KABackend`). Returns the new backend.
"""
set_backend!(backend::ExecutionBackend) = (_BACKEND[] = backend)
set_backend!(backend::KA.Backend; kwargs...) = set_backend!(KABackend(backend; kwargs...))

"""
    KernelView

`Adapt` target that turns host objects into the form a kernel body sees (see `DeviceAdapt.jl`): mutable
metadata becomes an immutable snapshot, `Ref`s plain values, `Growable` buffers the arrays they hold, and
host-only parts are dropped. Arrays are left where they are. Every `foreach_particle` launch adapts its
arguments this way, so kernel bodies see the same argument types on every backend.
"""
struct KernelView end

"The kernel-body form of `x` (see `KernelView`)."
kernel_view(x) = adapt(KernelView(), x)

"""
    foreach_particle(body, backend, N, args::Tuple)

Runs `body(i, args...)` for `i in 1:N` on `backend`. `body` must be a kernel body in the sense of
`Backend.jl` (no allocation, no dynamic dispatch, only `args` as input). The arguments are passed
through `kernel_view` first; on a `KABackend` every array among them must live on that backend.
"""
function foreach_particle(body::F, ::PolyesterBackend, N::Int, args::Tuple) where {F}
    kargs = kernel_view(args)
    @batch per=core for i in 1:N
        body(i, kargs...)
    end
    return nothing
end

function foreach_particle(body::F, ::ThreadsBackend, N::Int, args::Tuple) where {F}
    kargs = kernel_view(args)
    Threads.@threads :dynamic for i in 1:N
        body(i, kargs...)
    end
    return nothing
end

@kernel function _foreach_kernel(body, args)
    i = @index(Global, Linear)
    body(i, args...)
end

function foreach_particle(body::F, b::KABackend, N::Int, args::Tuple) where {F}
    N == 0 && return nothing
    kargs = kernel_view(args)
    check_resident(b.backend, kargs)
    kernel = _foreach_kernel(b.backend, b.workgroupsize)
    kernel(body, kargs; ndrange = N)
    KA.synchronize(b.backend)
    return nothing
end

"""
    check_resident(backend::KernelAbstractions.Backend, x)

Throws an `ArgumentError` if any array reachable from `x` (through tuples and struct fields) does not live
on `backend`. Guards the kernel launches: AMDGPU.jl, for one, would otherwise copy a host array to the
device for the launch and silently drop the kernel's writes to it.
"""
@generated function check_resident(backend, x)
    (isbitstype(x) || x <: Type || !isconcretetype(x)) && return :(nothing)
    checks = [:(check_resident(backend, getfield(x, $i))) for i in 1:fieldcount(x)]
    return quote
        $(checks...)
        return nothing
    end
end

function check_resident(backend, x::AbstractArray)
    isbits(x) && return nothing
    KA.get_backend(x) == backend && return nothing
    throw(ArgumentError("a kernel argument of type $(typeof(x)) is not resident on $(backend); build the grid " *
                        "with `device_grid` and the interpolator and time stepper with the matching `storage` type"))
end
