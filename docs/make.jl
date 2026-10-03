# docs/make.jl
using Documenter
using HyperCloudCore

makedocs(
    sitename = "HyperCloudCore.jl",
    modules = [HyperCloudCore],
    checkdocs = :none,
    remotes = nothing,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://blhackslash.github.io/HyperCloudCore.jl/",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Flux Functions" => "fluxes.md",
        "MLS Interpolation" => "mls_interpolation.md",
        "Divergence Interpolators" => [
            "Central" => "interpolators/central.md",
            "Upwind" => "interpolators/upwind.md",
            "MUSCL" => "interpolators/muscl.md",
            "WENO" => "interpolators/weno.md",
        ],
        "Time Stepper" => [
            "Runge-Kutta" => "time_stepper/rk_timestepper.md",
            "IMEX" => "time_stepper/imex_timestepper.md",
        ],
        "Types" => [
            "Interfaces" => "types/interfaces.md",
            "Schemes" => "types/schemes.md",
            "Timesteppers" => "types/timesteppers.md"
        ],

        "Base API" => "base_api.md",
    ]
)

deploydocs(
    repo = "github.com/blhackslash/HyperCloudCore.jl.git",
    devbranch = "main",
    push_preview = true,
)