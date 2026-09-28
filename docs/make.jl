# docs/make.jl
using Documenter
using HyperCloud

makedocs(
    sitename = "HyperCloud.jl",
    modules = [HyperCloud],
    checkdocs = :none,
    remotes = nothing,  
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://blhackslash.github.io/HyperCloud.jl/",
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

        "Base API" => "base_api.md",
        "Advanced API" => "advanced_api.md",
    ]
)

deploydocs(
    repo = "github.com/blhackslash/HyperCloud.jl.git",
    devbranch = "main",
    push_preview = true,
)