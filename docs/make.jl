# docs/make.jl
using Documenter
using HyperCloudCore

makedocs(
    sitename = "HyperCloudCore.jl",
    modules = [HyperCloudCore],
    checkdocs = :exports,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://blhackslash.github.io/HyperCloudCore.jl/",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Particle Grid" => [
            "Overview" => "particle_grids/overview.md",
            "Domains" => "particle_grids/domains.md",
            "Boundary Conditions" => "particle_grids/boundary_conditions.md",
            "Weight Functions" => "particle_grids/weights.md"
        ],
        "Flux Functions" => "fluxes.md",
        "MLS Interpolation" => "mls_interpolation.md",
        "MOOD" => "mood.md",
        
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
        "Types" => "types.md",
        "API" => "api.md",
    ]
)

deploydocs(
    repo = "github.com/blhackslash/HyperCloudCore.jl.git",
    devbranch = "main",
    push_preview = true,
)