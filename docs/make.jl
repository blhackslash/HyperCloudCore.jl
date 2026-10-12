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
        mathengine = Documenter.MathJax3(),
    ),
    pages = [
        "Home" => "index.md",
        "Particle Grid" => [
            "Overview"            => "particle_grids/overview.md",
            "Domains"             => "particle_grids/domains.md",
            "Boundary Conditions" => "particle_grids/boundary_conditions.md",
            "Weight Functions"    => "particle_grids/weights.md",
        ],
        "Spatial Discretization" => [
            "MLS Interpolation" => "spatial/mls_interpolation.md",
            "Numerical Fluxes"  => "spatial/fluxes.md",
            "Central"           => "spatial/central.md",
            "Upwind"            => "spatial/upwind.md",
            "MUSCL"             => "spatial/muscl.md",
            "WENO"              => "spatial/weno.md",
        ],
        "Time Integration" => [
            "Time Stepper"     => "time_integration/time_stepper.md",
            "Butcher Tableaus" => "time_integration/tableaus.md",
            "MOOD"             => "time_integration/mood.md",
            "Moving Grids"     => "time_integration/moving_grids.md",
        ],
        "Extending" => [
            "API Contracts"  => "extending/api.md",
            "Abstract Types" => "extending/types.md",
        ],
    ],
)

deploydocs(
    repo = "github.com/blhackslash/HyperCloudCore.jl.git",
    devbranch = "main",
    push_preview = true,
)
