# docs/make.jl
using Documenter
using DocumenterInterLinks
using HyperCloudCore

# Cross-references into the physics wrapper, e.g. [`run_direct_simulation`](@extref HyperCloud).
# HYPERCLOUD_INVENTORY overrides the inventory, e.g. with a local HyperCloud.jl/docs/build/objects.inv
# to check references to names that are not deployed yet.
links = InterLinks(
    "HyperCloud" => (
        "https://blhackslash.github.io/HyperCloud.jl/dev/",
        get(ENV, "HYPERCLOUD_INVENTORY", "https://blhackslash.github.io/HyperCloud.jl/dev/objects.inv"),
    ),
)

makedocs(
    sitename = "HyperCloudCore.jl",
    modules = [HyperCloudCore],
    # Explicit, so the build does not depend on the `origin` remote (Forgejo checkouts).
    repo = Remotes.GitHub("blhackslash", "HyperCloudCore.jl"),
    plugins = [links],
    checkdocs = :exports,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://blhackslash.github.io/HyperCloudCore.jl/",
        assets = String[],
        mathengine = Documenter.MathJax3(),
    ),
    pages = [
        "Home" => "index.md",
        "Backends & GPU roadmap" => "gpu_roadmap.md",
        "Particle Grid" => [
            "Overview"            => "particle_grids/overview.md",
            "Domains"             => "particle_grids/domains.md",
            "Boundary Conditions" => "particle_grids/boundary_conditions.md",
            "Weight Functions"    => "particle_grids/weights.md",
            "Volumes & Adaptivity" => "particle_grids/volumes.md",
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
            "Builders"       => "extending/builders.md",
        ],
    ],
)

deploydocs(
    repo = "github.com/blhackslash/HyperCloudCore.jl.git",
    devbranch = "main",
    push_preview = true,
)
