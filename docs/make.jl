# docs/make.jl
using Documenter
using HyperCloud

makedocs(
    sitename = "HyperCloud.jl",
    modules = [HyperCloud],
    checkdocs = :exports, 
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://blhackslash.github.io/HyperCloud.jl/",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Time Stepper" => [
            "Explicit RK" => "" 
        ],
        "Render Options" => [
            "Lines" => "render/lines.md",
            "Contour" => "render/contour.md",
            "Scatter" => "render/scatter.md",
            "Heatmap" => "render/heatmap.md",
            "Volume" => "render/volume.md",
        ],
        "API Reference" => "api.md",
    ]
)

deploydocs(
    repo = "github.com/blhackslash/PDEStudio.jl.git",
    devbranch = "main",
    push_preview = true,
)