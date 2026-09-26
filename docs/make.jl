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
        "Base API" => "base_api.md",
        "Advanced API" => "advanced_api.md",
    ]
)

deploydocs(
    repo = "github.com/blhackslash/HyperCloud.jl.git",
    devbranch = "main",
    push_preview = true,
)