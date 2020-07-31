using InverseGaussTransform
using Documenter

makedocs(;
    modules=[InverseGaussTransform],
    authors="Patrick Eastham <peastham@math.fsu.edu> and contributors",
    repo="https://github.com/pseastham/InverseGaussTransform.jl/blob/{commit}{path}#L{line}",
    sitename="InverseGaussTransform.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pseastham.github.io/InverseGaussTransform.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/pseastham/InverseGaussTransform.jl",
)
