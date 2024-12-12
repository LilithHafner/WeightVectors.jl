using DynamicDiscreteSamplers
using Documenter

DocMeta.setdocmeta!(DynamicDiscreteSamplers, :DocTestSetup, :(using DynamicDiscreteSamplers); recursive=true)

makedocs(;
    modules=[DynamicDiscreteSamplers],
    authors="Lilith Orion Hafner <lilithhafner@gmail.com> and contributors",
    sitename="DynamicDiscreteSamplers.jl",
    format=Documenter.HTML(;
        canonical="https://LilithHafner.github.io/DynamicDiscreteSamplers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/LilithHafner/DynamicDiscreteSamplers.jl",
    devbranch="main",
)
