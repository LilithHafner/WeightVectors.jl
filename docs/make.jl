using WeightVectors
using Documenter

DocMeta.setdocmeta!(WeightVectors, :DocTestSetup, :(using WeightVectors); recursive=true)

makedocs(;
    modules=[WeightVectors],
    authors="Lilith Orion Hafner <lilithhafner@gmail.com> and contributors",
    sitename="WeightVectors.jl",
    format=Documenter.HTML(;
        canonical="https://LilithHafner.github.io/WeightVectors.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md"
    ],
)

deploydocs(;
    repo="github.com/LilithHafner/WeightVectors.jl",
    devbranch="main",
)
