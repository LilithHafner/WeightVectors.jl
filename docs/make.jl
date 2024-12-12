using DynamicDiscreteSampler
using Documenter

DocMeta.setdocmeta!(DynamicDiscreteSampler, :DocTestSetup, :(using DynamicDiscreteSampler); recursive=true)

makedocs(;
    modules=[DynamicDiscreteSampler],
    authors="Lilith Orion Hafner <lilithhafner@gmail.com> and contributors",
    sitename="DynamicDiscreteSampler.jl",
    format=Documenter.HTML(;
        canonical="https://LilithHafner.github.io/DynamicDiscreteSampler.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/LilithHafner/DynamicDiscreteSampler.jl",
    devbranch="main",
)
