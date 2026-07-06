using MTKNeuralToolkit
using Documenter

DocMeta.setdocmeta!(MTKNeuralToolkit, :DocTestSetup, :(using MTKNeuralToolkit); recursive=true)

makedocs(;
    modules=[MTKNeuralToolkit],
    authors="Dhruva V. Raman, Elouan Simonneau",
    sitename="MTKNeuralToolkit.jl",
    format=Documenter.HTML(;
        canonical="https://Dhruva2.github.io/MTKNeuralToolkit.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Dhruva2/MTKNeuralToolkit.jl",
    devbranch="main",
)
