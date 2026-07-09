using MTKNeuralToolkit
using Documenter
using Literate

DocMeta.setdocmeta!(MTKNeuralToolkit, :DocTestSetup, :(using MTKNeuralToolkit); recursive=true)

# ---------------------------------------------------------
# 1. Process Examples with Literate.jl
# ---------------------------------------------------------
examples_in = joinpath(@__DIR__, "examples")   # Looks in the root examples/ folder
examples_out = joinpath(@__DIR__, "src", "examples") # Outputs .md files here for Documenter
mkpath(examples_out) # Create the folder if it doesn't exist

example_pages = Pair{String, String}[]

# Loop through all files in the examples folder (sorted to guarantee 01, 02, 03...)
for script in sort(readdir(examples_in))
    if endswith(script, ".jl")
        # Generate the markdown file
        Literate.markdown(
            joinpath(examples_in, script),
            examples_out;
            documenter=true
        )
        
        # Create a clean title for the sidebar
        name = first(splitext(script))
        
        # Optional: Improve title formatting to avoid "01 Hh" instead of "01 HH"
        # We can simply replace underscores with spaces and keep the original case
        title = replace(name, "_" => " ")
        
        # Add to the pages list
        push!(example_pages, title => joinpath("examples", name * ".md"))
    end
end

# ---------------------------------------------------------
# 2. Build the Documentation
# ---------------------------------------------------------
makedocs(;
    modules=[MTKNeuralToolkit],
    authors="Dhruva V. Raman, Elouan Simonneau, Ella Bennison",
    sitename="MTKNeuralToolkit.jl",
    format=Documenter.HTML(;
        canonical="https://Dhruva2.github.io/MTKNeuralToolkit.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => example_pages, # Dynamically populated from the loop above
    ],
)

deploydocs(;
    repo="github.com/VRaman-Lab/MTKNeuralToolkit.jl",
    devbranch="main",
)

