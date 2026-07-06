using MTKNeuralToolkit
using Test

@testset "MTKNeuralToolkit.jl" begin
    # Write your tests here.
end

@testset "Plot Scripts" begin
    # Find all plot*.jl files in scripts directory
    scripts_dir = joinpath(dirname(@__DIR__), "scripts")
    plot_files = filter(f -> startswith(f, "plot") && endswith(f, ".jl"), readdir(scripts_dir))
    
    if isempty(plot_files)
        @warn "No plot*.jl files found in scripts directory"
    else
        println("Found $(length(plot_files)) plot scripts to test")
        
        for script in plot_files
            @testset "Testing $script" begin
                script_path = joinpath(scripts_dir, script)
                
                result = run(`julia $script_path`)
                @test success(result)
            end
        end
    end
end