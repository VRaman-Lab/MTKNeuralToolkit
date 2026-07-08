using SafeTestsets
using SciMLTesting

run_tests(;
    core = function ()
        @safetestset "Core Channels & Custom Components" begin
            include(joinpath(@__DIR__, "test_channels.jl"))
        end
        @safetestset "Directed Synapses & STDP" begin
            include(joinpath(@__DIR__, "test_synapses.jl"))
        end
        @safetestset "Acausal Couplings & Gap Junctions" begin
            include(joinpath(@__DIR__, "test_couplings.jl"))
        end
        @safetestset "Vectorized & Mixed Topologies" begin
            include(joinpath(@__DIR__, "test_topologies.jl"))
        end
        @safetestset "Calcium Dynamics & Nernst Potentials" begin
            include(joinpath(@__DIR__, "test_calcium.jl"))
        end
        @safetestset "Standard Model Libraries & STG" begin
            include(joinpath(@__DIR__, "test_libraries.jl"))
        end
        return @safetestset "Explicit Imports Compliance" begin
            include(joinpath(@__DIR__, "explicit_imports.jl"))
        end
    end,
    groups = Dict(
        "Opt" => joinpath(@__DIR__, "test_optimization.jl"),
        "ExplicitImports" => joinpath(@__DIR__, "explicit_imports.jl"),
    ),
    qa = (; env = joinpath(@__DIR__, "qa"), body = joinpath(@__DIR__, "qa", "qa.jl")),
    all = ["Core", "Opt"],
)
