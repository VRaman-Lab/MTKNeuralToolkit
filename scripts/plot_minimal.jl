import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using MTKNeuralToolkit 
import MTKNeuralToolkit
using Plots

@named inp = TimeVaryingFunction(f=t -> (exp(sin(t))))
neurons = [build_HH(inp; name=:HH), build_Liu(;name=:Liu)]
connections = Dict(
    (1, 2) => [(type=:Exc, weight=1.0)]
)
network = build_network(connections, neurons)
prob = ODEProblem(network, Pair[], (0.0, 200.0) )
sol = solve(prob, Tsit5());

plot(sol)

