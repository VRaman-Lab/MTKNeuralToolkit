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
neurons = [build_HH(inp; name=:HH), build_HH(;name=:Liu)]
connections = Dict(
    (1, 2) => [(type=:Exc, weight=1.0)],
    (2, 1) => [(type=:Inh, weight=100.0)]
)
println("building_network")
network = build_network(connections, neurons)

println("building ODE_Problem")
prob = ODEProblem(network, Pair[], (0.0, 10.0) )
println("solvering")
sol = solve(prob, Tsit5());

p = plot(sol, idxs=[network.Liu.Liu.V, network.HH.HH.V])
gui(p)