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

neurons = [build_modular_HH(inp;name=:LIF), build_modular_HH(; name=:HH)]
connections = Dict(
    (1,2) => [(type=:Exc, weight=5.0)]
)
network = build_network(connections, neurons)
#network = build_synapse(neurons[1], neurons[2], :Exc, 3.0; name=:minimal_network)

println("building ODE_Problem")
prob = ODEProblem(network, Pair[], (0.0, 10.0) )
println("solvering")
sol = solve(prob, Tsit5());

p = plot(sol, idxs=[network.LIF.LIF.V, network.HH.HH.V])
gui(p)