using ModelingToolkit
using OrdinaryDiffEq
using OrdinaryDiffEqNonlinearSolve
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using ModelingToolkit: t_nounits as t, D_nounits as D
import MTKNeuralToolkit.Types: SYNAPSE_TYPES, NEURON_TYPES, CustomSynapseParams
using MTKNeuralToolkit 
import MTKNeuralToolkit.IntegrateAndFire as IaF
import MTKNeuralToolkit.HodgkinHuxley as HH
import MTKNeuralToolkit.Synapse as Synapse
import MTKNeuralToolkit.Config as cfg
import MTKNeuralToolkit
#using script_utils.jl
using Plots

@named inp = TimeVaryingFunction(f = t -> ifelse((t > 10) & (t < 20),30.0, 0.0))
num_neurons = 10

neurons = [build_IF(inp;name=:IF1)]

for i in 2:num_neurons
    push!(neurons, build_IF(; name=Symbol("IF$i")))
end 

connections =  Dict{Tuple{Int64,Int64}, Vector{@NamedTuple{type::Symbol, weight::Float64}}}()
weight = 9.0

for i in 2:num_neurons
    num = i - 1
    connections[(num,i)] = [(type=:LIF, weight=weight)]
end

println("Total connections: ", length(connections))

println("Building network")
@time network = build_network(connections, neurons)

build_start = time()

println("Building ODEProblem")
@time prob = ODEProblem(network, Pair[], (0.0, 500.0))
println("Equations: ", length(equations(network)))

build_end = time()

println("Solving")
solve_start = time()
@time sol = solve(prob, Tsit5());
solve_end = time()

println("\n=====================================")
println("Building time     : $(round(build_end - build_start, digits=2)) s")
println("Simulation time   : $(round(solve_end - solve_start, digits=2)) s")
println("=====================================")

plot(sol, idxs=[IF10.IF10.oneport.v])