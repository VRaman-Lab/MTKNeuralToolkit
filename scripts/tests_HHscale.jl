using BenchmarkTools
using ModelingToolkit
using OrdinaryDiffEq
using Statistics
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction
using ModelingToolkit: t_nounits as t, D_nounits as D
import MTKNeuralToolkit.Synapse as Synapse
import MTKNeuralToolkit.HodgkinHuxley as HH
import MTKNeuralToolkit.Liu as Liu
import MTKNeuralToolkit.Types: SYNAPSE_TYPES, NEURON_TYPES, CustomSynapseParams
using MTKNeuralToolkit

@named inp1 = TimeVaryingFunction(f = t -> exp(sin(t)))
@named inp2 = TimeVaryingFunction(f = t -> exp(sin(t)))
@named inp3 = TimeVaryingFunction(f = t -> exp(sin(t)))

neurons = [
    build_HH(inp1; name = :HH1),
    build_HH(inp2; name = :HH2),
    build_HH(inp3; name = :HH3),
    build_HH(; name = :IHH1),
    build_HH(; name = :IHH2),
    build_HH(; name = :IHH3),
]

connections = Dict(
    (1,4) => [(type = :Exc, weight = 3rand())],
    (2,5) => [(type = :Exc, weight = 3rand())],
    (3,6) => [(type = :Exc, weight = 3rand())],
    (4,2) => [(type = :Inh, weight = 20.0)],
    (4,3) => [(type = :Inh, weight = 20.0)],
    (5,1) => [(type = :Inh, weight = 20.0)],
    (5,3) => [(type = :Inh, weight = 20.0)],
    (6,1) => [(type = :Inh, weight = 20.0)],
    (6,2) => [(type = :Inh, weight = 20.0)],
)

network = build_network(connections, neurons)
prob = ODEProblem(network, Pair[], (0.0, 500.0))
println(equations(prob))
solve(prob, TRBDF2())

bench_build = @benchmark build_network($connections, $neurons) samples=15
println("\n--- Network Build ---")
println(bench_build)
println("Mean build time: ", mean(bench_build.times) / 1e9, " seconds")

network = build_network(connections, neurons)

bench_ODE = @benchmark ODEProblem(network, Pair[], (0.0, 500.0)) samples=15
println("\n--- ODEProblem build ---")
println(bench_ODE)
println("Mean build time: ", mean(bench_ODE.times) / 1e9, " seconds")
prob = ODEProblem(network, Pair[], (0.0, 500.0))

network = build_network(connections, neurons)
prob = ODEProblem(network, Pair[], (0.0, 500.0))

bench_solve = @benchmark solve($prob, TRBDF2()) samples=15
println("\n--- ODE Solve ---")
println(bench_solve)
println("Mean solve time: ", mean(bench_solve.times) / 1e9, " seconds")
