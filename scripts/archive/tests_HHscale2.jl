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

# ---------------- Case 3 Exc, 3 Inh ----------------
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
solve(prob, Tsit5())
println("Equations")
println(length(equations(network)))
bench_build = @benchmark build_network($connections, $neurons) samples=5
println("\n--- Network Build (3 Exc, 3 Inh) ---")
println(bench_build)
println("Mean build time: ", mean(bench_build.times) / 1e9, " seconds")
network = build_network(connections, neurons)
bench_ODE = @benchmark ODEProblem(network, Pair[], (0.0, 500.0)) samples=5
println("\n--- ODEProblem build (3 Exc, 3 Inh) ---")
println(bench_ODE)
println("Mean build time: ", mean(bench_ODE.times) / 1e9, " seconds")
prob = ODEProblem(network, Pair[], (0.0, 500.0))
bench_solve = @benchmark solve($prob, Tsit5()) samples=5
println("\n--- ODE Solve (3 Exc, 3 Inh) ---")
println(bench_solve)
println("Mean solve time: ", mean(bench_solve.times) / 1e9, " seconds")

# ---------------- Case 5 Exc, 5 Inh ----------------
@named e1 = TimeVaryingFunction(f = t -> exp(sin(t)))
@named e2 = TimeVaryingFunction(f = t -> exp(sin(t)))
@named e3 = TimeVaryingFunction(f = t -> exp(sin(t)))
@named e4 = TimeVaryingFunction(f = t -> exp(sin(t)))
@named e5 = TimeVaryingFunction(f = t -> exp(sin(t)))
neurons = [
    build_HH(e1; name = :E1),
    build_HH(e2; name = :E2),
    build_HH(e3; name = :E3),
    build_HH(e4; name = :E4),
    build_HH(e5; name = :E5),
    build_HH(; name = :I1),
    build_HH(; name = :I2),
    build_HH(; name = :I3),
    build_HH(; name = :I4),
    build_HH(; name = :I5),
]
connections = Dict(
    (1,6) => [(type = :Exc, weight = 3rand())],
    (2,7) => [(type = :Exc, weight = 3rand())],
    (3,8) => [(type = :Exc, weight = 3rand())],
    (4,9) => [(type = :Exc, weight = 3rand())],
    (5,10) => [(type = :Exc, weight = 3rand())],
    (6,2) => [(type = :Inh, weight = 20.0)],
    (6,3) => [(type = :Inh, weight = 20.0)],
    (6,4) => [(type = :Inh, weight = 20.0)],
    (6,5) => [(type = :Inh, weight = 20.0)],
    (7,1) => [(type = :Inh, weight = 20.0)],
    (7,3) => [(type = :Inh, weight = 20.0)],
    (7,4) => [(type = :Inh, weight = 20.0)],
    (7,5) => [(type = :Inh, weight = 20.0)],
    (8,1) => [(type = :Inh, weight = 20.0)],
    (8,2) => [(type = :Inh, weight = 20.0)],
    (8,4) => [(type = :Inh, weight = 20.0)],
    (8,5) => [(type = :Inh, weight = 20.0)],
    (9,1) => [(type = :Inh, weight = 20.0)],
    (9,2) => [(type = :Inh, weight = 20.0)],
    (9,3) => [(type = :Inh, weight = 20.0)],
    (9,5) => [(type = :Inh, weight = 20.0)],
    (10,1) => [(type = :Inh, weight = 20.0)],
    (10,2) => [(type = :Inh, weight = 20.0)],
    (10,3) => [(type = :Inh, weight = 20.0)],
    (10,4) => [(type = :Inh, weight = 20.0)],
)
network = build_network(connections, neurons)
prob = ODEProblem(network, Pair[], (0.0, 500.0))
solve(prob, Tsit5())
println("Equations")
println(length(equations(network)))
bench_build = @benchmark build_network($connections, $neurons) samples=5
println("\n--- Network Build (5 Exc, 5 Inh) ---")
println(bench_build)
println("Mean build time: ", mean(bench_build.times) / 1e9, " seconds")
network = build_network(connections, neurons)
bench_ODE = @benchmark ODEProblem(network, Pair[], (0.0, 500.0)) samples=5
println("\n--- ODEProblem build (5 Exc, 5 Inh) ---")
println(bench_ODE)
println("Mean build time: ", mean(bench_ODE.times) / 1e9, " seconds")
prob = ODEProblem(network, Pair[], (0.0, 500.0))
bench_solve = @benchmark solve($prob, Tsit5()) samples=5
println("\n--- ODE Solve (5 Exc, 5 Inh) ---")
println(bench_solve)
println("Mean solve time: ", mean(bench_solve.times) / 1e9, " seconds")

# ---------------- Case 2 Exc, 2 Inh ----------------

@named e1 = TimeVaryingFunction(f = t -> exp(sin(t)))
@named e2 = TimeVaryingFunction(f = t -> exp(sin(t)))
neurons = [
    build_HH(e1; name = :E1),
    build_HH(e2; name = :E2),
    build_HH(; name = :I1),
    build_HH(; name = :I2),
]
connections = Dict(
    (1,3) => [(type = :Exc, weight = 3rand())],
    (2,4) => [(type = :Exc, weight = 3rand())],
    (3,2) => [(type = :Inh, weight = 20.0)],
    (4,1) => [(type = :Inh, weight = 20.0)],
)
network = build_network(connections, neurons)
prob = ODEProblem(network, Pair[], (0.0, 500.0))
solve(prob, Tsit5())
println("Equations")
println(length(equations(network)))
bench_build = @benchmark build_network($connections, $neurons) samples=5
println("\n--- Network Build (2 Exc, 2 Inh) ---")
println(bench_build)
println("Mean build time: ", mean(bench_build.times) / 1e9, " seconds")
network = build_network(connections, neurons)
bench_ODE = @benchmark ODEProblem(network, Pair[], (0.0, 500.0)) samples=5
println("\n--- ODEProblem build (2 Exc, 2 Inh) ---")
println(bench_ODE)
println("Mean build time: ", mean(bench_ODE.times) / 1e9, " seconds")
prob = ODEProblem(network, Pair[], (0.0, 500.0))
bench_solve = @benchmark solve($prob, Tsit5()) samples=5
println("\n--- ODE Solve (2 Exc, 2 Inh) ---")
println(bench_solve)
println("Mean solve time: ", mean(bench_solve.times) / 1e9, " seconds")

# ---------------- Case 7 Exc, 7 Inh ----------------

@named e1 = TimeVaryingFunction(f = t -> exp(sin(t)))
@named e2 = TimeVaryingFunction(f = t -> exp(sin(t)))
@named e3 = TimeVaryingFunction(f = t -> exp(sin(t)))
@named e4 = TimeVaryingFunction(f = t -> exp(sin(t)))
@named e5 = TimeVaryingFunction(f = t -> exp(sin(t)))
@named e6 = TimeVaryingFunction(f = t -> exp(sin(t)))
@named e7 = TimeVaryingFunction(f = t -> exp(sin(t)))
neurons = [
    build_HH(e1; name = :E1),
    build_HH(e2; name = :E2),
    build_HH(e3; name = :E3),
    build_HH(e4; name = :E4),
    build_HH(e5; name = :E5),
    build_HH(e6; name = :E6),
    build_HH(e7; name = :E7),
    build_HH(; name = :I1),
    build_HH(; name = :I2),
    build_HH(; name = :I3),
    build_HH(; name = :I4),
    build_HH(; name = :I5),
    build_HH(; name = :I6),
    build_HH(; name = :I7),
]
connections = Dict(
    (1,8) => [(type = :Exc, weight = 3rand())],
    (2,9) => [(type = :Exc, weight = 3rand())],
    (3,10) => [(type = :Exc, weight = 3rand())],
    (4,11) => [(type = :Exc, weight = 3rand())],
    (5,12) => [(type = :Exc, weight = 3rand())],
    (6,13) => [(type = :Exc, weight = 3rand())],
    (7,14) => [(type = :Exc, weight = 3rand())],
    (8,2) => [(type = :Inh, weight = 20.0)],
    (8,3) => [(type = :Inh, weight = 20.0)],
    (8,4) => [(type = :Inh, weight = 20.0)],
    (8,5) => [(type = :Inh, weight = 20.0)],
    (8,6) => [(type = :Inh, weight = 20.0)],
    (8,7) => [(type = :Inh, weight = 20.0)],
    (9,1) => [(type = :Inh, weight = 20.0)],
    (9,3) => [(type = :Inh, weight = 20.0)],
    (9,4) => [(type = :Inh, weight = 20.0)],
    (9,5) => [(type = :Inh, weight = 20.0)],
    (9,6) => [(type = :Inh, weight = 20.0)],
    (9,7) => [(type = :Inh, weight = 20.0)],
    (10,1) => [(type = :Inh, weight = 20.0)],
    (10,2) => [(type = :Inh, weight = 20.0)],
    (10,4) => [(type = :Inh, weight = 20.0)],
    (10,5) => [(type = :Inh, weight = 20.0)],
    (10,6) => [(type = :Inh, weight = 20.0)],
    (10,7) => [(type = :Inh, weight = 20.0)],
    (11,1) => [(type = :Inh, weight = 20.0)],
    (11,2) => [(type = :Inh, weight = 20.0)],
    (11,3) => [(type = :Inh, weight = 20.0)],
    (11,5) => [(type = :Inh, weight = 20.0)],
    (11,6) => [(type = :Inh, weight = 20.0)],
    (11,7) => [(type = :Inh, weight = 20.0)],
    (12,1) => [(type = :Inh, weight = 20.0)],
    (12,2) => [(type = :Inh, weight = 20.0)],
    (12,3) => [(type = :Inh, weight = 20.0)],
    (12,4) => [(type = :Inh, weight = 20.0)],
    (12,6) => [(type = :Inh, weight = 20.0)],
    (12,7) => [(type = :Inh, weight = 20.0)],
    (13,1) => [(type = :Inh, weight = 20.0)],
    (13,2) => [(type = :Inh, weight = 20.0)],
    (13,3) => [(type = :Inh, weight = 20.0)],
    (13,4) => [(type = :Inh, weight = 20.0)],
    (13,5) => [(type = :Inh, weight = 20.0)],
    (13,7) => [(type = :Inh, weight = 20.0)],
    (14,1) => [(type = :Inh, weight = 20.0)],
    (14,2) => [(type = :Inh, weight = 20.0)],
    (14,3) => [(type = :Inh, weight = 20.0)],
    (14,4) => [(type = :Inh, weight = 20.0)],
    (14,5) => [(type = :Inh, weight = 20.0)],
    (14,6) => [(type = :Inh, weight = 20.0)],
)
network = build_network(connections, neurons)
prob = ODEProblem(network, Pair[], (0.0, 500.0))
solve(prob, Tsit5())
println("Equations")
println(length(equations(network)))
bench_build = @benchmark build_network($connections, $neurons) samples=5
println("\n--- Network Build (7 Exc, 7 Inh) ---")
println(bench_build)
println("Mean build time: ", mean(bench_build.times) / 1e9, " seconds")
network = build_network(connections, neurons)
bench_ODE = @benchmark ODEProblem(network, Pair[], (0.0, 500.0)) samples=5
println("\n--- ODEProblem build (7 Exc, 7 Inh) ---")
println(bench_ODE)
println("Mean build time: ", mean(bench_ODE.times) / 1e9, " seconds")
prob = ODEProblem(network, Pair[], (0.0, 500.0))
bench_solve = @benchmark solve($prob, Tsit5()) samples=5
println("\n--- ODE Solve (7 Exc, 7 Inh) ---")
println(bench_solve)
println("Mean solve time: ", mean(bench_solve.times) / 1e9, " seconds")

# ---------------- Case 6 Exc, 6 Inh ----------------

@named e1 = TimeVaryingFunction(f = t -> exp(sin(t)))
@named e2 = TimeVaryingFunction(f = t -> exp(sin(t)))
@named e3 = TimeVaryingFunction(f = t -> exp(sin(t)))
@named e4 = TimeVaryingFunction(f = t -> exp(sin(t)))
@named e5 = TimeVaryingFunction(f = t -> exp(sin(t)))
@named e6 = TimeVaryingFunction(f = t -> exp(sin(t)))
@named e7 = TimeVaryingFunction(f = t -> exp(sin(t)))
@named e8 = TimeVaryingFunction(f = t -> exp(sin(t)))
neurons = [
    build_HH(e1; name = :E1),
    build_HH(e2; name = :E2),
    build_HH(e3; name = :E3),
    build_HH(e4; name = :E4),
    build_HH(e5; name = :E5),
    build_HH(e6; name = :E6),
    build_HH(; name = :I1),
    build_HH(; name = :I2),
    build_HH(; name = :I3),
    build_HH(; name = :I4),
    build_HH(; name = :I5),
    build_HH(; name = :I6)
]
connections = Dict(
    (1,7) => [(type = :Exc, weight = 3rand())],
    (2,8) => [(type = :Exc, weight = 3rand())],
    (3,9) => [(type = :Exc, weight = 3rand())],
    (4,10) => [(type = :Exc, weight = 3rand())],
    (5,11) => [(type = :Exc, weight = 3rand())],
    (6,12) => [(type = :Exc, weight = 3rand())],
    (7,2) => [(type = :Inh, weight = 20.0)],
    (7,3) => [(type = :Inh, weight = 20.0)],
    (7,4) => [(type = :Inh, weight = 20.0)],
    (7,5) => [(type = :Inh, weight = 20.0)],
    (7,6) => [(type = :Inh, weight = 20.0)],
    (8,1) => [(type = :Inh, weight = 20.0)],
    (8,3) => [(type = :Inh, weight = 20.0)],
    (8,4) => [(type = :Inh, weight = 20.0)],
    (8,5) => [(type = :Inh, weight = 20.0)],
    (8,6) => [(type = :Inh, weight = 20.0)],
    (9,1) => [(type = :Inh, weight = 20.0)],
    (9,2) => [(type = :Inh, weight = 20.0)],
    (9,4) => [(type = :Inh, weight = 20.0)],
    (9,5) => [(type = :Inh, weight = 20.0)],
    (9,6) => [(type = :Inh, weight = 20.0)],
    (10,1) => [(type = :Inh, weight = 20.0)],
    (10,2) => [(type = :Inh, weight = 20.0)],
    (10,3) => [(type = :Inh, weight = 20.0)],
    (10,5) => [(type = :Inh, weight = 20.0)],
    (10,6) => [(type = :Inh, weight = 20.0)],
    (11,1) => [(type = :Inh, weight = 20.0)],
    (11,2) => [(type = :Inh, weight = 20.0)],
    (11,3) => [(type = :Inh, weight = 20.0)],
    (11,4) => [(type = :Inh, weight = 20.0)],
    (11,6) => [(type = :Inh, weight = 20.0)]
)
network = build_network(connections, neurons)
prob = ODEProblem(network, Pair[], (0.0, 500.0))
solve(prob, Tsit5())
println("Equations")
println(length(equations(network)))
bench_build = @benchmark build_network($connections, $neurons) samples=5
println("\n--- Network Build (6 Exc, 6 Inh) ---")
println(bench_build)
println("Mean build time: ", mean(bench_build.times) / 1e9, " seconds")
network = build_network(connections, neurons)
bench_ODE = @benchmark ODEProblem(network, Pair[], (0.0, 500.0)) samples=5
println("\n--- ODEProblem build (6 Exc, 6 Inh) ---")
println(bench_ODE)
println("Mean build time: ", mean(bench_ODE.times) / 1e9, " seconds")
prob = ODEProblem(network, Pair[], (0.0, 500.0))
bench_solve = @benchmark solve($prob, Tsit5()) samples=5
println("\n--- ODE Solve (6 Exc, 6 Inh) ---")
println(bench_solve)
println("Mean solve time: ", mean(bench_solve.times) / 1e9, " seconds")

