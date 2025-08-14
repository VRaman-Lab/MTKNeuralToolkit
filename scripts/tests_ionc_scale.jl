using BenchmarkTools
using ModelingToolkit
using OrdinaryDiffEq
using Statistics
using ModelingToolkitStandardLibrary.Blocks: TimeVaryingFunction 
using MTKNeuralToolkit 
import MTKNeuralToolkit.HodgkinHuxley as HH

@named inp = TimeVaryingFunction(f = t -> exp(sin(t)))

function create_channels(n_na, n_k)
    channels = []
    for i in 1:n_na
        na = build_channel(HH.NaGates(;g=40, E=55), FixedReversal(;E=55); name=Symbol("Na$i"))
        push!(channels, na)
    end
    for i in 1:n_k
        k = build_channel(HH.KGates(;g=35, E=-77), FixedReversal(;E=-77); name=Symbol("K$i"))
        push!(channels, k)
    end
    return channels
end

function benchmark_neuron_scaling(n_na, n_k)
    println("\n=== $n_na Na + $n_k K channels (≈$(n_na*7 + n_k*4 + 1) equations) ===")
    
    # Warm-up compilation
    fn = BasicSoma(; C=1, name=:soma)
    channels = create_channels(n_na, n_k)
    neuron = build_neuron(fn, inp; channels=channels)
    neuron_simplified = structural_simplify(neuron)
    prob = ODEProblem(neuron_simplified, Pair[], (0.0, 100.0))
    solve(prob, TRBDF2())
    println("Equations: ", length(equations(neuron_simplified)))

    # Benchmark Part 1: Neuron build time
    fn = BasicSoma(; C=1, name=:soma)
    channels = create_channels(n_na, n_k)
    bench_build = @benchmark begin
        neuron = build_neuron($fn, $inp; channels=$channels)
        structural_simplify(neuron)
    end samples=15
    
    println("\n--- Neuron Build + Simplify ---")
    println(bench_build)
    println("Mean build time: ", mean(bench_build.times) / 1e9, " seconds")

    # Prepare neuron for ODE benchmarking
    neuron_simplified = structural_simplify(build_neuron(fn, inp; channels=channels))

    # Benchmark Part 2: ODEProblem build time
    bench_ODE = @benchmark ODEProblem($neuron_simplified, Pair[], (0.0, 100.0)) samples=15
    println("\n--- ODEProblem build ---")
    println(bench_ODE)
    println("Mean build time: ", mean(bench_ODE.times) / 1e9, " seconds")

    # Prepare problem for solve benchmarking
    prob = ODEProblem(neuron_simplified, Pair[], (0.0, 100.0))

    # Benchmark Part 3: Solve time
    bench_solve = @benchmark solve($prob, TRBDF2()) samples=15
    println("\n--- ODE Solve ---")
    println(bench_solve)
    println("Mean solve time: ", mean(bench_solve.times) / 1e9, " seconds")
    
    return (mean(bench_build.times), mean(bench_ODE.times), mean(bench_solve.times))
end

# Run scaling benchmarks
configs = [
    (1, 1),   # ~12 equations
    (3, 3),   # ~34 equations  
    (9, 9),   # ~100 equations
    (27, 27),  # ~298 equations
    (71, 71),  # ~298 equations
    (213, 213),  # ~298 equations
    (426, 426)
]

results = []
for (n_na, n_k) in configs
    times = benchmark_neuron_scaling(n_na, n_k)
    push!(results, (n_na + n_k, times...))
end

println("\n=== SCALING SUMMARY ===")
println("Channels\tBuild(s)\tODE(s)\t\tSolve(s)")
for (n_channels, build_time, ode_time, solve_time) in results
    println("$n_channels\t\t$(build_time/1e9:.4f)\t\t$(ode_time/1e9:.4f)\t\t$(solve_time/1e9:.4f)")
end