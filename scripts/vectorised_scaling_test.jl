using MTKNeuralToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named, System
using OrdinaryDiffEq
using Plots

# =============================================================================
# 1. Scaling Benchmark Harness
# =============================================================================
function run_vectorized_benchmark(sizes::Vector{Int})
    compile_times = Float64[]
    solve_times = Float64[]

    println("Starting Vectorized MTK Scaling Benchmark...")
    println("--------------------------------------------------")

    for N in sizes
        println("Testing Network Size: N = $N neurons...")

        # 1. Build a population of Hodgkin-Huxley neurons
        neurons = System[]
        for i in 1:N
            @named soma = Capacitor(C = 1.0)

            channels = System[]
            push!(channels, build_channel(nagates(name=:gate), FixedReversal(E = 50.0, name=:batt); name=Symbol(:na_, i)))
            push!(channels, build_channel(kgates(name=:gate),  FixedReversal(E = -77.0, name=:batt); name=Symbol(:k_, i)))
            push!(channels, build_channel(lgates(name=:gate),  FixedReversal(E = -54.4, name=:batt); name=Symbol(:l_, i)))

            nrn = build_compartment(soma, channels; name = Symbol(:nrn_, i))
            push!(neurons, nrn)
        end

        # 2. Setup Vectorized Synapse Matrices (All-to-all, no self-connections)
        W = fill(0.1, N, N)
        for i in 1:N
            W[i, i] = 0.0
        end
        tau_mat = fill(5.0, N, N)
        gmax_mat = fill(0.5 / N, N, N) # Scale g_max to prevent exploding activity

        # 3. Instantiate the single Vectorized Synapse Component
        @named exc_synapses = VectorizedAlphaSynapse(
            N = N, W = W, tau = tau_mat, g_max = gmax_mat,
            E_rev = 0.0, v_th = -20.0
        )

        # 4. Setup Driving Stimulus
        @named stim = Blocks.Sine(frequency = 0.05, amplitude = 15.0)
        drivers = [(1, stim)]

        # 5. Benchmark the Compilation Phase
        t_compile = @elapsed begin
            @named net = build_vectorized_network(neurons, [exc_synapses]; drivers=drivers)
            net_compiled = mtkcompile(net)
        end
        push!(compile_times, t_compile)
        println("   > Compilation Time: $(round(t_compile, digits=2)) seconds")

        # 6. Benchmark the Solving Phase (simulating 0 to 50 ms)
        prob = ODEProblem(net_compiled, [], (0.0, 50.0);)
        t_solve = @elapsed begin
            sol = solve(prob, Tsit5())
        end
        push!(solve_times, t_solve)
        println("   > Simulation Time:  $(round(t_solve, digits=2)) seconds")
        println("--------------------------------------------------")
    end

    return sizes, compile_times, solve_times
end

# =============================================================================
# 2. Run and Plot Results
# =============================================================================
sizes = [2, 5, 10, 20,50]
network_sizes, compile_trends, solve_trends = run_vectorized_benchmark(sizes)

# Generate the benchmark visualization
plot(network_sizes, [compile_trends, solve_trends],
     line = :rocket,
     marker = :circle,
     lw = 2,
     title = "Vectorized MTK Neural Network Scaling",
     label = ["MTK Compile Time" "ODE Solve Time"],
     xlabel = "Network Size (Number of Neurons)",
     ylabel = "Time (seconds)",
     legend = :topleft)
