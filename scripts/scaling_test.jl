using MTKNeuralToolkit
using ModelingToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named, System, t_nounits as t, SymbolicT, Equation
using OrdinaryDiffEq
using Plots

# =============================================================================
# 1. Scaling Benchmark Harness
# =============================================================================
function run_scaling_benchmark(sizes::Vector{Int})
    compile_times = Float64[]
    solve_times = Float64[]

    println("Starting MTKNeuralToolkit Scaling Benchmark...")
    println("--------------------------------------------------")

    for N in sizes
        println("Testing Network Size: N = $N neurons...")

        # 1. Build a population of Hodgkin-Huxley neurons
        neurons = System[]
        for i in 1:N
            @named soma = Capacitor(C = 1.0)

            # Create unique channel names per neuron to avoid namespace collisions
            channels = System[]
            push!(channels, build_channel(nagates(name=:gate), FixedReversal(E = 50.0, name=:batt); name=Symbol(:na_, i)))
            push!(channels, build_channel(kgates(name=:gate),  FixedReversal(E = -77.0, name=:batt); name=Symbol(:k_, i)))
            push!(channels, build_channel(lgates(name=:gate),  FixedReversal(E = -54.4, name=:batt); name=Symbol(:l_, i)))

            nrn = build_compartment(soma, channels; name = Symbol(:nrn_, i))
            push!(neurons, nrn)
        end

        # 2. Build an all-to-all connection edge list (excluding self-connections)
        # Using a generic Tuple[] vector prevents type-mismatch errors during push!
        connections = Tuple[]
        for i in 1:N, j in 1:N
            i == j && continue
            push!(connections, (
                neurons[i],
                neurons[j],
                (; name) -> AlphaSynapse(; name=name, g_max = 0.5 / N, τ = 3.0, E_rev = 0.0, v_th = -20.0, w = 0.1)
            ))
        end

        # 3. Attach a driving stimulus to the first neuron
        @named stim = Blocks.Sine(frequency = 0.05, amplitude = 15.0)
        drivers = [(neurons[1], stim)]

        # 4. Benchmark the Compilation Phase
        t_compile = @elapsed begin
            # The refactored engine builds an isolated, flattened synapse block
            synapse_block = build_factored_synapse_network(neurons, connections; drivers=drivers, name=Symbol(:synapse_net_, N))

            # Top-level assembly to connect the neurons to the synapse block's IO arrays
            net_eqs = Equation[]
            all_systems = System[]
            append!(all_systems, neurons)
            push!(all_systems, synapse_block)

            for i in 1:N
                # Map neuron membrane voltage to the synapse block input array
                push!(net_eqs, synapse_block.V_in.u[i] ~ neurons[i].V)
                # Map synapse block current output array back into the neuron's injector
                push!(net_eqs, neurons[i].injector.I.u ~ synapse_block.I_out.u[i])
            end

            @named net = System(net_eqs, t, SymbolicT[], SymbolicT[]; systems=all_systems)
            net_compiled = mtkcompile(net)
        end
        push!(compile_times, t_compile)
        println("   > Compilation Time: $(round(t_compile, digits=2)) seconds")

        # 5. Benchmark the Solving Phase (simulating 0 to 50 ms)
        prob = ODEProblem(net_compiled, Pair[], (0.0, 50.0))
        t_solve = @elapsed begin
            sol = solve(prob, Rosenbrock23()) # Rosenbrock23 handles the stiff HH gating equations smoothly
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
# Start small to test the curve safely (e.g., populations of 2, 5, 10 neurons)
sizes = [2, 5, 10]
network_sizes, compile_trends, solve_trends = run_scaling_benchmark(sizes)

# Generate the benchmark visualization
plot(network_sizes, [compile_trends, solve_trends],
     line = :rocket,
     marker = :circle,
     lw = 2,
     title = "MTK Neural Network Scaling Bottlenecks",
     label = ["MTK Compile Time" "ODE Solve Time"],
     xlabel = "Network Size (Number of Neurons)",
     ylabel = "Time (seconds)",
     legend = :topleft)
