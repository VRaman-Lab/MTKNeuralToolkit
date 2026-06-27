using MTKNeuralToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named, System
using OrdinaryDiffEq
using Plots

# =============================================================================
# 1. Helper Functions for Building Neurons
# =============================================================================
function build_elegant_neurons(N)
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
    return neurons
end

function build_inlined_neurons(N)
    neurons = System[]
    for i in 1:N
        # Uses the fixed InlinedHHNeuron from the package
        nrn = InlinedHHNeuron(name = Symbol(:nrn_, i))
        push!(neurons, nrn)
    end
    return neurons
end

# =============================================================================
# 2. Scaling Benchmark Harness
# =============================================================================
function run_comparison_benchmark(sizes::Vector{Int})
    # Arrays to store results
    elegant_compile = Float64[]
    elegant_solve = Float64[]
    inlined_compile = Float64[]
    inlined_solve = Float64[]
    vec_compile = Float64[]
    vec_solve = Float64[]

    println("Starting MTK Scaling Benchmark: 3-Way Comparison...")
    println("--------------------------------------------------")

    for N in sizes
        println("Testing Network Size: N = $N neurons...")

        # Setup Vectorized Synapse Matrices (All-to-all, no self-connections)
        W = fill(0.1, N, N)
        for i in 1:N
            W[i, i] = 0.0
        end
        tau_mat = fill(5.0, N, N)
        gmax_mat = fill(0.5 / N, N, N) # Scale g_max to prevent exploding activity

        # Setup Driving Stimulus
        @named stim = Blocks.Sine(frequency = 0.05, amplitude = 15.0)
        drivers = [(1, stim)]

        # ---------------------------------------------------------
        # 1. BENCHMARK ELEGANT (COMPOSITE) VERSION
        # ---------------------------------------------------------
        println("   -> Building Elegant (Composite) Network...")
        neurons_elegant = build_elegant_neurons(N)
        @named exc_syn_elegant = VectorizedAlphaSynapse(
            N = N, W = W, tau = tau_mat, g_max = gmax_mat,
            E_rev = 0.0, v_th = -20.0
        )

        t_c = @elapsed begin
            @named net = build_vectorized_network(neurons_elegant, [exc_syn_elegant]; drivers=drivers)
            net_compiled = mtkcompile(net)
        end
        push!(elegant_compile, t_c)

        prob = ODEProblem(net_compiled, [], (0.0, 50.0), jac=false)
        t_s = @elapsed begin
            sol = solve(prob, Tsit5(); reltol=1e-3, abstol=1e-3)
        end
        push!(elegant_solve, t_s)
        println("      Elegant Compile: $(round(t_c, digits=2))s | Solve: $(round(t_s, digits=2))s")

        # ---------------------------------------------------------
        # 2. BENCHMARK INLINED VERSION
        # ---------------------------------------------------------
        println("   -> Building Inlined Network...")
        neurons_inlined = build_inlined_neurons(N)
        @named exc_syn_inlined = VectorizedAlphaSynapse(
            N = N, W = W, tau = tau_mat, g_max = gmax_mat,
            E_rev = 0.0, v_th = -20.0
        )

        t_c = @elapsed begin
            @named net = build_vectorized_network(neurons_inlined, [exc_syn_inlined]; drivers=drivers)
            net_compiled = mtkcompile(net)
        end
        push!(inlined_compile, t_c)

        prob = ODEProblem(net_compiled, [], (0.0, 50.0), jac=false)
        t_s = @elapsed begin
            sol = solve(prob, Tsit5(); reltol=1e-3, abstol=1e-3)
        end
        push!(inlined_solve, t_s)
        println("      Inlined Compile: $(round(t_c, digits=2))s | Solve: $(round(t_s, digits=2))s")

        # ---------------------------------------------------------
        # 3. BENCHMARK FULLY VECTORIZED VERSION
        # ---------------------------------------------------------
        println("   -> Building Fully Vectorized Network...")
        @named vec_neurons = VectorizedHHNeuron(N=N)
        @named exc_syn_vec = VectorizedAlphaSynapse(
            N = N, W = W, tau = tau_mat, g_max = gmax_mat,
            E_rev = 0.0, v_th = -20.0
        )

        t_c = @elapsed begin
            @named net = build_fully_vectorized_network(vec_neurons, [exc_syn_vec]; drivers=drivers)
            net_compiled = mtkcompile(net)
        end
        push!(vec_compile, t_c)

        prob = ODEProblem(net_compiled, [], (0.0, 50.0), jac=false)
        t_s = @elapsed begin
            sol = solve(prob, Tsit5(); reltol=1e-3, abstol=1e-3)
        end
        push!(vec_solve, t_s)
        println("      Vectorized Compile: $(round(t_c, digits=2))s | Solve: $(round(t_s, digits=2))s")
        println("--------------------------------------------------")
    end

    return sizes, elegant_compile, elegant_solve, inlined_compile, inlined_solve, vec_compile, vec_solve
end

# =============================================================================
# 3. Run and Plot Results
# =============================================================================
sizes = [2, 5, 10, 20]
network_sizes, elegant_comp, elegant_sol, inlined_comp, inlined_sol, vec_comp, vec_sol = run_comparison_benchmark(sizes)

# Generate the benchmark visualization
plot(network_sizes, [elegant_comp, elegant_sol, inlined_comp, inlined_sol, vec_comp, vec_sol],
     marker = :circle,
     lw = 2,
     title = "MTK Neural Network Scaling: 3-Way Comparison",
     label = ["Elegant Compile" "Elegant Solve" "Inlined Compile" "Inlined Solve" "Vectorized Compile" "Vectorized Solve"],
     xlabel = "Network Size (Number of Neurons)",
     ylabel = "Time (seconds)",
     legend = :topleft,
     palette = :tab10)
