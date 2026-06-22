using MTKNeuralToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named, System
using OrdinaryDiffEq
using Plots

# =============================================================================
# 1. Build Neurons
# =============================================================================
N = 3
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

# =============================================================================
# 2. Setup Vectorized Synapse Matrices
# =============================================================================
# W[j, i] = weight from neuron j to neuron i
W = fill(0.5, N, N)
for i in 1:N
    W[i, i] = 0.0 # No self-connections
end

tau_mat = fill(5.0, N, N)  # All synapses have tau = 5.0 ms
gmax_mat = fill(0.5, N, N) # All synapses have g_max = 0.5

# =============================================================================
# 3. Instantiate the Vectorized Synapse Component
# =============================================================================
@named exc_synapses = VectorizedAlphaSynapse(
    N = N,
    W = W,
    tau = tau_mat,
    g_max = gmax_mat,
    E_rev = 0.0,
    v_th = -20.0
)

# =============================================================================
# 4. Setup Stimulus Drivers
# =============================================================================
@named stim = Blocks.Sine(frequency = 0.05, amplitude = 15.0)
drivers = [(1, stim)] # Drive the first neuron

# =============================================================================
# 5. Build and Compile the Network
# =============================================================================
@named net = build_vectorized_network(neurons, [exc_synapses]; drivers=drivers)
net_compiled = mtkcompile(net)

# =============================================================================
# 6. Solve and Plot
# =============================================================================
prob = ODEProblem(net_compiled, [], (0.0, 100.0); fully_determined=true)
sol = solve(prob, Rosenbrock23())

plot(sol, idxs=[net.nrn_1.V, net.nrn_2.V, net.nrn_3.V],
     title="Vectorized Network Dynamics (N=$N)",
     label=["Neuron 1" "Neuron 2" "Neuron 3"],
     xlabel="Time (ms)", ylabel="Voltage (mV)",
     lw=2)
