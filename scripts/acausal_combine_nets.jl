using MTKNeuralToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named, System, SymbolicT
using OrdinaryDiffEq
using Plots
using ModelingToolkit

# =============================================================================
# 2. Mixed Network Test Script
# =============================================================================

# 1. Build a population of 3 Hodgkin-Huxley style neurons
neurons = System[]
for i in 1:3
    @named soma = Capacitor(C = 1.0)

    channels = System[]
    push!(channels, build_channel(nagates(name=:gate), FixedReversal(E = 50.0, name=:batt); name=Symbol(:na_, i)))
    push!(channels, build_channel(kgates(name=:gate),  FixedReversal(E = -77.0, name=:batt); name=Symbol(:k_, i)))
    push!(channels, build_channel(lgates(name=:gate),  FixedReversal(E = -54.4, name=:batt); name=Symbol(:l_, i)))

    nrn = build_compartment(soma, channels; name = Symbol(:nrn_, i))
    push!(neurons, nrn)
end

event_synapse_conn = (
    neurons[1],
    neurons[2],
    (; name) -> ChemicalSynapse(name=name, g_max=0.5, τ=5.0, v_th=-20.0, w=0.1, E_rev=0.0)
)

# Connection B: Acausal Gap Junction (Neuron 2 <-> Neuron 3)
gap_junction_conn = (
    neurons[2],
    neurons[3],
    (; name) -> GapJunction(name=name, R=0.5)
)

# Combine them into a single flat edge list
all_connections = [event_synapse_conn, gap_junction_conn]

# 3. Setup a driving stimulus on the first neuron
@named stim = Blocks.Sine(frequency = 0.05, amplitude = 15.0)
drivers = [(1, stim)] # Target the first neuron by index

# 4. Build the complete mixed network using the explicit acausal builder
println("Building mixed acausal network...")
@named net = build_electrical_network(neurons, all_connections; drivers=drivers)
net_compiled = mtkcompile(net)

# 5. Simulate the network
println("Compiling and solving...")
prob = ODEProblem(net_compiled, Pair[], (0.0, 50.0))
sol = solve(prob, Rosenbrock23())

println("Simulation complete! Check `sol` for voltage traces of nrn_1, nrn_2, and nrn_3.")
plot(sol, idxs=[neurons[1].V, neurons[2].V, neurons[3].V],
            label=["Neuron 1" "Neuron 2" "Neuron 3"], ylabel="Voltage (mV)", xlabel="Time (ms)")
