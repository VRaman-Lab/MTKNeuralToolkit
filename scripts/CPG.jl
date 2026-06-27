using MTKNeuralToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named, System
using OrdinaryDiffEq, Plots

# 1. Populating your neural array using the framework's constructors
make_compartment(id) = build_compartment(SpikingCapacitor(C=1.0, name=:soma), []; name=Symbol(:n, id))
neurons = System[make_compartment(i) for i in 1:3]

# 2. Map standard connection factories (Anonymous functions)
# FIX: Use the unified ChemicalSynapse TwoPort component directly!
function make_excitatory_synapse(; g_max=0.4, τ=5.0, v_th=-55.0, w=0.1, E_rev=0.0)
    # Must accept 'name' as a keyword to match the updated build_electrical_network
    return (; name) -> ChemicalSynapse(name=name, g_max=g_max, τ=τ, v_th=v_th, w=w, E_rev=E_rev)
end

connections = [
    (neurons[1], neurons[2], make_excitatory_synapse(), :synapse_1_to_2),
    (neurons[2], neurons[3], make_excitatory_synapse(), :synapse_2_to_3),
    (neurons[3], neurons[1], make_excitatory_synapse(), :synapse_3_to_1)
]

# 3. Setup Native External Driving Stimulus using Blocks (No Callbacks)
@named stim = Blocks.Sine(frequency = 0.05, amplitude = 15.0)
drivers = [
    (neurons[1], stim) # Maps and routes causal block right into Neuron 1's injector
]

# 4. Compile and Run Network via the internal architecture
@named net = build_electrical_network(neurons, connections; drivers=drivers)
net_compiled = mtkcompile(net)

prob = ODEProblem(net_compiled, [], (0.0, 100.0))
sol = solve(prob, Tsit5())

# 5. Plot Results Natively
plot(sol, idxs=[net.n1.V, net.n2.V, net.n3.V],
     title="3-Neuron LIF Network Dynamics",
     label=["Neuron 1 (Driven)" "Neuron 2" "Neuron 3"],
     xlabel="Time (ms)", ylabel="Voltage (mV)", lw=1.5)
