using MTKNeuralToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq
using Plots

@named soma1 = Capacitor(C = 1.0)
@named soma2 = Capacitor(C = 1.0)

function make_hh_channels(suffix)
    # BEFORE: nagates(name=Symbol(:na_gate_, suffix))
    # NOW: Keep inner component names simple because build_channel hides them behind its own .p and .n pins!
    na = build_channel(nagates(name=:gate), FixedReversal(E = 50.0, name=:batt); name=Symbol(:na_, suffix))
    k  = build_channel(kgates(name=:gate),  FixedReversal(E = -77.0, name=:batt); name=Symbol(:k_, suffix))
    l  = build_channel(lgates(name=:gate),  FixedReversal(E = -54.4, name=:batt); name=Symbol(:l_, suffix))
    return [na, k, l]
end

# Build individual compartments (setting has_synapses=true prevents grounding out the injector)
nrn1 = build_compartment(soma1, make_hh_channels("n1"); name=:nrn1, open_injector=true)
nrn2 = build_compartment(soma2, make_hh_channels("n2"); name=:nrn2, open_injector=false)
neurons = [nrn1, nrn2]

# 2. Define the Connectivity Graph
# Format per tuple: (pre_idx, post_idx, gate, battery, synapse_system_name)
@named syn_gate = EventSynapseGate(g_max = 2.0, τ = 5.0, v_th = -20.0, w = 0.5)
@named syn_batt = FixedReversal(E = 0.0) # Excitatory synapse

connections = [
    (1, 2, syn_gate, syn_batt, :synapse_1_to_2)
]

# 3. Setup External Driving Stimulus for Neuron 1
@named stim = Blocks.Sine(frequency = 0.05, amplitude = 15.0)
drivers = [
    (1, stim) # Connects stim block to src injector, and wraps into Neuron 1
]

# 4. Compile and Run Network
@named net = build_electrical_network(neurons, connections; drivers=drivers)
net_compiled = mtkcompile(net)

prob = ODEProblem(net_compiled, [], (0.0, 100.0))
sol = solve(prob, Rosenbrock23())

# 5. Plot Results
plot(sol, idxs=[net.nrn1.V, net.nrn2.V], 
     title="Explicit Circuit Network Dynamics", 
     label=["Neuron 1 (Pre)" "Neuron 2 (Post)"],
     xlabel="Time (ms)", ylabel="Voltage (mV)")
