
using MTKNeuralToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named, t_nounits as t, D_nounits as D, Pre
using OrdinaryDiffEq
using Plots



# =============================================================================
# 2. Build N =============================================================================
@named soma1 = Capacitor(C = 1.0)
@named soma2 = Capacitor(C = 1.0)

function make_hh_channels(suffix)
    # Fixed the typo :gaate -> :gate and added unique suffixes
    na = build_channel(nagates(name=:gate), FixedReversal(E = 50.0, name=:batt); name=Symbol(:na_, suffix))
    k  = build_channel(kgates(name=:gate),  FixedReversal(E = -77.0, name=:batt); name=Symbol(:k_, suffix))
    l  = build_channel(lgates(name=:gate),  FixedReversal(E = -54.4, name=:batt); name=Symbol(:l_, suffix))
    return [na, k, l]
end

nrn1 = build_compartment(soma1, make_hh_channels("n1"); name=:nrn1)
nrn2 = build_compartment(soma2, make_hh_channels("n2"); name=:nrn2)
neurons = [nrn1, nrn2]

# =============================================================================
# 3. Define Connections
# =============================================================================
# We pass the generator directly! No wrapper functions needed because it
# natively exposes p1, n1, p2, n2 via TwoPort.
connections = [
    # (pre_idx, post_idx, synapse_blueprint, unique_system_name)
    (nrn1, nrn2, (; name) -> ChemicalSynapse(name=name, g_max=2.0), :synapse_1_to_2)
]

# =============================================================================
# 4. Setup External Driving Stimulus for Neuron 1
# =============================================================================
@named stim = Blocks.Sine(frequency = 0.05, amplitude = 15.0)
drivers = [
    (nrn1, stim) # Connects stim block to injector, and wraps into Neuron 1
]

# =============================================================================
# 5. Compile and Run Network
# =============================================================================
@named net = build_electrical_network(neurons, connections; drivers=drivers)
net_compiled = mtkcompile(net)

prob = ODEProblem(net_compiled, [], (0.0, 100.0))
sol = solve(prob, Rosenbrock23())

# =============================================================================
# 6. Plot Results
# =============================================================================
plot(sol, idxs=[net.nrn1.V, net.nrn2.V],
     title="Explicit Circuit Network Dynamics",
     label=["Neuron 1 (Pre)" "Neuron 2 (Post)"],
     xlabel="Time (ms)", ylabel="Voltage (mV)")
