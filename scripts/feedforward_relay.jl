using MTKNeuralToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named, connect, System
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using Plots
import ModelingToolkitStandardLibrary.Electrical: Ground  # Ensure Ground is in scope

# ==========================================
# 1. Stimulus, Source, & Reference Ground
# ==========================================
@named dc_stimulus = Blocks.Constant(k = 15.0) 
@named source = CurrentSource()            # Your custom acausal current bridge
@named global_ground = Ground()            # Absolute 0V reference anchor

# ==========================================
# 2. Helper Factories
# ==========================================
build_soma() = LIFCapacitor(C = 1.0; name=:unnamed_membrane)


# ==========================================
# 3. Build Neurons via List Comprehension
# ==========================================
# Cleanly separating acausal neurons from external driving forces
neurons = [
    build_compartment(
        build_soma(), []; 
        name = Symbol(:neuron_, i)
    ) for i in 1:2
]

# ==========================================
# 4. The Excitatory Synapse
# ==========================================
@named syn_gate = EventSynapseGate(g_max = 0.6, τ = 5.0, v_th = -55.0, w = 0.3)
@named syn_battery = FixedReversal(E = 0.0) 
synapse = build_synapse(syn_gate, syn_battery, name = :syn)

# ==========================================
# 5. Connect Network & Source Loops
# ==========================================
eqs = [
    # Network Loops
    connect(neurons[1].p, synapse.p1),
    connect(neurons[1].n, synapse.n1),
    connect(neurons[2].p, synapse.p2),
    connect(neurons[2].n, synapse.n2),
    
    # 🌟 Stimulus Injection via Current Source
    connect(dc_stimulus.output, source.I),   # Causal signal into source block
    connect(source.p, neurons[1].p),         # Positive injection into Neuron 1
    connect(source.n, neurons[1].n),         # Complete current loop to Neuron 1 ground
    
    # 🌟 Anchor global voltage references to clear "floating states"
]

# ==========================================
# 6. Compile Balanced Network
# ==========================================
@named network = System(
    eqs, t, [], []; 
    systems = [neurons..., synapse, dc_stimulus, source]
)
network_c = mtkcompile(network)

# ==========================================
# 7. Solve and Plot
# ==========================================
prob = ODEProblem(network_c, [], (0.0, 3.0))
sol = solve(prob, Rosenbrock23())
plot(sol, idxs=[neurons[1].V, neurons[2].V], title="Synaptic Relay Test")
