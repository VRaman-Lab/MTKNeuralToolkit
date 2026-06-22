using MTKNeuralToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named, System, SymbolicT
using OrdinaryDiffEq
using Plots

function make_lif_population(prefix::Symbol, size=2)
    neurons = System[]
    for i in 1:size
        @named soma = LIFCapacitor(C = 1.0)

        # Type-stable channels vector collection
        channels = System[]
        push!(channels, build_channel(lgates(name=:gate), FixedReversal(E = -54.4, name=:batt); name=:leak))

        nrn = build_compartment(soma, channels; name = Symbol(prefix, i))
        push!(neurons, nrn)
    end
    internal_connections = [
        (neurons[1], neurons[2], (; name) -> AlphaSynapse(; name=name, g_max = 3.0, τ = 2.0, E_rev = -80.0, v_th = -30.0, w = 0.5)),
        (neurons[2], neurons[1], (; name) -> AlphaSynapse(; name=name, g_max = 2.5, τ = 4.0, E_rev = 0.0,  v_th = -30.0, w = 0.5))
    ]

    return neurons, internal_connections
end

# Generates a group of Hodgkin-Huxley neurons and their internal connections
function make_hh_population(prefix::Symbol, size=2)
    neurons = System[]

    function make_hh_channels(suffix)
        channels = System[]
        push!(channels, build_channel(nagates(name=:gate), FixedReversal(E = 50.0, name=:batt); name=Symbol(:na_, suffix)))
        push!(channels, build_channel(kgates(name=:gate),  FixedReversal(E = -77.0, name=:batt); name=Symbol(:k_, suffix)))
        push!(channels, build_channel(lgates(name=:gate),  FixedReversal(E = -54.4, name=:batt); name=Symbol(:l_, suffix)))
        return channels
    end

    for i in 1:size
        @named soma = Capacitor(C = 1.0)
        nrn = build_compartment(soma, make_hh_channels(string(prefix, i)); name = Symbol(prefix, i))
        push!(neurons, nrn)
    end

    internal_connections = [
        (neurons[1], neurons[2], (; name) -> AlphaSynapse(; name=name, g_max = 1.2, τ = 3.0, E_rev = 0.0, v_th = -20.0, w = 0.5))
    ]
    return neurons, internal_connections
end

# =============================================================================
# 2. Instantiate and Flatten Nodes & Edges
# =============================================================================
lif_nodes, lif_edges = make_lif_population(:lif, 2)
hh_nodes,  hh_edges  = make_hh_population(:hh, 2)

# Combine neurons cleanly
all_neurons = System[]
append!(all_neurons, lif_nodes)
append!(all_neurons, hh_nodes)

# =============================================================================
# 3. Inter-Population Cross-Connectivity Specifications
# =============================================================================
inter_edges = [
    (lif_nodes[1], hh_nodes[1], (; name) -> AlphaSynapse(; name=name, g_max = 2.0, τ = 5.0, E_rev = 0.0, v_th = -20.0, w = 0.5)),
    (lif_nodes[2], hh_nodes[2], (; name) -> AlphaSynapse(; name=name, g_max = 2.0, τ = 5.0, E_rev = 0.0, v_th = -20.0, w = 0.5))
]

# Flatten all connections into a uniform Tuple vector matching our new engine schema
# Using generic `Tuple[]` avoids type mismatch errors when appending heterogeneous tuples
all_connections = Tuple[]
append!(all_connections, lif_edges)
append!(all_connections, hh_edges)
append!(all_connections, inter_edges)

# =============================================================================
# 4. Setup Driving Stimulus
# =============================================================================
@named stim = Blocks.Sine(frequency = 0.05, amplitude = 14.0)
drivers = [
    (lif_nodes[1], stim) # Target the specific neuron system directly
]

# =============================================================================
# 5. Build the Isolated Factored Synapse Block
# =============================================================================
# The refactored engine builds an isolated synapse block that handles event
# logic and stimulus current internally, exposing array IO boundaries.
synapse_block = build_factored_synapse_network(all_neurons, all_connections; drivers=drivers, name=:synapse_net)

# =============================================================================
# 6. Assemble Top-Level Network: Connect Neurons to Synapse Block
# =============================================================================
net_eqs = Equation[]
all_systems = System[]
append!(all_systems, all_neurons)
push!(all_systems, synapse_block)

for i in 1:length(all_neurons)
    # Map neuron membrane voltage to the synapse block input array
    push!(net_eqs, synapse_block.V_in.u[i] ~ all_neurons[i].V)

    # Map synapse block current output array back into the neuron's current injector
    push!(net_eqs, all_neurons[i].injector.I.u ~ synapse_block.I_out.u[i])
end

@named net = System(net_eqs, t, SymbolicT[], SymbolicT[]; systems=all_systems)

# =============================================================================
# 7. Compile and Simulate
# =============================================================================
net_compiled = mtkcompile(net)
prob = ODEProblem(net_compiled, Pair[], (0.0, 100.0))
sol = solve(prob, Rosenbrock23())

# =============================================================================
# 6. Plot the Heterogeneous Network Output
# =============================================================================
plot(sol, idxs=[net.lif1.V, net.hh1.V], 
     title="Heterogeneous Network Dynamics (LIF -> HH)", 
     label=["LIF 1 (Driven Pre)" "HH 1 (Post)"],
     xlabel="Time (ms)", ylabel="Voltage (mV)", lw=1.5)
