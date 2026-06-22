using MTKNeuralToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named, System
using OrdinaryDiffEq
using Plots

# =============================================================================
# 1. Define the Neurons (Exposing .V and keeping .injector open)
# =============================================================================
@named soma1 = Capacitor(C = 1.0)
@named soma2 = Capacitor(C = 1.0)

function make_hh_channels(suffix)
    na = build_channel(nagates(name=:gate), FixedReversal(E = 50.0, name=:batt); name=Symbol(:na_, suffix))
    k  = build_channel(kgates(name=:gate),  FixedReversal(E = -77.0, name=:batt); name=Symbol(:k_, suffix))
    l  = build_channel(lgates(name=:gate),  FixedReversal(E = -54.4, name=:batt); name=Symbol(:l_, suffix))
    return [na, k, l]
end

nrn1 = build_compartment(soma1, make_hh_channels("n1"); name=:nrn1)
nrn2 = build_compartment(soma2, make_hh_channels("n2"); name=:nrn2)
neurons = [nrn1, nrn2]

# =============================================================================
# 2. Define the Connectivity Matrix (Using functional constructors)
# =============================================================================
# Initialize an empty N x N matrix holding Any (Nothing or our closures)
connectivity = Matrix{Any}(nothing, 2, 2)

# Populating a direct component factory from Neuron 1 -> Neuron 2 using keyword-name mapping
connectivity[1, 2] = (; name) -> AlphaSynapse(; name, g_max = 2.0, τ = 5.0, E_rev = 0.0, v_th = -20.0, w = 0.5)

# Convert the adjacency matrix to the flat connections list format required by your engine
connections_list = Tuple{Int, Int, Any, Symbol}[]
for i in 1:size(connectivity, 1)
    for j in 1:size(connectivity, 2)
        if !isnothing(connectivity[i, j])
            # Perfect! nameof gets the clean :nrn1 and :nrn2 symbols
            syn_name = Symbol("synapse_", nameof(neurons[i]), "_to_", nameof(neurons[j]))
            push!(connections_list, (i, j, connectivity[i, j], syn_name))
        end
    end
end

# =============================================================================
# 3. Setup External Driving Stimulus for Neuron 1
# =============================================================================
@named stim = Blocks.Sine(frequency = 0.05, amplitude = 15.0)

# Pass the tuple directly targeting Neuron 1
drivers = [
    (1, stim) 
]

# =============================================================================
# 4. Compile and Run Factored Network
# =============================================================================
@named net = build_factored_synapse_network(neurons, connections_list; drivers=drivers)
net_compiled = mtkcompile(net)

# Solve using an event-friendly solver
prob = ODEProblem(net_compiled, Pair[], (0.0, 100.0))
sol = solve(prob, Rosenbrock23())

# =============================================================================
# 5. Plot Results
# =============================================================================
plot(sol, idxs=[net.nrn1.V, net.nrn2.V], 
     title="Factored Equation Network Dynamics", 
     label=["Neuron 1 (Pre)" "Neuron 2 (Post)"],
     xlabel="Time (ms)", ylabel="Voltage (mV)", lw=1.5)
