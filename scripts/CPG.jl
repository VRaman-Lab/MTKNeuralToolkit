"""
WAS WORKING BUT NOT SINCE I TRIED TO CHANGE BUILD HET NETWORK. 
    
"""


using ModelingToolkit
using ModelingToolkit: mtkcompile, @named, System
using ModelingToolkit: t_nounits as t
using OrdinaryDiffEq
using Plots


# ==============================================================================
# 1. Instantiate the isolated neurons with clean, local V_init defaults
# ==============================================================================
# Define a clean, reusable component factory
make_soma() = LIFCapacitor(C = 1.0, name = :soma)

# ==============================================================================
# 1. Instantiate the isolated neurons with clean, component-level V_init defaults
# ==============================================================================
neurons = System[
    build_compartment(make_soma(), []; has_synapses = true, name = :n1),
    build_compartment(make_soma(), []; has_synapses = true, name = :n2),
    build_compartment(make_soma(), []; has_synapses = true, name = :n3)
]

# ==============================================================================
# 2. Build the heterogeneous synapse factory matrix (3x3)
# ==============================================================================
synapse_matrix = Matrix{Any}(nothing, 3, 3)

# n1 -> n2
synapse_matrix[1, 2] = (; name) -> AlphaSynapse(; name, g_max=0.4, τ=5.0, E_rev=0.0, v_th=-55.0, w=0.1)
# n2 -> n3
synapse_matrix[2, 3] = (; name) -> AlphaSynapse(; name, g_max=0.4, τ=5.0, E_rev=0.0, v_th=-55.0, w=0.1)
# n3 -> n1
synapse_matrix[3, 1] = (; name) -> AlphaSynapse(; name, g_max=0.4, τ=5.0, E_rev=0.0, v_th=-55.0, w=0.1)

# ==============================================================================
# 3. Define the Symbolic Stimulus Expression
# ==============================================================================
sine_stimulus = 12.5 * sin(2 * π * 0.1 * t)
stimulus_exprs = [sine_stimulus, nothing, nothing]

# ==============================================================================
# 4. Assemble the network using behavioral routing
# ==============================================================================
@named final_network = build_heterogeneous_network(neurons, synapse_matrix)

# Grab the namespaced discrete handle safely from your system object
stim1 = final_network.user_stim_1

# Build the discrete callbacks to change the value at exact points in time
kick_on  = ModelingToolkit.SymbolicDiscreteCallback(1.0  => [stim1 ~ 15.0], discrete_parameters = stim1, iv = t)
kick_off = ModelingToolkit.SymbolicDiscreteCallback(30.0 => [stim1 ~ 0.0],  discrete_parameters = stim1, iv = t)

compiled_net = mtkcompile(final_network, discrete_events = [kick_on, kick_off])

# Map initial condition values at t = 0.0
my_initial_maps = [
    stim1 => 0.0,
    final_network.user_stim_2 => 0.0,
    final_network.user_stim_3 => 0.0
]

prob = ODEProblem(compiled_net, my_initial_maps, (0.0, 50.0))
sol = solve(prob, Tsit5())



# Plot the membrane potentials of the three neurons to watch the spikes cascade!
plot(sol, idxs=[neurons[1].soma.v, neurons[2].soma.v, neurons[3].soma.v], 
     label=["Neuron 1" "Neuron 2" "Neuron 3"], ylabel="Voltage (mV)", xlabel="Time (ms)")
