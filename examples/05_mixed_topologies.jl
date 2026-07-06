# ==============================================================================
# Example 5: Mixed Topologies (Scalar Hub and Vectorized Populations)
# ============================================================================== 
# 
# This example demonstrates the flexibility of MTKNeuralToolkit's symbolic 
# interfaces. You can seamlessly mix `Scalar` and `Vectorized` topologies in the 
# same network. Because vectorized compartments expose their states as symbolic 
# arrays, you can directly index into them (e.g., `pop.interfaces.V[1]`) to wire 
# up standard scalar synapses to specific elements of a population.
#
# We create a "hub" neuron (Scalar) that interacts with an E/I population pair 
# (Vectorized). The hub excites the first neuron of the E-pop, which inhibits 
# the hub. The E-pop also projects densely to the I-pop using a weight matrix.

using MTKNeuralToolkit
using MTKNeuralToolkit.HodgkinHuxley: SodiumChannel, PotassiumChannel, LeakChannel
using ModelingToolkit: mtkcompile, @named
using ModelingToolkitStandardLibrary.Blocks: Sine
using OrdinaryDiffEq
using Plots

# ------------------------------------------------------------------------------
# 1. Build the Scalar "Hub" Neuron
# ------------------------------------------------------------------------------
top_scalar = Scalar()

function build_hh_neuron(name::Symbol)
    @named cap  = Capacitor(topology=top_scalar, C=1.0)
    @named na   = SodiumChannel(topology=top_scalar)
    @named k    = PotassiumChannel(topology=top_scalar)
    @named leak = LeakChannel(topology=top_scalar)
    return build_compartment(cap, [na, k, leak]; name=name, V_init=-65.0, topology=top_scalar)
end

hub_neuron = build_hh_neuron(:hub_neuron)

# ------------------------------------------------------------------------------
# 2. Build the Vectorized Populations (E and I)
# ------------------------------------------------------------------------------
N_E = 5
N_I = 3
top_E = Vectorized(N_E)
top_I = Vectorized(N_I)

function build_vector_pop(name::Symbol, top)
    @named cap  = Capacitor(topology=top, C=1.0)
    @named na   = SodiumChannel(topology=top)
    @named k    = PotassiumChannel(topology=top)
    @named leak = LeakChannel(topology=top)
    return build_compartment(cap, [na, k, leak]; name=name, V_init=-65.0, topology=top)
end

pop_E = build_vector_pop(:pop_E, top_E)
pop_I = build_vector_pop(:pop_I, top_I)

# ------------------------------------------------------------------------------
# 3. Wire Synapses
# ------------------------------------------------------------------------------
# A. Mixed Topology (Scalar <-> Vectorized)
# The scalar hub excites the FIRST neuron in pop_E, and pop_E[1] inhibits the hub.
@named syn_hub_to_E = ExpSynapse(g_max=3.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)
@named syn_E_to_hub = ExpSynapse(g_max=5.0, τ=5.0, E_rev=-80.0, V_th=-20.0, slope=2.0)

# B. Vectorized <-> Vectorized
# pop_E projects densely to pop_I using a weight matrix
W_EI = 0.5 .* rand(N_I, N_E)
syn_EI = build_synapse_block(pop_E, pop_I, W_EI; name=:syn_EI, E_rev=0.0)

synapse_specs = [
    # Hub -> pop_E[1]
    SynapseSpec(hub_neuron.interfaces.V, pop_E.interfaces.V[1], pop_E.interfaces.I_syn[1], syn_hub_to_E),
    # pop_E[1] -> Hub
    SynapseSpec(pop_E.interfaces.V[1], hub_neuron.interfaces.V, hub_neuron.interfaces.I_syn, syn_E_to_hub),
    # pop_E -> pop_I (Vectorized block)
    syn_EI
]

# ------------------------------------------------------------------------------
# 4. Driving Stimuli & Network Assembly
# ------------------------------------------------------------------------------
# Drive the hub neuron with a steady current to keep it spiking
@named current_driver = Sine(amplitude=8.0, frequency=0.05, offset=8.0)
drivers = [(1, current_driver)]

# The network builder handles all grounding implicitly. Undriven elements 
# in pop_E and pop_I will be grounded automatically.
net = build_acausal_network([hub_neuron, pop_E, pop_I]; 
                            synapse_specs=synapse_specs, 
                            drivers=drivers, 
                            name=:mixed_net)

println("Compiling mixed-topology network...")
sys = mtkcompile(net.sys)

# Mixed systems generate block-diagonal Jacobians, so sparse solvers excel here.
prob = ODEProblem(sys, [], (0.0, 200.0), jac=true, sparse=true)

println("Solving...")
sol = solve(prob, Rosenbrock23())

# ------------------------------------------------------------------------------
# 5. Plot the Results
# ------------------------------------------------------------------------------
p1 = plot(sol, idxs=[sys.hub_neuron.cap.v], 
          title="Scalar Hub Neuron", ylabel="V (mV)", legend=false)

p2 = plot(sol, idxs=[sys.pop_E.cap.v...], 
          title="Excitatory Pop (Pop[1] synapsed)", ylabel="V (mV)", legend=false)

p3 = plot(sol, idxs=[sys.pop_I.cap.v...], 
          title="Inhibitory Pop (Driven by E)", ylabel="V (mV)", xlabel="Time (ms)", legend=false)

plot(p1, p2, p3, layout=(3,1), size=(800, 600))
