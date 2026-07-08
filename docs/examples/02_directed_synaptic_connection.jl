# # Example 2: Directed Synaptic Connection (Two Neurons)
# 
# Now let's connect two neurons. This example introduces network assembly and 
# chemical synapses by linking a presynaptic neuron to a postsynaptic one via an 
# exponential synapse.
#
# Instead of building the HH channels from scratch like we did in Example 1, we'll 
# use the pre-built channels from the `MTKNeuralToolkit.HodgkinHuxley` standard 
# library to show how quickly you can spin up standard models.

using MTKNeuralToolkit
using MTKNeuralToolkit.HodgkinHuxley: SodiumChannel, PotassiumChannel, LeakChannel
using ModelingToolkit: mtkcompile, @named
using ModelingToolkitStandardLibrary.Blocks: Sine
using OrdinaryDiffEq
using Plots

# ------------------------------------------------------------------------------
# 1. Build Two Neurons using the Standard Library
# ------------------------------------------------------------------------------
top = Scalar()

# This helper function shows how to pass kwargs to the standard library channels. 
# You can tweak maximal conductances (g) or reversal potentials (E_rev) without 
# rewriting the gating dynamics from scratch.
function build_hh_neuron(name::Symbol; gNa=120.0, ENa=50.0, gK=36.0, EK=-77.0, 
                                       gleak=0.3, Eleak=-54.4)
    @named cap  = Capacitor(topology=top, C=1.0) 
    @named na   = SodiumChannel(topology=top, g=gNa, E_rev=ENa)
    @named k    = PotassiumChannel(topology=top, g=gK, E_rev=EK)
    @named leak = LeakChannel(topology=top, g=gleak, E_rev=Eleak)
    
    return build_compartment(cap, [na, k, leak]; name=name, V_init=-65.0, topology=top)
end

pre_neuron  = build_hh_neuron(:pre_neuron)

# We'll give the postsynaptic neuron altered channel parameters (e.g., a lower 
# sodium conductance and a different leak reversal) to make it distinct.
post_neuron = build_hh_neuron(:post_neuron; gNa=80.0, ENa=45.0, gleak=0.5)


# ------------------------------------------------------------------------------
# 2. Define the Synapse
# ------------------------------------------------------------------------------
# `ExpSynapse` is a continuous, sigmoid-driven exponential synapse. When the 
# presynaptic voltage crosses the threshold (V_th), the gating variable `s` 
# increases, injecting current into the postsynaptic compartment.

@named excitatory_synapse = ExpSynapse(g_max=2.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)

# Tip: To make this synapse inhibitory, you just change E_rev to a value below 
# the postsynaptic resting potential (e.g., E_rev=-80.0). No other code changes needed.
# @named inhibitory_synapse = ExpSynapse(g_max=2.0, τ=5.0, E_rev=-80.0, V_th=-20.0, slope=2.0)
#
# Tip: Look at the source code for ExpSynapse: it's pretty simple! You can modify the equations to make your own custom synapse. The interface is extremely flexible: you can have complicated multi-state synapses 


# ------------------------------------------------------------------------------
# 3. Wire the Network using SynapseSpec
# ------------------------------------------------------------------------------
# A `SynapseSpec` maps the pre- and post-synaptic voltages to the synapse block, 
# and designates which current variable in the postsynaptic compartment will 
# receive the injected current.

synapse_specs = [
    SynapseSpec(
        pre_neuron.interfaces.V,
        post_neuron.interfaces.V,
        post_neuron.interfaces.I_syn,
        excitatory_synapse
    )
]

# ------------------------------------------------------------------------------
# 4. Driving Stimuli
# ------------------------------------------------------------------------------
# We'll drive the presynaptic neuron with a time-varying sinusoidal current so 
# it spikes rhythmically. The postsynaptic neuron receives no external drive, 
# so any spiking it does is purely from the synaptic connection.

@named current_driver = Sine(amplitude=8.0, frequency=0.05, offset=8.0)

drivers = [(1, current_driver)] 

# ------------------------------------------------------------------------------
# 5. Build and Simulate the Network
# ------------------------------------------------------------------------------
net = build_acausal_network([pre_neuron, post_neuron]; 
                            synapse_specs=synapse_specs, 
                            drivers=drivers, 
                            name=:two_neuron_net)

println("Compiling 2-neuron network...")
sys = mtkcompile(net.sys)
prob = ODEProblem(sys, [], (0.0, 200.0))

println("Solving...")
sol = solve(prob, Rosenbrock23())

# ------------------------------------------------------------------------------
# 6. Plot the Results
# ------------------------------------------------------------------------------
p1 = plot(sol, idxs=[sys.pre_neuron.cap.v], 
          title="Presynaptic Neuron (Driven)", ylabel="V (mV)", legend=false)

p2 = plot(sol, idxs=[sys.post_neuron.cap.v], 
          title="Postsynaptic Neuron (Synaptically Driven)", ylabel="V (mV)", legend=false)

p3 = plot(sol, idxs=[sys.excitatory_synapse.I_syn], 
          title="Synaptic Current", ylabel="I_syn", xlabel="Time (ms)", legend=false)

plot(p1, p2, p3, layout=(3,1), size=(800, 700))
