using MTKNeuralToolkit
using MTKNeuralToolkit.ContinuousSpikers: FitzHughNagumo
using ModelingToolkit: mtkcompile, @named
using ModelingToolkitStandardLibrary.Blocks: Sine, RealInput
using OrdinaryDiffEq, Plots

println("=== Building STDP Pairing Protocol Network ===")

top = Scalar()

# 1. Build two quiescent FHN compartments (no intrinsic spiking without heavy input)
@named cap1 = Capacitor(topology=top, C=1.0)
@named fhn1 = FitzHughNagumo(topology=top, c=3.0, a=0.7, b=0.8, tau=12.5)
comp1 = build_compartment(cap1, [fhn1]; name=:comp1, V_init=-2.0, topology=top)

@named cap2 = Capacitor(topology=top, C=1.0)
@named fhn2 = FitzHughNagumo(topology=top, c=3.0, a=0.7, b=0.8, tau=12.5)
comp2 = build_compartment(cap2, [fhn2]; name=:comp2, V_init=-2.0, topology=top)

# 2. Create an STDP synapse with a higher learning rate to see the effect quickly
@named stdp_syn = STDPSynapse(g_max=3.0, τ_s=5.0, τ_plus=15.0, τ_minus=15.0, 
                             A_plus=0.5, A_minus=0.5, V_th=1.0, w_init=0.5, 
                             w_max=1.0, w_min=0.0)

synapse_specs = [
    SynapseSpec(comp1.interfaces.V, comp2.interfaces.V, comp2.interfaces.I_syn, stdp_syn)
]

# 3. External Driving Stimuli (Pairing Protocol)
# FHN spikes when given a strong current push. We use Sine waves to rhythmically push them.
# Frequency = 0.2 Hz (period of 5s). We give them strong amplitude to force spikes.
# Pre starts at phase 0, Post starts at phase -0.4 (which means Post fires slightly *before* Pre initially)

# We'll create custom MTK blocks for the periodic forcing
@named pre_stim = Sine(amplitude=3.0, frequency=0.2, phase=0.0, offset=0.0)
@named post_stim = Sine(amplitude=3.0, frequency=0.2, phase=-0.4, offset=0.0) # Post fires first

drivers = [
    (1, pre_stim), 
    (2, post_stim)
]

# 4. Build & Solve
net = build_acausal_network([comp1, comp2]; 
                            synapse_specs=synapse_specs, 
                            drivers=drivers, 
                            name=:stdp_protocol_net)

println("Compiling...")
sys = mtkcompile(net.sys)
prob = ODEProblem(sys, [], (0.0, 25.0))

println("Solving...")
sol = solve(prob, Rosenbrock23())

# 5. Plotting
p1 = plot(sol, idxs=[sys.comp1.cap1.v, sys.comp2.cap2.v], 
          label=["Pre" "Post"], title="Forced Spiking", ylabel="V")

p2 = plot(sol, idxs=[sys.stdp_syn.w], 
          label="Synaptic Weight (w)", title="STDP Weight (LTP/LTD)", 
          ylabel="w", xlabel="Time", legend=true)

plot(p1, p2, layout=(2,1), size=(800, 600))
