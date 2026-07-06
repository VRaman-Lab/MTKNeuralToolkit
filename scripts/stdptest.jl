using MTKNeuralToolkit
using MTKNeuralToolkit.ContinuousSpikers: FitzHughNagumo
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq, Plots

println("=== Building STDP Network ===")

top = Scalar()

# ==========================================
# 1. Two FitzHugh-Nagumo Compartments
# ==========================================
# Neuron 1
@named cap1 = Capacitor(topology=top, C=1.0)
@named fhn1 = FitzHughNagumo(topology=top, c=3.0, a=0.7, b=0.8, tau=12.5)
comp1 = build_compartment(cap1, [fhn1]; name=:comp1, V_init=-2.0, topology=top)

# Neuron 2 (Slightly different 'b' parameter to create a frequency mismatch)
@named cap2 = Capacitor(topology=top, C=1.0)
@named fhn2 = FitzHughNagumo(topology=top, c=3.0, a=0.7, b=0.85, tau=12.5)
comp2 = build_compartment(cap2, [fhn2]; name=:comp2, V_init=-2.0, topology=top)

# ==========================================
# 2. Connect with STDP Synapse
# ==========================================
# V_th=1.0 captures the peak of the FHN spikes nicely.
@named stdp_syn = STDPSynapse(g_max=2.0, τ_s=5.0, τ_plus=15.0, τ_minus=15.0, 
                             A_plus=0.05, A_minus=0.05, V_th=1.0, w_init=0.5, 
                             w_max=1.0, w_min=0.0)

synapse_specs = [
    SynapseSpec(comp1.interfaces.V, comp2.interfaces.V, comp2.interfaces.I_syn, stdp_syn)
]

# ==========================================
# 3. Build & Solve Network
# ==========================================
# Different drivers to ensure they start out of phase
drivers = [(1, 1.0), (2, 0.5)]

net = build_acausal_network([comp1, comp2]; 
                            synapse_specs=synapse_specs, 
                            drivers=drivers, 
                            name=:stdp_net)

println("Compiling network...")
sys = mtkcompile(net.sys)
prob = ODEProblem(sys, [], (0.0, 1000.0))

println("Solving STDP network...")
sol = solve(prob, Rosenbrock23())

# ==========================================
# 4. Plot Results
# ==========================================
p1 = plot(sol, idxs=[sys.comp1.cap1.v, sys.comp2.cap2.v], 
          label=["Pre-synaptic (Comp1)" "Post-synaptic (Comp2)"], 
          title="Membrane Potentials", ylabel="V (mV)", legend=true)

p2 = plot(sol, idxs=[sys.stdp_syn.w], 
          label="Synaptic Weight (w)", 
          title="STDP Weight Evolution", ylabel="w", xlabel="Time", legend=true)

# Zoom in on the weight change
p3 = plot(sol, idxs=[sys.stdp_syn.x, sys.stdp_syn.y], 
          label=["Pre trace (x)" "Post trace (y)"], 
          title="Activity Traces", xlabel="Time", legend=true)

final_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 800))
savefig(final_plot, "stdp_demo.png")
println("Plot saved to stdp_demo.png")
