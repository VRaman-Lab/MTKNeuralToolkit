using MTKNeuralToolkit
using MTKNeuralToolkit.ContinuousSpikers: FitzHughNagumo, MorrisLecar
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq, Plots

println("=== Building Continuous Spiker Network ===")

top = Scalar()

# ==========================================
# 1. FitzHugh-Nagumo Compartment
# ==========================================
@named cap_fhn = Capacitor(topology=top, C=1.0)
@named fhn_ch = FitzHughNagumo(topology=top, c=3.0, a=0.7, b=0.8, tau=12.5)

fhn_comp = build_compartment(cap_fhn, [fhn_ch]; name=:fhn_comp, V_init=-2.0, topology=top)

# ==========================================
# 2. Morris-Lecar Compartment
# ==========================================
@named cap_ml = Capacitor(topology=top, C=20.0)
@named ml_channels = MorrisLecar(topology=top, V_init=-20.0)

ml_comp = build_compartment(cap_ml, collect(ml_channels); name=:ml_comp, V_init=-20.0, topology=top)

# ==========================================
# 3. Connect them with a Synapse
# ==========================================
@named syn_fhn_to_ml = ExpSynapse(g_max=0.5, τ=5.0, E_rev=0.0, V_th=0.0, slope=1.0)

synapse_specs = [
    SynapseSpec(fhn_comp.interfaces.V, ml_comp.interfaces.V, ml_comp.interfaces.I_syn, syn_fhn_to_ml)
]

# ==========================================
# 4. Build & Solve Network
# ==========================================
drivers = [(1, 1.0), (2, 100.0)]

net = build_acausal_network([fhn_comp, ml_comp]; 
                            synapse_specs=synapse_specs, 
                            drivers=drivers, 
                            name=:cont_spiker_net)

println("Compiling network...")
sys = mtkcompile(net.sys)
prob = ODEProblem(sys, [], (0.0, 200.0))

println("Solving network...")
sol = solve(prob, Rosenbrock23())

# ==========================================
# 5. Plot Results
# ==========================================
p1 = plot(sol, idxs=[sys.fhn_comp.cap_fhn.v], 
          title="FitzHugh-Nagumo (Pre-synaptic)", 
          ylabel="V (mV)", legend=false)

p2 = plot(sol, idxs=[sys.ml_comp.cap_ml.v], 
          title="Morris-Lecar (Post-synaptic)", 
          ylabel="V (mV)", legend=false)

p3 = plot(sol, idxs=[sys.syn_fhn_to_ml.I_syn], 
          title="Synaptic Current", 
          ylabel="I_syn", legend=false)

final_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 800))
