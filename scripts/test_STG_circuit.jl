using MTKNeuralToolkit
using MTKNeuralToolkit.PrinzNeuron
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq, Plots

# ==========================================
# 1. Network Parameters 
# ==========================================
#

const geom = PrinzGeometry(area=0.0628, C_m=10.0)  # area in cm^2, C_m in uF/cm^2
const tauCa = 200.0
const Ca_inf = 0.05

# Ion config for Calcium Tracker
const prinz_ion_config = CalciumTracker(
    decay=ca -> (Ca_inf .- ca) ./ tauCa, 
    Ca_init=Ca_inf
)

# Nernst factor for Calcium
const nernst_factor = 500.0 * 8.6174e-5 * 283.15

# ==========================================
# 2. Local Channel Builders 
# ==========================================

NaCh(g; name)   = GenericChannel(name=name, g=g, E_rev=50.0, gates=na_gates, geometry=geom)
CaSCh(g; name)  = CaVChannel(name=name, g=g, gates=cas_gates, Ca_out=3000.0, 
                             nernst_factor=nernst_factor, geometry=geom, tauCa=tauCa)
CaTCh(g; name)  = CaVChannel(name=name, g=g, gates=cat_gates, Ca_out=3000.0, 
                             nernst_factor=nernst_factor, geometry=geom, tauCa=tauCa)
HCh(g; name)    = GenericChannel(name=name, g=g, E_rev=-20.0, gates=h_gates, geometry=geom)
KaCh(g; name)   = GenericChannel(name=name, g=g, E_rev=-80.0, gates=ka_gates, geometry=geom)
KCaCh(g; name)  = KCaChannel(name=name, g=g, E_rev=-80.0, gates=kca_gates, geometry=geom)
KdrCh(g; name)  = GenericChannel(name=name, g=g, E_rev=-80.0, gates=kdr_gates, geometry=geom)
LeakCh(g; name) = GenericChannel(name=name, g=g, E_rev=-50.0, gates=GateSpec[], geometry=geom)

# ==========================================
# 3. Build Neurons from Channels
# ==========================================
function build_AB()
    @named cap  = Capacitor(geometry=geom)  # C is computed from geom.C_m * geom.area
    @named na   = NaCh(100.0)
    @named cas  = CaSCh(6.0)
    @named cat  = CaTCh(2.5)
    @named h    = HCh(0.01)
    @named ka   = KaCh(50.0)
    @named kca  = KCaCh(5.0)
    @named kdr  = KdrCh(100.0)
    
    return build_compartment(cap, [na, cas, cat, h, ka, kca, kdr]; 
                             name=:AB, V_init=-60.0, ion_config=prinz_ion_config)
end

function build_PY()
    @named cap  = Capacitor(geometry=geom)
    @named na   = NaCh(100.0)
    @named cas  = CaSCh(2.0)
    @named cat  = CaTCh(2.4)
    @named h    = HCh(0.05)
    @named ka   = KaCh(50.0)
    @named kdr  = KdrCh(125.0)
    @named leak = LeakCh(0.01)

    return build_compartment(cap, [na, cas, cat, h, ka, kdr, leak]; 
                             name=:PY, V_init=-55.0, ion_config=prinz_ion_config)
end

function build_LP()
    @named cap  = Capacitor(geometry=geom)
    @named na   = NaCh(100.0)
    @named cas  = CaSCh(4.0)
    @named h    = HCh(0.05)
    @named ka   = KaCh(20.0)
    @named kdr  = KdrCh(25.0)
    @named leak = LeakCh(0.03)

    return build_compartment(cap, [na, cas, h, ka, kdr, leak]; 
                             name=:LP, V_init=-65.0, ion_config=prinz_ion_config)
end

AB = build_AB(); LP = build_LP(); PY = build_PY()
neurons = [AB, PY, LP]

# ==========================================
# 4. Define Synapses & Network
# ==========================================
# For synapses, we can simply scale g_max by the area if assuming specific conductance,
# or just use absolute values if preferred. Here we pass absolute values for simplicity.
function STG_synapses()
    @named ABLP_chol = CholSynapse(g_max=30.0, geometry=geom)
    @named ABPY_chol = CholSynapse(g_max=3.0 , geometry=geom)
    @named ABLP_glut = GlutSynapse(g_max=30.0, geometry=geom)
    @named ABPY_glut = GlutSynapse(g_max=10.0, geometry=geom)
    @named LPAB_glut = GlutSynapse(g_max=30.0, geometry=geom)
    @named LPPY_glut = GlutSynapse(g_max=1.0 , geometry=geom)
    @named PYLP_glut = GlutSynapse(g_max=30.0, geometry=geom)

    synapse_specs = [
        SynapseSpec(LP.interfaces.V, AB.interfaces.V, AB.interfaces.I_syn, LPAB_glut),
        SynapseSpec(AB.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, ABPY_chol),
        SynapseSpec(AB.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, ABPY_glut),
        SynapseSpec(LP.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, LPPY_glut),
        SynapseSpec(AB.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, ABLP_chol),
        SynapseSpec(AB.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, ABLP_glut),
        SynapseSpec(PY.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, PYLP_glut)
    ]
    return synapse_specs
end

net = build_acausal_network(neurons; synapse_specs=STG_synapses(), name=:stg)
net_compiled = mtkcompile(net.sys)

# ==========================================
# 5. Simulate & Plot
# ==========================================
tspan = (0.0, 10000.0)
prob = ODEProblem(net_compiled, [], tspan, jac=true, sparse=true)

@time sol = solve(prob, Rosenbrock23())

p1 = plot(sol, idxs=[net_compiled.AB.cap.v], title="AB Neuron", legend=false)
p2 = plot(sol, idxs=[net_compiled.LP.cap.v], title="LP Neuron", legend=false)
p3 = plot(sol, idxs=[net_compiled.PY.cap.v], title="PY Neuron", legend=false)

plot(p1, p2, p3, layout=(3,1), size=(800,600))
