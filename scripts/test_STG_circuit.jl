using MTKNeuralToolkit
using MTKNeuralToolkit.PrinzNeuron
using ModelingToolkit: t_nounits as t, D_nounits as D, @named, @component, @variables, @parameters, System, Equation, mtkcompile
using OrdinaryDiffEq
using Plots

# ==========================================
# 1. Define Custom Synapses (Chol & Glut)
# ==========================================

@component function CholSynapse(; name, g_max=30.0, E_rev=-80.0, k_minus=0.01, V_th=-35.0, delta=5.0)
    @variables s(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max E_rev=E_rev k_minus=k_minus V_th=V_th delta=delta
    
    # Prinz Synapse dynamics
    s_inf = 1.0 / (1.0 + exp((V_th - V_pre) / delta))
    tau_s = (1.0 - s_inf) / k_minus
    
    eqs = Equation[
        D(s) ~ (s_inf - s) / tau_s,
        I_syn ~ g_max * s * (E_rev - V_post)  # inward current when E_rev < V_post
    ]
    return System(eqs, t, [s, I_syn, V_pre, V_post], [g_max, E_rev, k_minus, V_th, delta]; systems=System[], name=name)
end

@component function GlutSynapse(; name, g_max=30.0, E_rev=-70.0, k_minus=0.025, V_th=-35.0, delta=5.0)
    @variables s(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max E_rev=E_rev k_minus=k_minus V_th=V_th delta=delta
    
    s_inf = 1.0 / (1.0 + exp((V_th - V_pre) / delta))
    tau_s = (1.0 - s_inf) / k_minus
    
    eqs = Equation[
        D(s) ~ (s_inf - s) / tau_s,
        I_syn ~ g_max * s * (E_rev - V_post)
    ]
    return System(eqs, t, [s, I_syn, V_pre, V_post], [g_max, E_rev, k_minus, V_th, delta]; systems=System[], name=name)
end

# ==========================================
# 2. Define Network Parameters
# ==========================================
const Area = 0.0628 
const Cm = 10.0 
const Prinz_conv = Cm / Area
const synaptic_conv = 1e-3 / Area^2 

# ==========================================
# 3. Build Neurons (AB, PY, LP)
# ==========================================
AB = build_prinz_neuron(name=:AB, Cm=Cm, tauCa=200.0, Ca_inf=0.05, V_init=-60.0,
                        gNa=100.0 * Prinz_conv, gCaS=6.0 * Prinz_conv, gCaT=2.5 * Prinz_conv, 
                        gH=0.01 * Prinz_conv, gKa=50.0 * Prinz_conv, gKCa=5.0 * Prinz_conv, 
                        gKdr=100.0 * Prinz_conv, gleak=0.0)

PY = build_prinz_neuron(name=:PY, Cm=Cm, tauCa=200.0, Ca_inf=0.05, V_init=-55.0,
                        gNa=100.0 * Prinz_conv, gCaS=2.0 * Prinz_conv, gCaT=2.4 * Prinz_conv,
                        gH=0.05 * Prinz_conv, gKa=50.0 * Prinz_conv, gKCa=0.0, gKdr=125.0 * Prinz_conv,
                        gleak=0.01 * Prinz_conv)

LP = build_prinz_neuron(name=:LP, Cm=Cm, tauCa=200.0, Ca_inf=0.05, V_init=-65.0,
                        gNa=100.0 * Prinz_conv, gCaS=4.0 * Prinz_conv, gCaT=0.0, gH=0.05 * Prinz_conv,
                        gKa=20.0 * Prinz_conv, gKCa=0.0, gKdr=25.0 * Prinz_conv,
                        gleak=0.03 * Prinz_conv)

neurons = [AB, PY, LP]

# ==========================================
# 4. Define Synapses
# ==========================================
@named ABLP_chol = CholSynapse(g_max=30.0 * synaptic_conv)
@named ABPY_chol = CholSynapse(g_max=3.0 * synaptic_conv)
@named ABLP_glut = GlutSynapse(g_max=30.0 * synaptic_conv)
@named ABPY_glut = GlutSynapse(g_max=10.0 * synaptic_conv)
@named LPAB_glut = GlutSynapse(g_max=30.0 * synaptic_conv)
@named LPPY_glut = GlutSynapse(g_max=1.0 * synaptic_conv)
@named PYLP_glut = GlutSynapse(g_max=30.0 * synaptic_conv)

# Create SynapseSpecs mapping pre_V -> post_V, post_I_syn
synapse_specs = [
    SynapseSpec(LP.interfaces.V, AB.interfaces.V, AB.interfaces.I_syn, LPAB_glut),
    SynapseSpec(AB.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, ABPY_chol),
    SynapseSpec(AB.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, ABPY_glut),
    SynapseSpec(LP.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, LPPY_glut),
    SynapseSpec(AB.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, ABLP_chol),
    SynapseSpec(AB.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, ABLP_glut),
    SynapseSpec(PY.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, PYLP_glut)
]

# ==========================================
# 5. Build & Simulate Network
# ==========================================
net = build_acausal_network(neurons; synapse_specs=synapse_specs, name=:stg)
net_compiled = mtkcompile(net.sys)

tspan = (0.0, 10000.0)
prob = ODEProblem(net_compiled, [], tspan)
@time sol = solve(prob, AutoTsit5(Rosenbrock23()))

# ==========================================
# 6. Plotting
# ==========================================
p1 = plot(sol, idxs=net_compiled.AB.cap.v, title="AB Neuron", legend=false)
p2 = plot(sol, idxs=net_compiled.LP.cap.v, title="LP Neuron", legend=false)
p3 = plot(sol, idxs=net_compiled.PY.cap.v, title="PY Neuron", legend=false)

plot(p1, p2, p3, layout=(3,1), size=(800,600))
