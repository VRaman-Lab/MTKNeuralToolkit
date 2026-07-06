using MTKNeuralToolkit
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq
using Plots

N = 20
top = Vectorized(N)

# === Build vectorized HH compartment ===
@named soma = Capacitor(topology=top, C=1.0)

hh_na_m = v -> (
    0.182 .* (v .+ 35.0) ./ (1.0 .- exp.(-(v .+ 35.0) ./ 9.0)),
    -0.124 .* (v .+ 35.0) ./ (1.0 .- exp.((v .+ 35.0) ./ 9.0))
)
hh_na_h = v -> (
    0.25 .* exp.(-(v .+ 90.0) ./ 12.0),
    0.25 .* (exp.((v .+ 62.0) ./ 6.0)) ./ exp.(-(v .+ 90.0) ./ 12.0)
)
sodium_gates = [GateSpec(:m, 3, 0.0, hh_na_m), GateSpec(:h, 1, 0.0, hh_na_h)]

hh_k_n = v -> (
    0.02 .* (v .- 25.0) ./ (1.0 .- exp.(-(v .- 25.0) ./ 9.0)),
    -0.002 .* (v .- 25.0) ./ (1.0 .- exp.((v .- 25.0) ./ 9.0))
)
potassium_gates = [GateSpec(:n, 4, 0.0, hh_k_n)]

@named sodium_channel = GenericChannel(topology=top, g=120.0, E_rev=50.0, gates=sodium_gates)
@named potassium_channel = GenericChannel(topology=top, g=36.0, E_rev=-77.0, gates=potassium_gates)
@named leak_channel = GenericChannel(topology=top, g=0.3, E_rev=-54.4, gates=GateSpec[])

hh = build_compartment(soma, [sodium_channel, potassium_channel, leak_channel];
                        name=:hh, V_init=-65.0, topology=top)

# === Create synapses ===
@named syn_1to2 = ExpSynapse(g_max=2.0,  τ=5.0,  E_rev=0.0,   V_th=-20.0, slope=2.0)
@named syn_4to2 = ExpSynapse(g_max=1.5,  τ=5.0,  E_rev=0.0,   V_th=-20.0, slope=2.0)
@named syn_1to3 = ExpSynapse(g_max=1.0,  τ=8.0,  E_rev=0.0,   V_th=-20.0, slope=2.0)
@named syn_5to3 = ExpSynapse(g_max=1.2,  τ=8.0,  E_rev=-80.0, V_th=-20.0, slope=2.0)
@named syn_3to7 = ExpSynapse(g_max=2.0,  τ=5.0,  E_rev=0.0,   V_th=-20.0, slope=2.0)
@named syn_2to8 = ExpSynapse(g_max=1.5,  τ=5.0,  E_rev=0.0,   V_th=-20.0, slope=2.0)

# === SynapseSpecs with concrete compartment variables ===
synapse_specs = [
    SynapseSpec(hh.interfaces.V[1], hh.interfaces.V[2], hh.interfaces.I_syn[2], syn_1to2),
    SynapseSpec(hh.interfaces.V[4], hh.interfaces.V[2], hh.interfaces.I_syn[2], syn_4to2),
    SynapseSpec(hh.interfaces.V[1], hh.interfaces.V[3], hh.interfaces.I_syn[3], syn_1to3),
    SynapseSpec(hh.interfaces.V[5], hh.interfaces.V[3], hh.interfaces.I_syn[3], syn_5to3),
    SynapseSpec(hh.interfaces.V[3], hh.interfaces.V[7], hh.interfaces.I_syn[7], syn_3to7),
    SynapseSpec(hh.interfaces.V[2], hh.interfaces.V[8], hh.interfaces.I_syn[8], syn_2to8),
]

# === Drivers: graded current ===
drivers = [(1, collect(Float64, 1:2:2N))]

# === Build via refactored build_acausal_network ===
net = build_acausal_network([hh]; synapse_specs=synapse_specs, drivers=drivers)

@time net_compiled = mtkcompile(net.sys)
prob = ODEProblem(net_compiled, [], (0.0, 50.0))
@time sol = solve(prob, Rosenbrock23())

# === Plot ===
p1 = plot(sol, idxs=[net_compiled.hh.soma.v...],
          title="build_acausal_network API", xlabel="Time", ylabel="V (mV)")

p2 = plot(sol, idxs=[net_compiled.hh.soma.v[1], net_compiled.hh.soma.v[4], net_compiled.hh.soma.v[2]],
          label=["Pre 1" "Pre 4" "Post 2"],
          title="Convergent 1→2 + 4→2", xlabel="Time", ylabel="V (mV)")

p3 = plot(sol, idxs=[net_compiled.syn_1to2.I_syn, net_compiled.syn_4to2.I_syn],
          label=["1→2" "4→2"], title="Individual synapse currents", xlabel="Time", ylabel="I_syn")

plot(p1, p2, p3, layout=(3,1), size=(800,750))
