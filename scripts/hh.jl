using MTKNeuralToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq
using Plots

@named soma = Capacitor(C = 1.0)
@named stimulus_block = Blocks.Sine(frequency = 0.1, amplitude = 10.0)

hh_na_m = v -> (
    0.182 * (v + 35.0) / (1.0 - exp(-(v + 35.0) / 9.0)),   # alpha_m
    -0.124 * (v + 35.0) / (1.0 - exp((v + 35.0) / 9.0))    # beta_m
)

hh_na_h = v -> (
    0.25 * exp(-(v + 90.0) / 12.0),                        # alpha_h
    0.25 * (exp((v + 62.0) / 6.0)) / exp(-(v + 90.0) / 12.0) # beta_h
)

sodium_gates = [
    GateSpec(:m, 3, 0.0, hh_na_m),
    GateSpec(:h, 1, 0.0, hh_na_h)
]

hh_k_n = v -> (
    0.02 * (v - 25.0) / (1.0 - exp(-(v - 25.0) / 9.0)),     # alpha_n
    -0.002 * (v - 25.0) / (1.0 - exp((v - 25.0) / 9.0))    # beta_n
)

potassium_gates = [
    GateSpec(:n, 4, 0.0, hh_k_n)
]

@named sodium_channel = GenericChannel(g=120.0, E_rev=50.0, gates=sodium_gates)
@named potassium_channel = GenericChannel(g=36.0, E_rev=-77.0, gates=potassium_gates)
@named leak_channel = GenericChannel(g=0.3, E_rev=-54.4, gates=GateSpec[])

# This now uses the unified build_compartment
hh_neuron = build_compartment(soma, [sodium_channel, potassium_channel, leak_channel]; name = :hh_neuron, V_init=-65.0)

drivers = [
    (1, stimulus_block)
]

# This returns a Network struct
net = build_electrical_network([hh_neuron], []; drivers=drivers, name=:net)

# 4. Compile and solve the network system
net_compiled = mtkcompile(net.sys)

# No need for u0 dictionaries! V_init and GateSpec ICs are baked into the systems.
prob = ODEProblem(net_compiled, [], (0.0, 50.0))
sol = solve(prob, Rosenbrock23())

# Plot using the explicit interface variable we exposed
plot(sol, idxs=[net.sys.hh_neuron.p.v], title="Voltage trace", xlabel="Time", ylabel="Voltage (mV)")

