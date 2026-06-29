using MTKNeuralToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq
using Plots

N = 30

@named soma = Capacitor(N=N, C = 1.0)

# Broadcasting dots work natively for both scalar and array tracing
hh_na_m = v -> (
    0.182 .* (v .+ 35.0) ./ (1.0 .- exp.(-(v .+ 35.0) ./ 9.0)),
    -0.124 .* (v .+ 35.0) ./ (1.0 .- exp.((v .+ 35.0) ./ 9.0))
)
hh_na_h = v -> (
    0.25 .* exp.(-(v .+ 90.0) ./ 12.0),
    0.25 .* (exp.((v .+ 62.0) ./ 6.0)) ./ exp.(-(v .+ 90.0) ./ 12.0)
)

sodium_gates = [
    GateSpec(:m, 3, 0.0, hh_na_m),
    GateSpec(:h, 1, 0.0, hh_na_h)
]

hh_k_n = v -> (
    0.02 .* (v .- 25.0) ./ (1.0 .- exp.(-(v .- 25.0) ./ 9.0)),
    -0.002 .* (v .- 25.0) ./ (1.0 .- exp.((v .- 25.0) ./ 9.0))
)

potassium_gates = [
    GateSpec(:n, 4, 0.0, hh_k_n)
]

@named sodium_channel = GenericChannel(N=N, g=120.0, E_rev=50.0, gates=sodium_gates)
@named potassium_channel = GenericChannel(N=N, g=36.0, E_rev=-77.0, gates=potassium_gates)
@named leak_channel = GenericChannel(N=N, g=0.3, E_rev=-54.4, gates=GateSpec[])

hh_neurons = build_compartment(soma, [sodium_channel, potassium_channel, leak_channel]; name = :hh_neurons, V_init=-65.0, N=N)

# Just pass a Julia array directly as the driver!
drivers = [(1, 1:30)]

net = build_acausal_network([hh_neurons], [],[]; drivers=drivers, N=N, name=:net)

net_compiled = mtkcompile(net.sys)
prob = ODEProblem(net_compiled, [], (0.0, 50.0))
sol = solve(prob, Rosenbrock23())

plot(sol, idxs=[net_compiled.hh_neurons.soma.v...], title="Unified Vectorized HH Network", xlabel="Time", ylabel="Voltage (mV)")
