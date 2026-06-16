using MTKNeuralToolkit
# using ModelingToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq
using Plots


@named soma = Capacitor(C = 1.0)
@named stimulus_block = Blocks.Sine(frequency = 0.1, amplitude = 10.0)

sodium    = build_channel(nagates(name=:gate), FixedReversal(E = 50.0, name=:batt); name=:sodium)
potassium = build_channel(kgates(name=:gate), FixedReversal(E = -77.0, name=:batt); name=:potassium)
leak      = build_channel(lgates(name=:gate), FixedReversal(E = -54.4, name=:batt); name=:leak)

hh_neuron = build_compartment(soma, [sodium, potassium, leak]; stimulus_block = stimulus_block, name = :hh_neuron)

hh_compiled = mtkcompile(hh_neuron)
prob = ODEProblem(hh_compiled, [], (0.0, 50.0))
sol = solve(prob, Rosenbrock23())

plot(sol, idxs=[soma.V], title="Voltage trace", xlabel="Time", ylabel="Voltage (mV)")
# plot(sol, idxs = [potassium.gate.v])
