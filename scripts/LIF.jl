using MTKNeuralToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq
using Plots

@named soma = LIFCapacitor(C = 1.0)
@named stimulus_block = Blocks.Sine(frequency = 0.1, amplitude = 12.2)
leak      = build_channel(lgates(name=:gate), FixedReversal(E = -54.4, name=:batt); name=:leak)
LIF = build_compartment(soma, [leak]; stimulus_block = stimulus_block, name = :lif)
LIFc = mtkcompile(LIF)
prob = ODEProblem(LIFc, [], (0.0, 50.0))
sol = solve(prob, Tsit5())

plot(sol, idxs=[soma.V], title="Voltage trace", xlabel="Time", ylabel="Voltage (mV)")
#



