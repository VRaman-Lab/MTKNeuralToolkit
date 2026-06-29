using MTKNeuralToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq
using Plots

# 1. Instantiate the single compartment architecture without a built-in driver
@named soma = SpikingCapacitor(C = 1.0)
@named stimulus_block = Blocks.Sine(frequency = 0.1, amplitude = 12.2)
leak = build_channel(lgates(name=:gate), FixedReversal(E = -54.4, name=:batt); name=:leak)
LIF = build_compartment(soma, [leak]; name = :lif)

# 2. Map the driver to index 1 (the single neuron in our upcoming network array)
drivers = [
    (1, stimulus_block)
]

# 3. Wrap it inside a 1-neuron explicit circuit network to handle the input mapping
@named net = build_acausal_network([LIF], []; drivers=drivers)

# 4. Compile and solve the balanced system
LIFc = mtkcompile(net)
prob = ODEProblem(LIFc, [], (0.0, 50.0))
sol = solve(prob, Tsit5())

# 5. Plot using the network hierarchical path
plot(sol, idxs=[net.lif.V], title="Voltage trace", xlabel="Time", ylabel="Voltage (mV)")
#



