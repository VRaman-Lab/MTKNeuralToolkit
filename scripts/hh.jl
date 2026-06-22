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

hh_neuron = build_compartment(soma, [sodium, potassium, leak]; name = :hh_neuron)

# 2. Setup the driver mapping (Neuron 1 gets the stimulus_block)
drivers = [
    (1, stimulus_block)
]

# 3. Wrap it in a 1-neuron explicit circuit network
# The builder handles all the boundary math and connector balancing automatically!
@named net = build_electrical_network([hh_neuron], []; drivers=drivers)

# 4. Compile and solve the network system
net_compiled = mtkcompile(net)
prob = ODEProblem(net_compiled, [], (0.0, 50.0))
sol = solve(prob, Rosenbrock23())

plot(sol, idxs=[net.hh_neuron.V], title="Voltage trace", xlabel="Time", ylabel="Voltage (mV)")
# plot(sol, idxs = [potassium.gate.v])
