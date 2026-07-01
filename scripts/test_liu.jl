using MTKNeuralToolkit
using MTKNeuralToolkit.LiuCalciumNeuron
using ModelingToolkit: mtkcompile
using OrdinaryDiffEq, Plots

# Build the neuron using the module
comp = LiuCalciumNeuron.build_liu_neuron()

# Kick it off with a driver
drivers = [(1, 10.0)]
net = build_acausal_network([comp]; drivers=drivers)
net_compiled = mtkcompile(net.sys)

prob = ODEProblem(net_compiled, [], (0.0, 500.0))
sol = solve(prob, Rosenbrock23())

# Plot
p1 = plot(sol, idxs=net_compiled.Liu_AB_Neuron.cap.v, title="Membrane Potential")
p2 = plot(sol, idxs=net_compiled.Liu_AB_Neuron.Liu_AB_Neuron_ca_pool.Ca, title="Calcium")
plot(p1, p2, layout=(2,1))
