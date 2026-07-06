using MTKNeuralToolkit
using MTKNeuralToolkit.PrinzNeuron
using ModelingToolkit: mtkcompile
using OrdinaryDiffEq
using Plots

# Build the neuron
comp = build_prinz_neuron()

# Kick it off with a driver
drivers = [(1, 10.0)]
net = build_acausal_network([comp]; drivers=drivers)
net_compiled = mtkcompile(net.sys)

prob = ODEProblem(net_compiled, [], (0.0, 1000.0))
sol = solve(prob, Rosenbrock23())

# Plot
p1 = plot(sol, idxs=net_compiled.Prinz_Neuron.cap.v, title="Membrane Potential", legend=false)
p2 = plot(sol, idxs=net_compiled.Prinz_Neuron.Prinz_Neuron_ca_pool.Ca, title="Calcium Concentration", legend=false)

plot(p1, p2, layout=(2,1), size=(800,500))
