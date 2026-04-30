import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using OrdinaryDiffEqNonlinearSolve
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using ModelingToolkit: t_nounits as t, D_nounits as D
import MTKNeuralToolkit.Types: SYNAPSE_TYPES, NEURON_TYPES, CustomSynapseParams
using MTKNeuralToolkit 
import MTKNeuralToolkit.IntegrateAndFire as IaF
import MTKNeuralToolkit.HodgkinHuxley as HH
import MTKNeuralToolkit.Synapse as Synapse
import MTKNeuralToolkit.Config as cfg
import MTKNeuralToolkit.TestLoss as Loss
import MTKNeuralToolkit.GroundTruth as GroundTruth
using Plots
using CUDA
using DiffEqGPU  
using SciMLStructures: Tunable, canonicalize, replace, replace!

@named inp = TimeVaryingFunction(f = t -> ifelse((t > 10.0) & (t < 20.0), 16.0, 0.0))


neurons = [
    build_LIF(inp;name=:IF1),
    build_LIF(;name=:IF2),
    build_LIF(;name=:IF3),
    build_LIF(;name=:IF4)   
]
connections = Dict(
    (1, 2) => [(type=:LIF, weight = 11.0)],
    (2, 3) => [(type =:LIF, weight = 11.0)],
    (3, 4) => [(type =:LIF, weight = 11.0)],
)

sys = build_network(Dict(connections), neurons)

prob = ODEProblem(sys, Pair[], (0.0, 200.0))

cb, spike_times = make_spike_callback(prob, neurons, ad_compatible=false)

tsteps = 0.0:0.1:200.0 

sol = solve(prob, Tsit5(); callback=cb, saveat = tsteps, abstol = 1e-8,reltol = 1e-6);

#Loss.membrane_mse(sys, sol, prob)
#arr1, arr2 = Loss.optim_test(sys, sol, prob)
#Loss.Forwardiff_test(sys, sol, prob)
#Loss.Zygote_test(sys, sol, prob)
ground_sol, ground_spike_times = GroundTruth.make_ground_truth(prob, neurons, [10.0, 10.0, 10.0], tsteps, ad_sys=false)



loss_arr_BBO, ans_weights_BBO = Loss.MultiParamBBO(sys, prob, ground_sol, ground_spike_times, neurons,  ["g_max"], "BBO", 2000)
ans_sol_BBO, spike_times_BBO = GroundTruth.make_ground_truth(prob, neurons, ans_weights_BBO, tsteps)


loss_arr_fd, ans_weights_fd = Loss.MultiParamForward(sys, prob, ground_sol, ground_spike_times, neurons,  ["g_max"], "ADAM", 2000)
println(ans_weights_fd)
ans_sol_fd, spike_times_df = GroundTruth.make_ground_truth(prob, neurons, ans_weights_fd, tsteps)

loss_arr, ans_weights = Loss.MultiParamFinite(sys, prob, ground_sol, ground_spike_times, neurons,  ["g_max"], "ADAM", 2000)
println(ans_weights)
ans_sol, spike_times = GroundTruth.make_ground_truth(prob, neurons, ans_weights, tsteps)

best_loss = accumulate(min, loss_arr_BBO)
best_loss_forward = accumulate(min, loss_arr_fd)
best_loss_finite = accumulate(min, loss_arr)

p1 = plot(ground_sol, idxs=[sys.IF1.IF1.oneport.v], label="Neuron 1", ylabel="Voltage (mV)", xlabel="t", title="Ground Truth")
plot!(p1, ground_sol, idxs=[sys.IF2.IF2.oneport.v], label="Neuron 2")
plot!(p1, ground_sol, idxs=[sys.IF3.IF3.oneport.v], label="Neuron 3")
plot!(p1, ground_sol, idxs=[sys.IF4.IF4.oneport.v], label="Neuron 4")

p2 = plot(ans_sol, idxs=[sys.IF1.IF1.oneport.v], label="Neuron 1", ylabel="Voltage (mV)", xlabel="t", title="After Optimisation (Finite Diff)")
plot!(p2, ans_sol, idxs=[sys.IF2.IF2.oneport.v], label="Neuron 2")
plot!(p2, ans_sol, idxs=[sys.IF3.IF3.oneport.v], label="Neuron 3")
plot!(p2, ans_sol, idxs=[sys.IF4.IF4.oneport.v], label="Neuron 4")

p3 = plot(ans_sol_BBO, idxs=[sys.IF1.IF1.oneport.v], label="Neuron 1", ylabel="Voltage (mV)", xlabel="t", title="After Optimisation (BBO)")
plot!(p3, ans_sol_BBO, idxs=[sys.IF2.IF2.oneport.v], label="Neuron 2")
plot!(p3, ans_sol_BBO, idxs=[sys.IF3.IF3.oneport.v], label="Neuron 3")
plot!(p3, ans_sol_BBO, idxs=[sys.IF4.IF4.oneport.v], label="Neuron 4")

p4 = plot(ans_sol_fd, idxs=[sys.IF1.IF1.oneport.v], label="Neuron 1", ylabel="Voltage (mV)", xlabel="t", title="After Optimisation (ForwardDiff)")
plot!(p4, ans_sol_fd, idxs=[sys.IF2.IF2.oneport.v], label="Neuron 2")
plot!(p4, ans_sol_fd, idxs=[sys.IF3.IF3.oneport.v], label="Neuron 3")
plot!(p4, ans_sol_fd, idxs=[sys.IF4.IF4.oneport.v], label="Neuron 4")


p5 = plot(1:length(loss_arr), best_loss_finite, label="Finite Diff", linewidth=1)
plot!(p5, 1:length(loss_arr_BBO), best_loss, label="BBO", linewidth=1)
plot!(p5, 1:length(loss_arr_fd), best_loss_forward, label="ForwardDiff", linewidth=1)
xlabel!(p5, "Epoch")
ylabel!(p5, "Loss")
title!(p5, "Optimiser Comparison")

display(plot(p1, p2, p3, p4, layout=(4, 1), size=(1200, 1200)))
plot(p5)