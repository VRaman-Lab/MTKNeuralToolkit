import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using OrdinaryDiffEqNonlinearSolve
using SciMLStructures
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

@named inp = TimeVaryingFunction(f = t -> ifelse((t > 10.0) & (t < 20.0), 8.0, 0.0))

neurons = [

    build_LIF(inp; name=:L1N1),
    build_LIF(inp; name=:L1N2),
    build_LIF(inp; name=:L1N3),
    build_LIF(; name=:L2N1),
    build_LIF(; name=:L2N2),
    build_LIF(; name=:L2N3),
    build_LIF(; name=:L2N4),
    build_LIF(; name=:L3N1),
    build_LIF(; name=:L3N2),
    build_LIF(; name=:L4N1),
]

connections = Dict(
  
    (1, 4) => [(type=:LIF, weight=5.0)],
    (1, 5) => [(type=:LIF, weight=5.0)],
    (1, 6) => [(type=:LIF, weight=5.0)],
    (1, 7) => [(type=:LIF, weight=5.0)],
    (2, 4) => [(type=:LIF, weight=5.0)],
    (2, 5) => [(type=:LIF, weight=5.0)],
    (2, 6) => [(type=:LIF, weight=5.0)],
    (2, 7) => [(type=:LIF, weight=5.0)],
    (3, 4) => [(type=:LIF, weight=5.0)],
    (3, 5) => [(type=:LIF, weight=5.0)],
    (3, 6) => [(type=:LIF, weight=5.0)],
    (3, 7) => [(type=:LIF, weight=5.0)],
    # Layer 2 → Layer 3 (4×2)
    (4, 8)  => [(type=:LIF, weight=5.0)],
    (4, 9)  => [(type=:LIF, weight=5.0)],
    (5, 8)  => [(type=:LIF, weight=5.0)],
    (5, 9)  => [(type=:LIF, weight=5.0)],
    (6, 8)  => [(type=:LIF, weight=5.0)],
    (6, 9)  => [(type=:LIF, weight=5.0)],
    (7, 8)  => [(type=:LIF, weight=5.0)],
    (7, 9)  => [(type=:LIF, weight=5.0)],
    # Layer 3 → Layer 4 (2×1)
    (8, 10)  => [(type=:LIF, weight=5.0)],
    (9, 10)  => [(type=:LIF, weight=5.0)],
)

sys = build_network(Dict(connections), neurons)

prob = ODEProblem(sys, Pair[], (0.0, 200.0))

cb, spike_times = make_spike_callback(prob, neurons, ad_compatible=true)

tsteps = 0.0:0.1:200.0 

sol = solve(prob, Tsit5(); callback=cb, saveat = tsteps, abstol = 1e-8, reltol = 1e-6, dtmax = 0.1);

ground_weights = [
    1.0, 8.0, 2.0, 6.0,
    7.0, 14.0, 9.0, 5.0,
    3.0, 5.0, 2.0, 8.0,
    2.0, 7.0,
    9.0, 7.0,
    6.0, 4.0,
    7.0, 8.0,
    5.0, 9.0
]

ground_sol, ground_spike_times = GroundTruth.make_ground_truth(prob, neurons, ground_weights, tsteps, ad_sys=true)

p_ground, _, _ = SciMLStructures.canonicalize(Tunable(), ground_sol.prob.p)
p_ground = collect(p_ground)

param_syms = parameters(prob.f.sys)
p_array, params_idx, state_idx = Loss.get_parameters(prob, sys, ["g_max"], neurons)

for (i, idx) in enumerate(params_idx)
    println("$(param_syms[idx]) → Ground: $(round(p_ground[idx], digits=3))")
end


# Helper to plot all neurons on a given plot
function plot_all_neurons!(p, solution, sys)
    plot!(p, solution, idxs=[sys.L1N1.L1N1.oneport.v], label="L1N1")
    plot!(p, solution, idxs=[sys.L1N2.L1N2.oneport.v], label="L1N2")
    plot!(p, solution, idxs=[sys.L1N3.L1N3.oneport.v], label="L1N3")
    plot!(p, solution, idxs=[sys.L2N1.L2N1.oneport.v], label="L2N1")
    plot!(p, solution, idxs=[sys.L2N2.L2N2.oneport.v], label="L2N2")
    plot!(p, solution, idxs=[sys.L2N3.L2N3.oneport.v], label="L2N3")
    plot!(p, solution, idxs=[sys.L2N4.L2N4.oneport.v], label="L2N4")
    plot!(p, solution, idxs=[sys.L3N1.L3N1.oneport.v], label="L3N1")
    plot!(p, solution, idxs=[sys.L3N2.L3N2.oneport.v], label="L3N2")
    plot!(p, solution, idxs=[sys.L4N1.L4N1.oneport.v], label="L4N1")
end

loss_arr_BBO, ans_weights_BBO = Loss.MultiParamBBO(sys, prob, ground_sol, ground_spike_times, neurons,  ["g_max"], "BBO", 2000)
ans_sol_BBO, spike_times_BBO = GroundTruth.make_ground_truth(prob, neurons, ans_weights_BBO, tsteps)

loss_arr_fd, ans_weights_fd = Loss.MultiParamForward(sys, prob, ground_sol, ground_spike_times, neurons, ["g_max"], "ADAM", 4000)
ans_sol_fd, spike_times_fd = GroundTruth.make_ground_truth(prob, neurons, ans_weights_fd, tsteps)


loss_arr, ans_weights = Loss.MultiParamFinite(sys, prob, ground_sol, ground_spike_times, neurons, ["g_max"], "ADAM", 4000)
ans_sol, spike_times = GroundTruth.make_ground_truth(prob, neurons, ans_weights, tsteps)


best_loss = accumulate(min, loss_arr)
best_loss_BBO = accumulate(min, loss_arr_BBO)
best_loss_fd = accumulate(min, loss_arr_fd)


p1 = plot(ground_sol, idxs=[sys.L1N1.L1N1.oneport.v], label="L1N1", ylabel="Voltage (mV)", xlabel="t", title="Ground Truth")
plot_all_neurons!(p1, ground_sol, sys)

p3 = plot(ans_sol, idxs=[sys.L1N1.L1N1.oneport.v], label="L1N1", ylabel="Voltage (mV)", xlabel="t", title="After Optimisation (Finite Diff)")
plot_all_neurons!(p3, ans_sol, sys)

p4 = plot(ans_sol_BBO, idxs=[sys.L1N1.L1N1.oneport.v], label="L1N1", ylabel="Voltage (mV)", xlabel="t", title="After Optimisation (BBO)")
plot_all_neurons!(p4, ans_sol_BBO, sys)

p5 = plot(ans_sol_fd, idxs=[sys.L1N1.L1N1.oneport.v], label="L1N1", ylabel="Voltage (mV)", xlabel="t", title="After Optimisation (ForwardDiff)")
plot_all_neurons!(p5, ans_sol_fd, sys)

# Loss comparison
p6 = plot(1:length(best_loss), best_loss, label="Finite Diff", linewidth=1)
plot!(p6, 1:length(best_loss_BBO), best_loss_BBO, label="BBO", linewidth=1)
plot!(p6, 1:length(best_loss_fd), best_loss_fd, label="ForwardDiff", linewidth=1)
xlabel!(p6, "Epoch")
ylabel!(p6, "Loss")
title!(p6, "Optimiser Comparison")

display(plot(p1, p3, p4, p5, layout=(4, 1), size=(1800, 1800)))
display(p6)