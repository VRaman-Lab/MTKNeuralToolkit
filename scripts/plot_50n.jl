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

@named inp = TimeVaryingFunction(f = t -> ifelse((t > 10.0) & (t < 30.0), 5.0, 0.0))

# Layer sizes: 5 → 8 → 10 → 10 → 8 → 5 → 3 → 1 = 50 neurons
layer_sizes = [5, 8, 10, 10, 8, 5, 3, 1]

# Build neurons
neurons = []
for (l, size) in enumerate(layer_sizes)
    for n in 1:size
        name = Symbol("L$(l)N$(n)")
        if l == 1
            push!(neurons, build_LIF(inp; name=name))
        else
            push!(neurons, build_LIF(; name=name))
        end
    end
end

# Build fully-connected connections between adjacent layers
connections = Dict{Tuple{Int,Int}, Vector{@NamedTuple{type::Symbol, weight::Float64}}}()
for l in 1:(length(layer_sizes)-1)
    src_size = layer_sizes[l]
    dst_size = layer_sizes[l+1]
    src_offset = l == 1 ? 0 : sum(layer_sizes[1:l-1])
    dst_offset = sum(layer_sizes[1:l])
    for s in 1:src_size
        for d in 1:dst_size
            src_idx = src_offset + s
            dst_idx = dst_offset + d
            connections[(src_idx, dst_idx)] = [(type=:LIF, weight=0.5)]  # was 0.3
        end
    end
end

sys = build_network(connections, neurons)

prob = ODEProblem(sys, Pair[], (0.0, 500.0))

cb, spike_times = make_spike_callback(prob, neurons, ad_compatible=true)

tsteps = 0.0:0.1:500.0 

sol = solve(prob, Tsit5(); callback=cb, saveat=tsteps, abstol=1e-8, reltol=1e-6, dtmax=0.1)

# Ground truth weights
n_connections = sum(layer_sizes[l] * layer_sizes[l+1] for l in 1:length(layer_sizes)-1)
println("Total connections: $n_connections")

using Random
Random.seed!(42)
ground_weights = round.(rand(n_connections) .* 3.0 .+ 0.5, digits=2)  # range 0.5 to 3.5

ground_sol, ground_spike_times = GroundTruth.make_ground_truth(prob, neurons, ground_weights, tsteps, ad_sys=true)

# Helper to plot one layer
function plot_layer!(p, solution, sys, layer_num, layer_size, color)
    for n in 1:layer_size
        name = Symbol("L$(layer_num)N$(n)")
        accessor = getproperty(sys, name)
        accessor2 = getproperty(accessor, name)
        plot!(p, solution, idxs=[accessor2.oneport.v],
              label=(n == 1 ? "Layer $layer_num" : false),
              color=color, linewidth=0.8, alpha=0.7)
    end
end

# Plot ground truth by layer
colors = [:blue, :red, :green, :purple, :orange, :teal, :pink, :gray]

ground_plots = []
for (l, size) in enumerate(layer_sizes)
    pl = plot(title="Ground truth — Layer $l ($size neurons)", ylabel="V (mV)", xlabel="t")
    plot_layer!(pl, ground_sol, sys, l, size, colors[l])
    push!(ground_plots, pl)
end
display(plot(ground_plots..., layout=(length(layer_sizes), 1), size=(1200, 1200)))

# Run optimisers


loss_arr_BBO, ans_weights_BBO = Loss.MultiParamForward(sys, prob, ground_sol, ground_spike_times, neurons, ["g_max"], "ADAM", 2000)
ans_sol_BBO, _ = GroundTruth.make_ground_truth(prob, neurons, ans_weights_BBO, tsteps)

# Plot each optimiser by layer
for (label, opt_sol) in [("BBO", ans_sol_BBO)]
    opt_plots = []
    for (l, size) in enumerate(layer_sizes)
        pl = plot(title="$label — Layer $l ($size neurons)", ylabel="V (mV)", xlabel="t")
        plot_layer!(pl, opt_sol, sys, l, size, colors[l])
        push!(opt_plots, pl)
    end
    display(plot(opt_plots..., layout=(length(layer_sizes), 1), size=(1200, 1200)))
end

# Loss comparison

best_loss_BBO = accumulate(min, loss_arr_BBO)


p_loss = plot(1:length(best_loss_BBO), best_loss_BBO, label="BBO", linewidth=2)
xlabel!(p_loss, "Epoch")
ylabel!(p_loss, "Loss")
title!(p_loss, "Optimiser Comparison (50 neurons, 358 parameters)")
display(p_loss)