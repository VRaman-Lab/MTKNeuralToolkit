using Plots
using Statistics: mean
using ForwardDiff
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
using SciMLStructures: Tunable, canonicalize, replace, replace!

sigmoid(x) = 1 / (1 + exp(-x))

@named inp = TimeVaryingFunction(f = t -> ifelse((t > 10.0) & (t < 20.0), 14.0, 0.0))
neurons = [
    build_LIF(inp; name=:IF1),
    build_LIF(; name=:IF2),
    build_LIF(; name=:IF3),
    build_LIF(; name=:IF4)
]
connections = Dict(
    (1, 2) => [(type=:LIF, weight = 7.0)],
    (2, 3) => [(type=:LIF, weight = 7.0)],
    (3, 4) => [(type=:LIF, weight = 7.0)],
)
sys = build_network(Dict(connections), neurons)
prob = ODEProblem(sys, Pair[], (0.0, 200.0))
tsteps = 0.0:0.1:200.0

ground_sol, ground_spike_times = GroundTruth.make_ground_truth(
    prob, neurons, [8.0, 8.0, 8.0], tsteps)

ground_state_syms = unknowns(ground_sol.prob.f.sys)
truth_vec = []
for n in neurons
    nm = string(nameof(n))
    pattern = nm * "₊" * nm * "₊oneport₊v"
    gi = findfirst(s -> contains(string(s), pattern), ground_state_syms)
    push!(truth_vec, ground_sol(tsteps)[gi, :])
end

p_array, params_idx, state_idx = Loss.get_parameters(prob, sys, ["g_max"], neurons)

# ── Surrogate gradient loss (the original broken one) ─────────
function soft_spike_train(spike_times_vec, tsteps; σ=2.0)
    signal = zeros(Float64, length(tsteps))
    for t_spike in spike_times_vec
        signal .+= exp.(-(collect(tsteps) .- t_spike).^2 ./ (2σ^2))
    end
    return signal
end

function surrogate_loss(prob, x, tsteps, params_idx, state_idx, 
                        truth_vec, neurons, ground_spike_times)
    p_tunable, replace_p, _ = canonicalize(Tunable(), prob.p)
    p_tunable = collect(p_tunable)
    p_new = [i in params_idx ? x[findfirst(==(i), params_idx)] : p_tunable[i]
             for i in eachindex(p_tunable)]
    newprob = remake(prob; p = replace_p(p_new))
    cb, _ = make_spike_callback(newprob, neurons)
    sol = solve(newprob, Tsit5(); callback=cb, saveat=tsteps,
                abstol=1e-8, reltol=1e-6, verbose=false)

    total = 0.0
    threshold = -55.0
    σ_detect = 20.0
    v_min = -75.0
    v_range = 15.0

    for (i, neuron_state_i) in enumerate(state_idx)
        pred = [u[neuron_state_i] for u in sol.u]
        truth = truth_vec[i]

        sub_mask = (pred .< threshold) .& (truth .< threshold)
        if sum(sub_mask) > 0
            sub_pred = pred[sub_mask]
            sub_truth = truth[sub_mask]
            total += mean(abs2.((sub_pred .- v_min) ./ v_range .-
                                (sub_truth .- v_min) ./ v_range))
        end

        # Sigmoid surrogate spike detector
        pred_spike = sigmoid.((pred .- threshold) .* σ_detect)
        ground_kernel = soft_spike_train(ground_spike_times[i], tsteps; σ=3.0)

        pred_scaled = pred_spike ./ (maximum(pred_spike) + 1e-8)
        ground_scaled = ground_kernel ./ (maximum(ground_kernel) + 1e-8)

        # Spike shape loss
        total += mean(abs2.(pred_scaled .- ground_scaled))

    end
    return total 
end

# ── Smoothed trace MSE (your working version) ────────────────
function smoothed_mse(prob, x, tsteps, params_idx, state_idx,
                      truth_vec, neurons)
    p_tunable, replace_p, _ = canonicalize(Tunable(), prob.p)
    p_tunable = collect(p_tunable)
    p_new = [i in params_idx ? x[findfirst(==(i), params_idx)] : p_tunable[i]
             for i in eachindex(p_tunable)]
    newprob = remake(prob; p = replace_p(p_new))
    cb, _ = make_spike_callback(newprob, neurons)
    sol = solve(newprob, Tsit5(); callback=cb, saveat=tsteps,
                abstol=1e-8, reltol=1e-6, verbose=false)

    total = 0.0
    dt = tsteps[2] - tsteps[1]
    τ = 15.0
    α = dt / τ
    n = length(tsteps)

    for (i, si) in enumerate(state_idx)
        pred = [u[si] for u in sol.u]
        truth = truth_vec[i]

        pred_fwd = similar(pred, Float64)
        truth_fwd = similar(truth, Float64)
        pred_fwd[1] = pred[1]; truth_fwd[1] = truth[1]
        for k in 2:n
            pred_fwd[k] = α * pred[k] + (1 - α) * pred_fwd[k-1]
            truth_fwd[k] = α * truth[k] + (1 - α) * truth_fwd[k-1]
        end

        pred_bwd = similar(pred, Float64)
        truth_bwd = similar(truth, Float64)
        pred_bwd[n] = pred[n]; truth_bwd[n] = truth[n]
        for k in (n-1):-1:1
            pred_bwd[k] = α * pred[k] + (1 - α) * pred_bwd[k+1]
            truth_bwd[k] = α * truth[k] + (1 - α) * truth_bwd[k+1]
        end

        smooth_pred = (pred_fwd .+ pred_bwd) ./ 2
        smooth_truth = (truth_fwd .+ truth_bwd) ./ 2
        total += mean(abs2.(smooth_pred .- smooth_truth))
    end
    return total
end

# ── Sweep ─────────────────────────────────────────────────────
w_range = 5.0:0.1:12.0
surr_1d = Float64[]
smooth_1d = Float64[]

for w in w_range
    x = fill(w, length(params_idx))
    push!(surr_1d, surrogate_loss(prob, x, tsteps, params_idx, state_idx,
                                   truth_vec, neurons, ground_spike_times))
    push!(smooth_1d, smoothed_mse(prob, x, tsteps, params_idx, state_idx,
                                   truth_vec, neurons))
    println("w=$w done")
end

# Find where each loss has its minimum
# Normalise surrogate loss to 0-10 range for fair comparison
surr_min = minimum(surr_1d)
surr_max = maximum(surr_1d)
surr_normalised = 10.0 .* (surr_1d .- surr_min) ./ (surr_max - surr_min)

p1 = plot(collect(w_range), surr_normalised, linewidth=2, color=:red, 
          label="Surrogate (normalised)")
plot!(p1, collect(w_range), smooth_1d, linewidth=2, color=:steelblue, 
      label="Smoothed MSE")
vline!(p1, [8.0], linestyle=:dash, color=:black, label="Ground Truth", linewidth=2)
ylabel!(p1, "Loss (normalised)")
xlabel!(p1, "g_max")
title!(p1, "Normalised Comparison")