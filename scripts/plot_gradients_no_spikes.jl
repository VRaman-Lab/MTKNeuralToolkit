import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using GLMakie
using ForwardDiff
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




function compute_loss_landscape(prob, neurons, tsteps, truth_vec, 
                                 param_idx, state_idx, ground_spike_times,
                                 p_base, sweep_idx1, sweep_idx2;
                                 range1=0.5:0.5:20.0, range2=0.5:0.5:20.0)
    loss_grid = zeros(length(range1), length(range2))
    
    for (i, v1) in enumerate(range1)
        for (j, v2) in enumerate(range2)
            p_test = copy(p_base)
            p_test[sweep_idx1] = v1
            p_test[sweep_idx2] = v2
            try
                val = Loss.lif_loss(prob, p_test, tsteps, param_idx, 
                                    state_idx, truth_vec, neurons, 
                                    ground_spike_times)
                loss_grid[i, j] = isnan(val) || isinf(val) ? -1.0 : val
            catch e
                println("Failed at ($v1, $v2): $e")
                loss_grid[i, j] = -1.0
            end
        end
        println("Row $i / $(length(range1)) done — min so far: $(minimum(loss_grid[1:i, :]))")
    end
    return loss_grid
end

function compute_forwarddiff_gradients(prob, neurons, tsteps, truth_vec,
                                        param_idx, state_idx, ground_spike_times,
                                        p_base, sweep_idx1, sweep_idx2,
                                        range1, range2)
    n1, n2 = length(range1), length(range2)
    grad_x = zeros(n1, n2)
    grad_y = zeros(n1, n2)
    
    for (i, v1) in enumerate(range1)
        for (j, v2) in enumerate(range2)
            p_test = copy(p_base)
            p_test[sweep_idx1] = v1
            p_test[sweep_idx2] = v2
            try
                g = ForwardDiff.gradient(p_test) do p
                    Loss.lif_loss(prob, p, tsteps, param_idx,
                                  state_idx, truth_vec, neurons,
                                  ground_spike_times)
                end
                grad_x[i, j] = g[sweep_idx1]
                grad_y[i, j] = g[sweep_idx2]
            catch
                grad_x[i, j] = 0.0
                grad_y[i, j] = 0.0
            end
        end
        println("ForwardDiff row $i / $n1 done")
    end
    return grad_x, grad_y
end

function compute_gradient_field(loss_grid, range1, range2)
    n1, n2 = size(loss_grid)
    grad_x = zeros(n1, n2)
    grad_y = zeros(n1, n2)
    
    dx = step(range1)
    dy = step(range2)
    
    for i in 2:(n1-1)
        for j in 2:(n2-1)
            grad_x[i, j] = (loss_grid[i+1, j] - loss_grid[i-1, j]) / (2 * dx)
            grad_y[i, j] = (loss_grid[i, j+1] - loss_grid[i, j-1]) / (2 * dy)
        end
    end
    return grad_x, grad_y
end




@named inp = TimeVaryingFunction(f = t -> ifelse((t > 10.0) & (t < 20.0), 3.0, 0.0))

neurons = [
    build_LIF(inp;name=:IF1),
    build_LIF(;name=:IF2),
    build_LIF(;name=:IF3),
    build_LIF(;name=:IF4)   
]
connections = Dict(
    (1, 2) => [(type=:LIF, weight = 1.0)],
    (2, 3) => [(type =:LIF, weight = 1.0)],
    (3, 4) => [(type =:LIF, weight = 1.0)],
)

sys = build_network(Dict(connections), neurons)
prob = ODEProblem(sys, Pair[], (0.0, 200.0))
tsteps = 0.0:0.1:200.0

cb, spike_times = make_spike_callback(prob, neurons, ad_compatible=false)
sol = solve(prob, Tsit5(); callback=cb, saveat=tsteps, abstol=1e-8, reltol=1e-6)

ground_sol, ground_spike_times = GroundTruth.make_ground_truth(prob, neurons, [0.8, 0.8, 0.8], tsteps, ad_sys=true)

p_array, params_idx, state_idx = Loss.get_parameters(prob, sys, ["g_max"], neurons)
truth_vec = Loss.get_truth_vectors(ground_sol, neurons, tsteps)
p_base = Float64[p_array[x] for x in params_idx]

range1 = 0.1:0.2:3.0
range2 = 0.1:0.2:3.0

loss_grid = compute_loss_landscape(prob, neurons, tsteps, truth_vec,
                                    params_idx, state_idx, ground_spike_times,
                                    p_base, 1, 2;
                                    range1=range1, range2=range2)

loss_clamped = clamp.(loss_grid, 0.0, 100.0)

fd_grad_x, fd_grad_y = compute_gradient_field(loss_clamped, range1, range2)
fwd_grad_x, fwd_grad_y = compute_forwarddiff_gradients(prob, neurons, tsteps, truth_vec,
                                                         params_idx, state_idx, ground_spike_times,
                                                         p_base, 1, 2, range1, range2)

skip = 2
idx1 = 2:skip:length(range1)-1
idx2 = 2:skip:length(range2)-1

x_pts = [range1[i] for i in idx1 for j in idx2]
y_pts = [range2[j] for i in idx1 for j in idx2]

fd_u = [-fd_grad_x[i, j] for i in idx1 for j in idx2]
fd_v = [-fd_grad_y[i, j] for i in idx1 for j in idx2]
fd_mag = sqrt.(fd_u.^2 .+ fd_v.^2)
scale = 0.3
fd_u_norm = scale .* fd_u ./ (fd_mag .+ 1e-8)
fd_v_norm = scale .* fd_v ./ (fd_mag .+ 1e-8)

fwd_u = [-fwd_grad_x[i, j] for i in idx1 for j in idx2]
fwd_v = [-fwd_grad_y[i, j] for i in idx1 for j in idx2]
fwd_mag = sqrt.(fwd_u.^2 .+ fwd_v.^2)
fwd_u_norm = scale .* fwd_u ./ (fwd_mag .+ 1e-8)
fwd_v_norm = scale .* fwd_v ./ (fwd_mag .+ 1e-8)

fig = Figure(size=(800, 700))
ax = Axis(fig[1, 1],
          xlabel="g_max₁", ylabel="g_max₂",
          title="Gradient Comparison (Non-Spiking): Finite Diff vs ForwardDiff")

GLMakie.contourf!(ax, collect(range1), collect(range2), loss_clamped',
                   colormap=:viridis, levels=20)

offset = 0.08

GLMakie.arrows!(ax, x_pts .- offset, y_pts .- offset, fd_u_norm, fd_v_norm,
                color=:black, linewidth=2.0, arrowsize=12)

GLMakie.arrows!(ax, x_pts .+ offset, y_pts .+ offset, fwd_u_norm, fwd_v_norm,
                color=:red, linewidth=2.0, arrowsize=12)

GLMakie.scatter!(ax, [0.8], [0.8],
                 color=:yellow, markersize=15, marker=:star5)

Legend(fig[1, 2], 
       [LineElement(color=:black, linewidth=2),
        LineElement(color=:red, linewidth=2),
        MarkerElement(color=:yellow, marker=:star5, markersize=15)],
       ["Finite Diff", "ForwardDiff", "Ground Truth"])

save("gradient_comparison_no_spikes.png", fig, px_per_unit=2)
display(fig)