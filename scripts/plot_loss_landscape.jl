using GLMakie
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

# Sweep two g_max parameters while keeping the third fixed
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

sol = solve(prob, Tsit5(); callback=cb, saveat = tsteps, abstol = 1e-8,reltol = 1e-6);


# Setup
tsteps = 0.0:0.1:200.0
ground_sol, ground_spike_times = GroundTruth.make_ground_truth(prob, neurons, [10.0, 10.0, 10.0], tsteps, ad_sys=true)
p_array, params_idx, state_idx = Loss.get_parameters(prob, sys, ["g_max"], neurons)
truth_vec = Loss.get_truth_vectors(ground_sol, neurons, tsteps)
p_base = Float64[p_array[x] for x in params_idx]


# Sweep first two parameters
range1 = 1.0:2.0:20.0
range2 = 1.0:2.0:20.0

loss_grid = compute_loss_landscape(prob, neurons, tsteps, truth_vec,
                                    params_idx, state_idx, ground_spike_times,
                                    p_base, 1, 2;
                                    range1=range1, range2=range2)

# Clamp extreme values for better visualisation
loss_clamped = clamp.(loss_grid, 0.0, 100.0)

fig = Figure(size=(800, 600))
ax = Axis3(fig[1, 1], 
           xlabel="g_max₁", ylabel="g_max₂", zlabel="Loss",
           title="Loss Landscape")

GLMakie.surface!(ax, collect(range1), collect(range2), loss_clamped',
         colormap=:viridis)

# Mark ground truth
GLMakie.scatter!(ax, [10.0], [10.0], [0.0], 
         color=:red, markersize=20)

display(fig)

save("loss_landscape.png", fig, px_per_unit=2)