
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
import MTKNeuralToolkit
#using script_utils.jl
using Plots

@named inp = TimeVaryingFunction(f = t -> ifelse((t > 10) & (t < 20),20, 0.0))

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




sys = build_network(connections, neurons)


prob = ODEProblem(sys, Pair[], (0.0, 200.0))


sol = solve(prob, Tsit5());

#Loss.membrane_mse(sys, sol, prob)
#arr1, arr2 = Loss.optim_test(sys, sol, prob)
#Loss.Forwardiff_test(sys, sol, prob)
#Loss.Zygote_test(sys, sol, prob)
loss_arr = Loss.MulitParamZygote_test(sys, sol, prob, neurons, ["g_max"], "ADAM", 5000)

plot(loss_arr, label="Loss", ylabel="Loss value")

