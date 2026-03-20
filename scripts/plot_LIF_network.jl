
using ModelingToolkit
using OrdinaryDiffEq
using OrdinaryDiffEqNonlinearSolve
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using ModelingToolkit: t_nounits as t, D_nounits as D
using SymbolicIndexingInterface 
import MTKNeuralToolkit.Types: SYNAPSE_TYPES, NEURON_TYPES, CustomSynapseParams
using MTKNeuralToolkit 
import MTKNeuralToolkit.IntegrateAndFire as IaF
import MTKNeuralToolkit.HodgkinHuxley as HH
import MTKNeuralToolkit.Synapse as Synapse
import MTKNeuralToolkit.Config as cfg
import MTKNeuralToolkit.Loss as loss
import MTKNeuralToolkit
#using script_utils.jl
using Plots


@named inp = TimeVaryingFunction(f = t -> ifelse((t > 10) & (t < 20),10, 0.0))
@named inp2 = TimeVaryingFunction(f = t ->  ifelse((t > 20) & (t < 30),20.0, 0.0))
neurons = [
    build_LIF(inp;name=:IF1),
    build_LIF(;name=:IF2),
    build_LIF(;name=:IF3),
    build_LIF(;name=:IF4)

   
]
connections = Dict(
    (1, 2) => [(type=:LIF, weight = 5.0)],
    (2, 3) => [(type =:LIF, weight = 2.0)],
    (3, 4) => [(type =:LIF, weight = 2.0)],
)


sys = build_network(connections, neurons)


build_start = time()
@time prob = ODEProblem(sys, Pair[], (0.0, 100.0))
build_end = time() 

solve_start = time()
@time sol = solve(prob, Tsit5());
solve_end = time()

outputs_neurons = ["IF1", "IF2"]

#test = loss.membrane_mse(sys, sol, outputs_neurons)

plot(sol, idxs=[sys.IF1.IF1.oneport.v], label="Neuron One", ylabel="Voltage(V)")
plot!(sol, idxs=[sys.IF2.IF2.oneport.v], label="Neuron Two", xlabel="Time(ms)")
plot!(sol, idxs=[sys.IF3.IF3.oneport.v], label="Neuron Three", xlabel="Time(ms)")
plot!(sol, idxs=[sys.IF4.IF4.oneport.v], label="Neuron Three", xlabel="Time(ms)")



