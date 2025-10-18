
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
import MTKNeuralToolkit
#using script_utils.jl
using Plots

@named inp = TimeVaryingFunction(f = t -> ifelse((t > 10) & (t < 20),30.0, 0.0))
@named inp2 = TimeVaryingFunction(f = t ->  ifelse((t > 30) & (t < 40),30.0, 0.0))
neurons = [
    build_LIF(inp;name=:IF1),
    build_LIF(inp2;name=:IF2),
    build_LIF(;name=:IF3)
]
connections = Dict(
    (1, 3) => [(type=:LIF, weight=9.0)],
    (2, 3) => [(type=:LIF, weight=20.0)]
)
sys = build_network(connections, neurons)

prob = ODEProblem(sys, Pair[], (0.0, 100.0))
sol = solve(prob, Tsit5());

plot(sol, idxs=[sys.IF1.IF1.oneport.v], label="Neuron One", ylabel="Voltage(V)")
plot!(sol, idxs=[sys.IF2.IF2.oneport.v], label="Neuron Two", xlabel="Time(ms)")
plot!(sol, idxs=[sys.IF3.IF3.oneport.v], label="Neuron Three", xlabel="Time(ms)")