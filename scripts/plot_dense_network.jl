import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using ModelingToolkit: t_nounits as t, D_nounits as D
import MTKNeuralToolkit.Synapse as Synapse
import MTKNeuralToolkit.HodgkinHuxley as HH
import MTKNeuralToolkit.Liu as Liu
import MTKNeuralToolkit.Types: SYNAPSE_TYPES, NEURON_TYPES, CustomSynapseParams
using MTKNeuralToolkit
include("script_utils.jl")
using Plots

@named inp = TimeVaryingFunction(f=t -> min(log(t,10), 1.0))

@named n1 = build_Liu(inp; name=:n1)

#netw1 = make_dense_layer(25, :HH, :Exc; pre_neuron=n1)
#Todo, connect both ways :(

network = compose(ODESystem([], t; name=:network), [netw1])
network = structural_simplify(network)


prob = ODEProblem(network, Pair[], (0.0, 500.0) )
sol = solve(prob, TRBDF2());

plot(sol, idxs=parse_sol_for_membrane_voltages(sol), size=(1000, 800))
