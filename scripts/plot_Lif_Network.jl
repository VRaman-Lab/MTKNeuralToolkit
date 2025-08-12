import Pkg

Pkg.activate(@__DIR__)
import Pkg; Pkg.add("OrdinaryDiffEqNonlinearSolve")
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using DifferentialEquations
using OrdinaryDiffEq
using OrdinaryDiffEqNonlinearSolve
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
import MTKNeuralToolkit.Types: SYNAPSE_TYPES, NEURON_TYPES, CustomSynapseParams
using MTKNeuralToolkit 
import MTKNeuralToolkit.IntegrateAndFire as IaF
import MTKNeuralToolkit.Synapse as Syn
import MTKNeuralToolkit
#using script_utils.jl
using Plots
include("script_utils.jl")

@named inp = TimeVaryingFunction(f = t -> ifelse((t > 10) & (t < 40),0.0, 0.0))
@named inp2= TimeVaryingFunction(f = t -> ifelse((t > 10) & (t < 40),15.0, 0.0))
n1 = build_IF(inp; name = :n1)
n2 = build_IF(inp2; name = :n2)
n3 = build_IF(; name =:n3)

IF_synapse1 = Syn.LifSynapseComplex(;g_max=4, name = :IF_synapse1)
IF_synapse2 = Syn.LifSynapseComplex(;g_max=8, name = :IF_synapse2)

network = []

connection1 = make_lif_synapse(n1, n3, IF_synapse1; name = :connection1)
connection2 = make_lif_synapse(n2, n3, IF_synapse2; name = :connection2)

sys = build_LIF_network(connection1, connection2)
simple_network = structural_simplify(sys)

prob = ODEProblem(simple_network, Pair[], (0.0, 100.0))

sol  = solve(prob, Rodas5());
plot(sol, idxs=[connection1.IF_synapse1.v_pre])
plot!(sol, idxs=[connection2.IF_synapse2.v_pre])
plot!(sol, idxs=[n3.n3.v])