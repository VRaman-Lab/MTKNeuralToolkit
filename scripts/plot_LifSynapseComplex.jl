import Pkg

Pkg.activate(@__DIR__)
import Pkg; Pkg.add("OrdinaryDiffEqNonlinearSolve")
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using OrdinaryDiffEqNonlinearSolve
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
import MTKNeuralToolkit.Types: SYNAPSE_TYPES, NEURON_TYPES
using MTKNeuralToolkit 
import MTKNeuralToolkit.IntegrateAndFire as IaF
import MTKNeuralToolkit.Synapse as Syn
import MTKNeuralToolkit
#using script_utils.jl
using Plots


@named inp = TimeVaryingFunction(f = t -> ifelse((t > 10) & (t < 20),50.0, 0.0))
n1 = build_IF(inp; name = :n1)
n2 = build_IF(; name =:n2)

IF_synapse = Syn.LifSynapseComplex(;g_max=10, name = :IF_synapse)

network = make_lif_synapse(n1, n2, IF_synapse; name = :system)
simple_network = structural_simplify(network)

prob = ODEProblem(simple_network, Pair[], (0.0, 100.0))

sol  = solve(prob, Rodas5());
plot(sol, idxs=[IF_synapse.v_pre])
plot!(sol, idxs=[IF_synapse.v_post])

