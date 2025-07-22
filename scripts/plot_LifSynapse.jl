import Pkg

Pkg.activate(@__DIR__)
import Pkg; Pkg.add("OrdinaryDiffEqNonlinearSolve")
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using OrdinaryDiffEqNonlinearSolve
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using MTKNeuralToolkit 
import MTKNeuralToolkit.IntegrateAndFire as IaF
import MTKNeuralToolkit.Synapse as Syn
import MTKNeuralToolkit
#using script_utils.jl
using Plots

IF = build_channel(IaF.IF_channel(; E=-65, name = :conductance), FixedReversal(; E=-65); name = :IF)
IF2 = build_channel(IaF.IF_channel(; E=-65, name = :conductance), FixedReversal(; E=-65); name = :IF)

@named stim = TimeVaryingFunction(f = t -> ifelse((t > 10) & (t < 20), 100.0, 0.0))
@named stim2 = TimeVaryingFunction(f = t -> 0.0)

soma1 = BasicSoma(; C=10, name=:soma)
soma2 = BasicSoma(; C=10, name=:soma)

n1 = build_neuron(soma1, stim; channels = [IF])
n2 = build_neuron(soma2, stim2; channels = [IF2])

@named SynapseLif = Syn.LifSynapse()

model = make_lif_synapse(n1, n2, SynapseLif; name = :model)

model_simp = structural_simplify(model)