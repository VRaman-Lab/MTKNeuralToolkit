Pkg.activate(@__DIR__)
import Pkg; Pkg.add("OrdinaryDiffEqNonlinearSolve")
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using OrdinaryDiffEqNonlinearSolve
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using ModelingToolkit: t_nounits as t, D_nounits as D
import MTKNeuralToolkit.Types: SYNAPSE_TYPES, NEURON_TYPES, CustomSynapseParams
using MTKNeuralToolkit 
import MTKNeuralToolkit.IntegrateAndFire as IaF
import MTKNeuralToolkit.Synapse as Synapse
import MTKNeuralToolkit.Config as cfg
import MTKNeuralToolkit
#using script_utils.jl
using Plots
include("script_utils.jl")

@named inp = TimeVaryingFunction(f = t -> ifelse((t > 10) & (t < 40),10.0, 0.0))
@named inp2 = TimeVaryingFunction(f = t -> ifelse((t > 10) & (t < 40),40.0, 0.0))
neurons = Dict(
    "AB" => build_IF(inp;name=:AB),
    "BC" => build_IF(inp2;name=:BC),
    "CD" => build_IF(;name=:CD)
)
connections = Dict(
    ("AB", "CD") => (type=:LIF, weight=10.0),
    ("BC", "CD") => (type=:LIF, weight=10.0)
)
sys = build_network(connections, neurons)