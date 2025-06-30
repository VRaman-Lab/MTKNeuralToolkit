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
#include("script_types.jl")
using Plots

#Create your inputs here. Different options include inbuilt Julia functions ->Sin, cos, exp, log, etc. For log you need to specify base.
#Not every neuron needs an input; create prebuilt input neurons by specifying their input functions, other prebuilt neurons through an int param.
#Neurons are added to a dict depending on when they were created, which depends on their location within the build_network args.
#If 3 input HHs and 1 input Liu, the first non-input HH will be n4, the first non-input LIF will be n$(all_previously_made_neurons+1)

@named inp = TimeVaryingFunction(f=t -> min(log(t,10), 1.0))
@named inp2 = TimeVaryingFunction(f=t -> exp(sin(t)))
#--Workflow 1
connections = Dict(
    ("n1", "n3") => (type=:Exc, weight=0.6),
    ("n1", "n4") => (type=:Exc, weight=0.6),
    ("n1", "n5") => (type=:Exc, weight=1.0),
    ("n3", "n6") => (type=:Exc, weight=10.0),
    ("n4", "n6") => (type=:Exc, weight=10.0),
    ("n5", "n6") => (type=:Inh, weight=0.2),
    ("n6", "n5") => (type=:Inh, weight=0.2),
    ("n6", "n2") => (type=:Exc, weight=1.0),
    ("n2", "n5") => (type=:Inh, weight=1.0),
    ("n2", "n6") => (type=:Inh, weight=1.0)
)
network = build_network(connections, inpHH=[inp, inp2], noinpHH=5)
#--Workflow 2
#=
@named n1 = build_Liu(inp; name=:n1)
@named n2 = build_HH(;name=:n2)

s1 = put_synapse(n1,n2,:Exc,0.1; name=:s1)
network = compose(ODESystem([], t; name=:network), [s1])
network = structural_simplify(network)=#
#If using Liu neurons change solver from Tsit5 -> TRBDF2. 
#Might need to manually give more maxiters as well through solve(x,y, maxiters=[more lol])

prob = ODEProblem(network, Pair[], (0.0, 500.0) )
#inspect_network(network)
sol = solve(prob, TRBDF2());

plot(sol, idxs=parse_sol_for_membrane_voltages(sol), size=(1000, 800))