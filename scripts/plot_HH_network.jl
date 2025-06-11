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
using MTKNeuralToolkit
include("script_utils.jl")
using Plots

#Create your inputs here. Different options include inbuilt Julia functions ->Sin, cos, exp, log, etc. For log you need to specify base.
#Not every neuron needs an input; create prebuilt input neurons by specifying their input functions, other prebuilt neurons through an int param.
#Neurons are added to a dict depending on when they were created, which depends on their location within the build_network args.
#If 3 input HHs and 1 input Liu, the first non-input HH will be n4, the first non-input LIF will be n$(3+[number_of_HHs+number_of_Lius])

@named inp = TimeVaryingFunction(f=t -> log(2*t,10))
@named inp2 = TimeVaryingFunction(f=t -> log(t,10))
@named inp3 = TimeVaryingFunction(f=t -> log(2*t,10))
@named inp4 = TimeVaryingFunction(f=t -> log(t,10))
@named inp5 = TimeVaryingFunction(f=t -> log(2*t,10))
@named inp6 = TimeVaryingFunction(f=t -> log(t,10))

#--Workflow 1
#network = build_network([inp, inp2, inp3, inp4, inp5, inp6], [], 0, 0, [["n0","n1",false,0.9],["n2","n3",false,0.5],["n4","n5",false,0.1]])

#--Workflow 2

@named n1 = build_Liu(inp; name=:n1)
@named n2 = build_HH(;name=:n2)
s1 = put_synapse(n1,n2,true,0.1; name=:s1)
network = compose(ODESystem([], t; name=:network), [s1])
network = structural_simplify(network)
#If using Liu neurons change solver from Tsit5 -> TRBDF2. 
#Might need to manually give more maxiters as well through solve(x,y, maxiters=[more lol])


prob = ODEProblem(network, Pair[], (0.0, 200.0) )
sol = solve(prob, TRBDF2());

plot(sol, idxs=parse_sol_for_membrane_voltages(sol), size=(1000, 800))