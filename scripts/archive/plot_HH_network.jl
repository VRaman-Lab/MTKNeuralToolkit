import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using ModelingToolkit: t_nounits as t, D_nounits as D
import MTKNeuralToolkit.Synapse as Synapse
import MTKNeuralToolkit.Types: SYNAPSE_TYPES
using MTKNeuralToolkit

@named inp1 = TimeVaryingFunction(f=t -> (exp(sin(t))))
@named inp2 = TimeVaryingFunction(f=t -> (exp(sin(t))))
@named inp3 = TimeVaryingFunction(f=t -> (exp(sin(t))))
neurons = [
    build_HH(inp1; name=:HH1),
    build_HH(inp2; name=:HH2),
    build_HH(inp3; name=:HH3),
    build_HH(;name=:IHH1),
    build_HH(;name=:IHH2),
    build_HH(;name=:IHH3),
]
connections = Dict(
    (1,4) => [(type=:Exc, weight=3*(rand()))],
    (2,5) => [(type=:Exc, weight=3*(rand()))],
    (3,6) => [(type=:Exc, weight=3*(rand()))],

    (4,2) => [(type=:Inh, weight=20.0)],
    (4,3) => [(type=:Inh, weight=20.0)],

    (5,1) => [(type=:Inh, weight=20.0)],
    (5,3) => [(type=:Inh, weight=20.0)],

    (6,1) => [(type=:Inh, weight=20.0)],
    (6,2) => [(type=:Inh, weight=20.0)],
)
@time network = build_network(connections, neurons)

@time prob = ODEProblem(network, Pair[], (0.0, 500.0))
@time sol = solve(prob, TRBDF2());