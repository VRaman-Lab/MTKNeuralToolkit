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
using SciMLSensitivity, FiniteDiff

@named inp1 = TimeVaryingFunction(f=t -> (exp(sin(t+0.2)*sin(t+0.2))))
@named inp2 = TimeVaryingFunction(f=t -> (exp(sin(t)*sin(t))))

weights = [rand(), 5*rand()]

neurons = [
    build_HH(inp1; name=:HH1),
    build_HH(inp2; name=:HH2),
    build_HH(; name=:HH3)
]

function solve_network(weights)

connections = Dict(
    (1,3) => [(type=:Exc, weight=weights[1])],
    (2,3) => [(type=:Inh, weight=weights[2])],
)

    @time network = build_network(connections, neurons)

    @time prob = ODEProblem(network, Pair[], (0.0, 50.0))
    @time sol = solve(prob, TRBDF2(), sensealg=QuadratureAdjoint());
    return sol.u[end][3]
end

grad = FiniteDiff.finite_difference_gradient(solve_network, weights)

print(grad)
