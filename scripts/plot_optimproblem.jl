import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using Plots
using ModelingToolkit
using OrdinaryDiffEq
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using ModelingToolkit: t_nounits as t, D_nounits as D
import MTKNeuralToolkit.Synapse as Synapse
import MTKNeuralToolkit.Types: SYNAPSE_TYPES
using MTKNeuralToolkit
using SciMLSensitivity
import Statistics: mean
using Optim

@named inp1 = TimeVaryingFunction(f=t -> (exp(sin(t+0.1))))
@named inp2 = TimeVaryingFunction(f=t -> (exp(sin(t))))
@named inp3 = TimeVaryingFunction(f=t -> (exp(sin(t+0.2))))

weights = [1.0, 1.0]

neurons = [
    build_HH(inp1; name=:HH1),
    build_HH(inp2; name=:HH2),
    build_HH(inp3; name=:HH3),
    build_HH(; name=:HH4)
]
loss_history = Float64[]

function loss(weights)
    connections = Dict(
        (1,4) => [(type=:Inh, weight=3.0)],
        (2,4) => [(type=:Exc, weight=abs(weights[1]))],
        (3,4) => [(type=:Exc, weight=abs(weights[2]))],
    )
    
    network = build_network(connections, neurons)
    prob = ODEProblem(network, Pair[], (0.0, 10.0)) 
    sol = solve(prob, TRBDF2(); saveat=0.5)  
    
    voltages = sol[neurons[4].HH4.V]
    l = mean(voltages.^2)
    push!(loss_history, l)
    return l
end

result = optimize(loss, weights, BFGS(), 
                 Optim.Options(iterations=10, show_trace=true))

println("Optimized weights: ", result.minimizer)
println("Final loss: ", result.minimum)
p = plot(loss_history, label="Loss", xlabel="Function Evaluation", 
     ylabel="Loss", title="Optimization of Synaptic Weights")
gui(p)