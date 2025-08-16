import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using MTKNeuralToolkit 
import MTKNeuralToolkit
import MTKNeuralToolkit.HodgkinHuxley as HH_module
using Plots
@named inp = TimeVaryingFunction(f=t -> (exp(sin(t))))
neurons = [build_HH(inp; name=:HH), build_HH(;name=:Liu)]
connections = Dict(
    (1, 2) => [(type=:Exc, weight=1.0)],
    (2, 1) => [(type=:Inh, weight=100.0)]
)
print("build_network")
network = build_network(connections, neurons)
#neuron = mtkcompile(build_HH(inp; name=:HH))
print("ODE_Problem")
prob = ODEProblem(network, Pair[], (0.0, 10.0) )
sol = solve(prob, Tsit5());

p = plot(sol, idxs=[network.Liu.Liu.V, network.HH.HH.V])
gui(p)
#=
@named inp = TimeVaryingFunction(f=t -> (exp(sin(t))))

# Build just the components
@named soma = BasicSoma(C=1)
println("Soma nameof: ", nameof(soma))
println("Soma type: ", typeof(soma))

@named Na = build_channel(HH_module.NaGates(;g=120, E=50), FixedReversal(;E=50); name = :Na)
println("Channel nameof: ", nameof(Na))

# Now let's see what extend creates
channel_connections = [
    connect(Na.p, soma.p),
    connect(soma.ground.g, soma.n, Na.n)
]

# Try extend with a different name
extended = extend(soma, ODESystem(channel_connections, t, name=:connections; systems=[Na, inp]))
println("Extended system name: ", nameof(extended))
println("Extended has HH?: ", hasproperty(extended, :HH))

# Check the actual error location
try
    structural_simplify(extended)
catch e
    println("Error type: ", typeof(e))
    println("Error message: ", e.msg)
    # Get a more detailed stack trace
    for (i, frame) in enumerate(stacktrace(catch_backtrace()))
        if occursin("ModelingToolkit", string(frame))
            println("Frame $i: ", frame)
        end
    end
end
=#