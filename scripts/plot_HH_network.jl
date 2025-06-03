import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using MTKNeuralToolkit 
import MTKNeuralToolkit.Synapse as Synapse
import MTKNeuralToolkit.HodgkinHuxley as HH
#using script_utils.jl
using Plots

function build_HH(input=nothing; name)

    Na = build_channel(HH.NaGates(;g=40, E = 55), FixedReversal(;E=55); name = :Na)      
    K = build_channel(HH.KGates( ;g=35, E = -77), FixedReversal(;E=-77); name = :K)
    Leak = build_channel(HH.LGates( ;g=0.3, E = -65), FixedReversal(;E=-65); name = :Leak)

    fn=BasicSoma(; C=1, name = :soma)

    if input === nothing
        neur = build_neuron(fn; channels = [Na, K, Leak])
    else
        neur = build_neuron(fn, input; channels = [Na, K, Leak])
    end
    return(neur)
end
@named inp = TimeVaryingFunction(f=t -> sin(t))

@named pre_neur = build_HH(inp)
@named post_neur = build_HH()
#println(typeof(pre_neur.soma.p))
@named syn_channel = Synapse.E_syn_gates(;g=0.1, E = 0.0, name =:exc_syn)
synapse = add_synapse(syn_channel, pre_neur, post_neur)
synapse = structural_simplify(synapse)

prob = ODEProblem(synapse, Pair[], (0.0, 200.0) )
sol = solve(prob, Tsit5());

p = plot(sol)