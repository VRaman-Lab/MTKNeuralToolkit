import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkitNeuralNets
using Lux
using ModelingToolkit
using ModelingToolkitStandardLibrary
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks: Constant, RealInput, TimeVaryingFunction, Sum
using ModelingToolkit: t_nounits as t, D_nounits as D
using Random

using ModelingToolkit
using OrdinaryDiffEq
using MTKNeuralToolkit 
import MTKNeuralToolkit.HodgkinHuxley as HH
import MTKNeuralToolkit.RMM as RMM
import MTKNeuralToolkit
#using script_utils.jl
using Plots

function ANN(; g=1.0, E=-65.0, name=:ann_gate)
    @named oneport = OnePort()
    @variables t v(t) i(t) nn_out(t) nn_input(t)
    @parameters g=g E=E
    
    # Create the neural network component manually
    chain = multi_layer_feed_forward(1, 1)
    NN, params = SymbolicNeuralNetwork(chain=chain, n_input=1, n_output=1, 
                                     rng=Xoshiro(42))

    eqs = [
        v ~ oneport.v
        i ~ oneport.i
        nn_out ~ NN([v], params)[1]
        i ~ g * nn_out * (v - E)
    ]
    
    #return ODESystem(eqs, t, [v, i, nn_out, oneport.p, oneport.n], [g, E, p]; systems=[oneport], name=name)
    #sys = ODESystem(eqs, t, [v, i, nn_out], [g, E, params];name=name)
    #return extend(sys, oneport)
    
    return ODESystem(eqs, ModelingToolkit.t_nounits; systems = [oneport], name=name)
    #return extend(sys, oneport)

end


#const ANN_Chain = Lux.Chain(Lux.Dense(1 => 16, Lux.mish, use_bias = false),Lux.Dense(16 => 8, Lux.mish, use_bias = false),Lux.Dense(8 => 1, use_bias = false))
chain = multi_layer_feed_forward(1, 1)
#Specify inputs
#Na =    build_channel(HH.NaGates(;g=40, E = 55), FixedReversal(;E=55); name = :Na)      
#K =     build_channel(HH.KGates( ;g=35, E = -77), FixedReversal(;E=-77); name = :K)
#Leak =  build_channel(HH.LGates( ;g=0.3, E = -65), FixedReversal(;E=-65); name = :Leak)
#ANN =  build_channel(RMM.( ;g=0.3, E = -65), FixedReversal(;E=-65); name = :Poop)
ann_gates = ANN(g=0.3, E=-65; name =:conductance)
ann_gates = structural_simplify(ann_gates)
@named ann_channel = build_channel_ann(ann_gates, FixedReversal(E=-65); name =:DonkeyBallz)
@show propertynames(ann_channel)
@show hasproperty(ann_channel, :conductance)

@named inp = TimeVaryingFunction(f=t -> sin(t))
fn = BasicSoma(; C=1, name = :soma)

# neur = build_neuron(fn, channels = [leak, pot, sod], input = inp)
neur = build_neuron(fn, inp; channels = [ann_channel])
neur = structural_simplify(neur) 

prob = ODEProblem(neur, Pair[], (0.0, 200.0) )
sol = solve(prob, Tsit5());


p = plot(sol,idxs=[neur.Na.conductance.m_gate,neur.Na.conductance.h_gate], layout=(4,1), subplot=1)
plot!(p, sol, idxs=[neur.K.conductance.n_gate], subplot=2)
plot!(p, sol, idxs=[neur.soma.v], subplot=3)

