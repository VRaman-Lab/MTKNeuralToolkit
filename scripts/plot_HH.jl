import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using MTKNeuralToolkit  # replace with your actual package name
import MTKNeuralToolkit.HodgkinHuxley as HH
using Plots



Na =    build_channel(HH.NaGates(;g=40, E = 55), FixedReversal(;E=55); name = :Na)
K =     build_channel(HH.KGates( ;g=35, E = -77), FixedReversal(;E=-77); name = :K)
Leak =  build_channel(HH.LGates( ;g=0.3, E = -65), FixedReversal(;E=-65); name = :Leak)


# leak = build_channel(HH.LGates,FixedReversal)(;g=0.3,E=-65,name=:leak)
# pot =  build_channel(HH.KGates, FixedReversal)(;g=35,E=-77,name=:pot)
# sod =  build_channel(HH.NaGates, FixedReversal)(;g=40,E=55,name=:sod)

@named inp = TimeVaryingFunction(f=t -> sin(t))
fn = BasicSoma(; C=1, name = :soma)

# neur = build_neuron(fn, channels = [leak, pot, sod], input = inp)
neur = build_neuron(fn, channels = [Na, K, Leak], input = inp)
neur = structural_simplify(neur) 

prob = ODEProblem(neur, Pair[], (0.0, 200.0) )
sol = solve(prob, Tsit5());


p = plot(sol,idxs=[neur.Na.conductance.m_gate,neur.Na.conductance.h_gate], layout=(4,1), subplot=1)
plot!(p, sol, idxs=[neur.K.conductance.n_gate], subplot=2)
plot!(p, sol, idxs=[neur.soma.v], subplot=3)

