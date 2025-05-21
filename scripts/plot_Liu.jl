import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using MTKNeuralToolkit  # replace with your actual package name
import MTKNeuralToolkit.Liu as Liu
using Plots




# leak = build_channel(Liu.LGates,FixedReversal)(;g=0.3,E=-65,name=:leak)
# pot =  build_channel(Liu.KGates, FixedReversal)(;g=35,E=-77,name=:pot)
Na =  build_channel(Liu.NaGates(;g=40, E = 55), FixedReversal(;E=55); name = :Na)
KCa =  build_channel(Liu.KCaGates(;g=40, E = 55), FixedReversal(;E=55); name = :KCa)
CaS =  build_channel(Liu.CaSGates(;g=40), Liu.CalciumReversal(); name = :CaS)

@named inp = TimeVaryingFunction(f=t -> sin(t))
fn = Liu.CalciumSensitiveNeuron(; C=1, name = :soma)

neur = build_neuron(fn, channels = [KCa, Na, CaS], input = inp)
neur = structural_simplify(neur) 

prob = ODEProblem(neur, [neur.CaS.conductance.g => 20], (0.0, 20.0) )
sol = solve(prob, Tsit5());


# p = plot(sol,idxs=[neur.sod.conductance.m_gate,neur.sod.conductance.h_gate], layout=(4,1), subplot=1)
# plot!(p, sol, idxs=[neur.pot.conductance.n_gate], subplot=2)
# plot!(p, sol, idxs=[neur.soma.v], subplot=3)
plot(sol)

