import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using MTKNeuralToolkit  # replace with your actual package name
import MTKNeuralToolkit.Liu as Liu
#using script_utils.jl
using Plots




# leak = build_channel(Liu.LGates,FixedReversal)(;g=0.3,E=-65,name=:leak)
# pot =  build_channel(Liu.KGates, FixedReversal)(;g=35,E=-77,name=:pot)
Na =  build_channel(Liu.NaGates(;g=100, E = 50.0), FixedReversal(;E=50.0); name = :Na)
KCa =  build_channel(Liu.KCaGates(;g=10.0, E = -80.0), FixedReversal(;E=-80.0); name = :KCa)
CaS =  build_channel(Liu.CaSGates(;g=1.3), Liu.CalciumReversal(); name = :CaS)
CaT =  build_channel(Liu.CaTGates(;g=3.0), Liu.CalciumReversal(); name = :CaT)
K =  build_channel(Liu.KGates(;g=5.0, E = -80.0), FixedReversal(;E=-80.0); name = :K)
DRK =  build_channel(Liu.DRKGates(;g=20.0, E = -80.0), FixedReversal(;E=-80.0); name = :KDR)
H =  build_channel(Liu.HGates(;g=0.5, E = -20.0), FixedReversal(;E=-20.0); name = :H)
Leak =  build_channel(Liu.LeakGates(;g=0.1, E = -50.0), FixedReversal(;E=-50.0); name = :Leak)

@named inp = TimeVaryingFunction(f=t -> sin(t))
fn = Liu.CalciumSensitiveNeuron(; C=1, name = :soma)

neur = build_neuron(fn, inp;  channels = [KCa, Na, CaS, CaT, K, DRK, H, Leak])
neur = structural_simplify(neur) 

prob = ODEProblem(neur, [neur.CaS.conductance.g => 1.3], (0.0, 20.0) )
sol = solve(prob, TRBDF2(), maxiters=1e9);


# p = plot(sol,idxs=[neur.sod.conductance.m_gate,neur.sod.conductance.h_gate], layout=(4,1), subplot=1)
# plot!(p, sol, idxs=[neur.pot.conductance.n_gate], subplot=2)
# plot!(p, sol, idxs=[neur.soma.v], subplot=3)
#plot(sol, idxs=[neur.CaS.conductance.E])
#plot(sol, idxs=[neur.soma.Ca])
plot(sol, idxs=[neur.soma.Ca, neur.soma.v], layout=(2,1))