import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using MTKNeuralToolkit 
import MTKNeuralToolkit.Liu as Liu
#using script_utils.jl
using Plots

Na =  build_channel(Liu.NaGates(;g=100, E = 50.0), FixedReversal(;E=50.0); name = :Na)
KCa =  build_channel(Liu.KCaGates(;g=10.0, E = -80.0), FixedReversal(;E=-80.0); name = :KCa)
CaS =  build_channel(Liu.CaSGates(;g=1.3), FixedReversal(;E=0.0); name = :CaS)
CaT =  build_channel(Liu.CaTGates(;g=3.0), FixedReversal(;E=0.0); name = :CaT)
K =  build_channel(Liu.KGates(;g=5.0, E = -80.0), FixedReversal(;E=-80.0); name = :K)
DRK =  build_channel(Liu.DRKGates(;g=20.0, E = -80.0), FixedReversal(;E=-80.0); name = :KDR)
H =  build_channel(Liu.HGates(;g=0.5, E = -20.0), FixedReversal(;E=-20.0); name = :H)
Leak =  build_channel(Liu.LeakGates(;g=0.1, E = -50.0), FixedReversal(;E=-50.0); name = :Leak)

@named inp = TimeVaryingFunction(f=t -> exp(sin(t)*sin(t)))
fn = Liu.CalciumSensitiveNeuron(; C=1, name = :soma)

neur = build_neuron(fn, inp;  channels = [KCa, Na, CaS, CaT, K, DRK, H, Leak])
neur = structural_simplify(neur) 

prob = ODEProblem(neur, Pair[], (0.0, 400.0) )
sol = solve(prob, TRBDF2(), maxiters=1e9);


#p = plot(sol,idxs=[neur.Na.conductance.m_gate,neur.Na.conductance.h_gate], layout=(4,1), subplot=1)
#plot!(p, sol, idxs=[neur.Kca.conductance.n_gate], subplot=2)
#plot!(p, sol, idxs=[neur.soma.v], subplot=3)
p = plot(layout=(11,1), size=(1200,2000))
plot!(p, sol, idxs=[neur.Na.conductance.m, neur.Na.conductance.h], subplot=2)
plot!(p, sol, idxs=[neur.KCa.conductance.m], subplot=3)
plot!(p, sol, idxs=[neur.CaS.conductance.m, neur.CaS.conductance.h], subplot=4)
plot!(p, sol, idxs=[neur.CaT.conductance.m, neur.CaT.conductance.h], subplot=5)
plot!(p, sol, idxs=[neur.K.conductance.m, neur.K.conductance.h], subplot=6)
plot!(p, sol, idxs=[neur.KDR.conductance.m], subplot=7)
plot!(p, sol, idxs=[neur.H.conductance.m], subplot=8)
plot!(p, sol, idxs=[neur.soma.v], subplot=9)
plot!(p, sol, idxs=[neur.soma.Ca], subplot=10)
plot!(p, sol, idxs=[neur.soma.ca.i], subplot=11)
gui(p)
#savefig(p, "neuron_channels.png")
#plot(sol, idxs=[neur.CaS.conductance.E])
#plot(sol, idxs=[neur.soma.Ca])
#plot(sol, idxs=[neur.soma.Ca, neur.soma.v], layout=(2,1))