import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using MTKNeuralToolkit 
import MTKNeuralToolkit.Liu as Liu
using Plots

Na =  build_channel(Liu.NaGates(;g=100, E = 50.0), FixedReversal(;E=50.0); name = :Na)
KCa =  build_channel(Liu.KCaGates(;g=10.0, E = -80.0), FixedReversal(;E=-80.0); name = :KCa)
CaS =  build_channel(Liu.CaSChannel(;g=1.3); name = :CaS)
CaT =  build_channel(Liu.CaTChannel(;g=3.0); name = :CaT)
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

p = plot(layout=(2,1), size=(1200,2000))
plot!(p, sol, idxs=[neur.soma.V], subplot=1)
plot!(p, sol, idxs=[neur.soma.Ca, neur.soma.ca.i], subplot=2)
gui(p)