import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkitNeuralNets
using Lux
using ModelingToolkit
using ModelingToolkitStandardLibrary
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks: Constant, RealInput, TimeVaryingFunction, Sum, RealInputArray, RealOutputArray
using ModelingToolkit: t_nounits as t, D_nounits as D
using Random

using OrdinaryDiffEq
using MTKNeuralToolkit 
import MTKNeuralToolkit.HodgkinHuxley as HH
import MTKNeuralToolkit.RMM as RMM
import MTKNeuralToolkit
using Plots
using LinearAlgebra

A_Mat = [0.6065, 0.8465, 0.9048, 0.9310, 0.9512, 0.9834, 0.9900, 0.9929]
B_Vec = [0.3935, 0.1535, 0.0952, 0.0690, 0.0488, 0.0166, 0.0100, 0.0071]
Na =    build_channel(HH.NaGates(;g=40, E = 55), FixedReversal(;E=55); name = :Na)      
K =     build_channel(HH.KGates( ;g=35, E = -77), FixedReversal(;E=-77); name = :K)
Leak =  build_channel(HH.LGates( ;g=0.3, E = -65), FixedReversal(;E=-65); name = :Leak)
rmm_channel = build_channel(RMM.RMMVecf(g=0.1, E=-65,  A_Mat=A_Mat, B_Vec=B_Vec;name=:conductance), FixedReversal(E=-77); name =:RMM)

@named inp = TimeVaryingFunction(f=t -> sin(t))
fn = BasicSoma(; C=1, name = :soma)
println("________________")
neur = build_neuron(fn, inp; channels = [rmm_channel, Na, K, Leak])
neur_c = structural_simplify(neur) 

prob = ODEProblem(neur_c, Pair[], (0.0, 200.0) )
@time sol = solve(prob, Rodas5());
@time sol = solve(prob, Rodas5());

p = plot(sol,idxs=[neur.Na.conductance.m_gate,neur.Na.conductance.h_gate], layout=(4,1), subplot=1)
plot!(p, sol, idxs=[neur.K.conductance.n_gate], subplot=2)
plot!(p, sol, idxs=[neur.RMM.conductance.lti_v_plotter], subplot=3)
plot!(p, sol, idxs=[neur.soma.v], subplot=4)
