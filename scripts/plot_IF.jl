
import Pkg

Pkg.activate(@__DIR__)
import Pkg; Pkg.add("OrdinaryDiffEqNonlinearSolve")
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using OrdinaryDiffEqNonlinearSolve
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using MTKNeuralToolkit 
import MTKNeuralToolkit.IntegrateAndFire as IaF
import MTKNeuralToolkit.HodgkinHuxley as HH
import MTKNeuralToolkit
#using script_utils.jl
using Plots

IF = build_channel(IaF.IF_channel(; E=-65, name = :conductance), FixedReversal(; E=0); name =:IF)

@named inp = TimeVaryingFunction(f = t -> ifelse((t > 10) & (t < 20), 100.0, 0.0))
fn = BasicSoma(; C=10, name = :soma)

neur = build_neuron(fn, inp; channels = [IF])
neur = structural_simplify(neur)

prob = ODEProblem(neur, Pair[], (0.0, 40.0))

sol = solve(prob, Tsit5())

p = plot(sol, idxs=[neur.soma.oneport.v],label="LIF neuron", ylabel="Voltage(V)", xlabel="Time(ms)",layout=(2,1), subplot =1)
t_vec = 0:0.1:40  # Time vector
input_current = [ifelse((t > 10) & (t < 20), 100.0, 0.0) for t in t_vec]
plot!(t_vec, input_current, label="Input Current", xlabel="Time(ms)", ylabel="Current(A)", subplot=2)