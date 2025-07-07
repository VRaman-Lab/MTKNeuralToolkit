import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using ModelingToolkit: t_nounits as t, D_nounits as D
import MTKNeuralToolkit.Synapse as Synapse
import MTKNeuralToolkit.HodgkinHuxley as HH
import MTKNeuralToolkit.Liu as Liu
import MTKNeuralToolkit.Prinz as Prinz
import MTKNeuralToolkit.Config as cfg
import MTKNeuralToolkit.Types: SYNAPSE_TYPES, NEURON_TYPES, CustomSynapseParams
using MTKNeuralToolkit
include("script_utils.jl")
#include("script_types.jl")
using Plots

@named inp2 = TimeVaryingFunction(f=t -> exp(sin(t)))

#=neurons = Dict(
    "AB" => build_Prinz(inp2;name=:AB, config=cfg.PrinzConfig()),
    "PY" => build_Prinz(;name=:PD, config=cfg.PrinzConfig(KCa_g=0.0, CaS_g=2.0, CaT_g=2.4,H_g=0.05,K_g=50.0,DRK_g=100.0,Leak_g=0.01)),
    "LP" => build_Prinz(;name=:LP, config=cfg.PrinzConfig(KCa_g=0.0, CaS_g=4.0, CaT_g=0.0, H_g=0.05, K_g=20.0, DRK_g=25.0, Leak_g=0.03))
)=#
neurons = Dict(
    "AB" => build_Prinz(inp2; name=:AB, config=cfg.PrinzConfig(V0=-60.0)),
    "PY" => build_Prinz(;name=:PY, config=cfg.PrinzConfig(V0=-55.0, CaS_g=2.0, CaT_g=2.4,H_g=0.05, KCa_g=0.0, DRK_g=25.0, Leak_g=0.03)),
    "LP" => build_Prinz(;name=:LP, config=cfg.PrinzConfig(CaS_g=4.0, CaT_g=0.0, H_g=0.05, K_g=20.0, KCa_g=0.0, DRK_g = 25.0, Leak_g=0.03))
)
connections = Dict(
    ("AB", "LP") => (type=:Chol, weight=30.0),
    ("AB", "PY") => (type=:Chol, weight=3.0),
    ("AB", "LP") => (type=:Glut, weight=30.0),
    ("AB", "PY") => (type=:Glut, weight=10.0),

    ("LP", "AB") => (type=:Glut, weight=30.0),
    ("LP", "PY") => (type=:Glut, weight=1.0),
    
    ("PY", "LP") => (type=:Glut, weight=30.0),
)
@time network = build_network(connections, neurons)

@time prob = ODEProblem(network, Pair[], (0.0, 500.0) )
@time inspect_network(network)
@time sol = solve(prob, TRBDF2());

p = plot(sol, idxs=parse_sol_for_membrane_voltages(sol), size=(1000, 800))
gui(p)