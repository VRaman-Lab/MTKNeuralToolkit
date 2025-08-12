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
using Plots

@named inp = TimeVaryingFunction(f=t -> 1)


syn_cf = 0.254
prinz_cf = 159.2
neurons = Dict(
    "AB" => build_Prinz(inp; name=:AB, config=cfg.PrinzConfig(
        V0=-60.0, Na_g=100.0*prinz_cf, CaS_g=6.0*prinz_cf, CaT_g=2.5*prinz_cf, 
        H_g=0.01*prinz_cf, K_g=50.0*prinz_cf, KCa_g=5.0*prinz_cf, DRK_g=100.0*prinz_cf, Leak_g=0.0*prinz_cf)),
    "PY" => build_Prinz(;name=:PY, config=cfg.PrinzConfig(
        V0=-55.0, Na_g=100.0*prinz_cf, CaS_g=2.0*prinz_cf, CaT_g=2.4*prinz_cf, 
        H_g=0.05*prinz_cf, K_g=50.0*prinz_cf, KCa_g=0.0*prinz_cf, DRK_g=125.0*prinz_cf, Leak_g=0.01*prinz_cf)),
    "LP" => build_Prinz(;name=:LP, config=cfg.PrinzConfig(
        V0=-65.0, Na_g=100.0*prinz_cf, CaS_g=4.0*prinz_cf, CaT_g=0.0*prinz_cf, 
        H_g=0.05*prinz_cf, K_g=20.0*prinz_cf, KCa_g=0.0*prinz_cf, DRK_g=25.0*prinz_cf, Leak_g=0.03*prinz_cf))
)

connections = Dict(
    ("AB", "LP") => [(type=:Chol, weight=30.0*syn_cf), (type=:Glut, weight=30.0*syn_cf)],
    ("AB", "PY") => [(type=:Chol, weight=3.0*syn_cf), (type=:Glut, weight=10.0*syn_cf)],
    ("LP", "AB") => [(type=:Glut, weight=30.0*syn_cf)],
    ("LP", "PY") => [(type=:Glut, weight=1.0*syn_cf)],
    ("PY", "LP") => [(type=:Glut, weight=30.0*syn_cf)],
)

println("Building network")
@time network = build_network(connections, neurons)
println("Building ODEProblem")
@time prob = ODEProblem(network, Pair[], (0.0, 100.0) )
println("Equations: ", length(equations(network)))

#@time inspect_network(network)
println("Solving")
@time sol = solve(prob, TRBDF2());

p = plot(sol, idxs=parse_sol_for_membrane_voltages(sol), size=(1000, 800))
gui(p)