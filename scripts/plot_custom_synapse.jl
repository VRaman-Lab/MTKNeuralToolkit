import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using ModelingToolkit: t_nounits as t, D_nounits as D
using MTKNeuralToolkit
using Plots

@mtkmodel GapJunction begin
    @extend v_pre, v_post, i_post, i_pre = twoport = BiDirectionalTwoPort()
    @parameters begin
        g_gap, [description = "Gap junction conductance"] 
        V_threshold = -20.0, [description = "Voltage threshold for gating"]
        k_gate = 10.0, [description = "Gating sensitivity"]
    end
    @equations begin
        i_post ~ g_gap / (1.0 + exp(k_gate * (abs(v_pre - v_post) - V_threshold))) * (v_pre - v_post)
        i_pre ~ -i_post
    end
end

# Constructor function
gap_junction(;g, name, threshold=-20.0, sensitivity=10.0) = 
    GapJunction(g_gap=g, V_threshold=threshold, k_gate=sensitivity, name=name)

@named inp2 = TimeVaryingFunction(f=t -> (sin(t)))
neurons = [build_HH(;name=:x1), build_HH(;name=:x2)]

connections = Dict(
    (1,2) => [() -> GapJunction(;g_gap=30.0, name=:junk)]
)

@time network = build_network(connections, neurons)

@time prob = ODEProblem(network, Pair[], (0.0, 500.0))
#inspect_network(network)
@time sol = solve(prob, TRBDF2());

p = plot(sol, idxs=parse_sol_for_membrane_voltages(sol), size=(1000, 800))
gui(p)