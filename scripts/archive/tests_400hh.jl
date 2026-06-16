using ModelingToolkit
using OrdinaryDiffEq
using ModelingToolkitStandardLibrary.Blocks: TimeVaryingFunction 
using ModelingToolkit: t_nounits as t, D_nounits as D
import MTKNeuralToolkit.HodgkinHuxley as HH
import MTKNeuralToolkit.Config as cfg
using MTKNeuralToolkit

# Create 400 HH neurons with simple inputs
n_neurons = 400
neurons = Dict{String, Any}()

println("Building $n_neurons neurons...")
build_start = time()
for i in 1:n_neurons
    # Simple constant input for each neuron
    @eval @named $(Symbol("inp_$i")) = TimeVaryingFunction(f=t -> 0.1)
    neurons["N$i"] = build_HH(@eval $(Symbol("inp_$i")); 
                              name=Symbol("N$i"), 
                              config=cfg.HHConfig(V0=-65.0))
end

# Create sparse random inhibitory connections (~10% connectivity)
connections = Dict{Tuple{String,String}, Vector{@NamedTuple{type::Symbol, weight::Float64}}}()
syn_weight = 2.0

for pre in 1:n_neurons
    for post in 1:n_neurons
        if pre != post && rand() < 0.1
            connections[("N$pre", "N$post")] = [(type=:Inh, weight=syn_weight)]
        end
    end
end

println("Total connections: ", length(connections))

println("Building network")
@time network = build_network(connections, neurons)

println("Building ODEProblem")
@time prob = ODEProblem(network, missing, (0.0, 500.0))
println("Equations: ", length(equations(network)))

build_end = time()

println("Solving")
solve_start = time()
@time sol = solve(prob, TRBDF2());
solve_end = time()

println("\n=====================================")
println("Building time     : $(round(build_end - build_start, digits=2)) s")
println("Simulation time   : $(round(solve_end - solve_start, digits=2)) s")
println("=====================================")