using MTKNeuralToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named, System
using OrdinaryDiffEq
using Plots

const NUM_NEURONS = 8

function angular_distance(i, j, total)
    θ_i = (i - 1) * (2π / total)
    θ_j = (j - 1) * (2π / total)
    Δθ = abs(θ_i - θ_j)
    return Δθ > π ? 2π - Δθ : Δθ
end

neurons = System[]
for i in 1:NUM_NEURONS
    nrn = build_compartment(LIFCapacitor(C=1.0; name=:soma), []; name = Symbol(:neuron_, i))
    push!(neurons, nrn)
end

connections = Tuple[]
for i in 1:NUM_NEURONS, j in 1:NUM_NEURONS
    if i != j
        dist = angular_distance(i, j, NUM_NEURONS)
        syn_name = Symbol(:syn_, i, :_to_, j)

        if dist < (π / 4)
            g_max = 3.0 * cos(2 * dist)
            gen = (; name) -> ChemicalSynapse(name=name, g_max=g_max, τ=5.0, v_th=-55.0, w=1.0, E_rev=0.0)
            push!(connections, (i, j, gen, syn_name))
        else
            g_max = 0.5 * sin(dist)
            gen = (; name) -> ChemicalSynapse(name=name, g_max=g_max, τ=10.0, v_th=-55.0, w=0.5, E_rev=-70.0)
            push!(connections, (i, j, gen, syn_name))
        end
    end
end

@named stim = Blocks.Sine(frequency = 0.1, amplitude = 20.0)
drivers = [(1, stim)]

# 1. Time the Network Builder
println("Building network...")
t_build = @elapsed @named ring_system = build_electrical_network(neurons, connections; drivers=drivers)
println("   > Build Time: $(round(t_build, digits=2)) seconds")

# 2. Time the Compiler
println("Compiling network...")
t_compile = @elapsed ring_compiled = mtkcompile(ring_system)
println("   > Compile Time: $(round(t_compile, digits=2)) seconds")

# 3. Time the Solver
println("Solving network...")
prob = ODEProblem(ring_compiled, [], (0.0, 50.0))
t_solve = @elapsed sol = solve(prob, Tsit5(); reltol=1e-3, abstol=1e-3)
println("   > Solve Time: $(round(t_solve, digits=2)) seconds")
