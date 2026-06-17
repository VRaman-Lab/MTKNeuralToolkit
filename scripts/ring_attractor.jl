using MTKNeuralToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named, System
using ModelingToolkit: t_nounits as t
using OrdinaryDiffEq
using Plots

# ==========================================
# 1. Network Size & Topology Configuration
# ==========================================
const NUM_NEURONS = 8

function angular_distance(i, j, total)
    θ_i = (i - 1) * (2π / total)
    θ_j = (j - 1) * (2π / total)
    Δθ = abs(θ_i - θ_j)
    return Δθ > π ? 2π - Δθ : Δθ
end

# ==========================================
# 2. Eager Instantiation of Components
# ==========================================
println("Pre-allocating network components explicitly...")

# Constrain the comprehension type to System to bypass type-inference checks
neurons = System[build_compartment(LIFCapacitor(C = 1.0; name=:soma), []; name = Symbol(:neuron_, i)) for i in 1:NUM_NEURONS]

# Pre-allocate layout metadata
# Format: (pre_idx, post_idx, gate_component, battery_component, name)
connections_metadata = Tuple{Int, Int, System, System, Symbol}[]

for i in 1:NUM_NEURONS, j in 1:NUM_NEURONS
    if i != j
        dist = angular_distance(i, j, NUM_NEURONS)
        syn_name = Symbol(:syn_, i, :_to_, j)
        
        if dist < (π / 4)
            g_max = 3.0 * cos(2 * dist) 
            gate = EventSynapseGate(g_max = g_max, τ = 5.0, v_th = -55.0, w = 1.0; name = Symbol(:gate_, i, :_to_, j))
            batt = FixedReversal(E = 0.0; name = Symbol(:batt_, i, :_to_, j))
            push!(connections_metadata, (i, j, gate, batt, syn_name))
        else
            g_max = 0.5 * sin(dist)
            gate = EventSynapseGate(g_max = g_max, τ = 10.0, v_th = -55.0, w = 0.5; name = Symbol(:gate_, i, :_to_, j))
            batt = FixedReversal(E = -70.0; name = Symbol(:batt_, i, :_to_, j))
            push!(connections_metadata, (i, j, gate, batt, syn_name))
        end
    end
end

# Assemble the network
@named ring_system = build_network(neurons, connections_metadata; drivers=drivers)
ring_compiled = mtkcompile(ring_system)

# ==========================================
# 5. Simulation & Seamless Plot Unpacking
# ==========================================
prob = ODEProblem(ring_compiled, [], (0.0, 50.0); warn_initialize_determined = false)
sol = solve(prob, Tsit5())

time_steps = 0.0:0.5:50.0
voltage_matrix = zeros(NUM_NEURONS, length(time_steps))

for (t_idx, t_val) in enumerate(time_steps)
    for n_idx in 1:NUM_NEURONS
        voltage_matrix[n_idx, t_idx] = sol(t_val, idxs=neurons[n_idx].V)
    end
end

heatmap(
    time_steps, 
    1:NUM_NEURONS, 
    voltage_matrix, 
    xlabel="Time (ms)", 
    ylabel="Neuron Index around Ring", 
    title="Eager Ring Attractor Dynamics",
    c=:viridis
)
