using ModelingToolkit
using ModelingToolkit: mtkcompile, @named, System
using ModelingToolkit: t_nounits as t
using OrdinaryDiffEq
using Plots

# ==============================================================================
# 1. Setup 3 neurons using the clean build_compartment
# ==============================================================================
make_soma() = LIFCapacitor(C = 1.0, V_th = -55.0, V_reset = -67.0, V_init = -65.0, name = :soma)

neurons = [
    build_compartment(make_soma(), []; has_synapses = true, name = Symbol(:n, i)) 
    for i in 1:3
]

# ==============================================================================
# 2. Build the Dense/Symbolic Ring Network Matrix Setup
# ==============================================================================
W = [0.0  3.0  0.0;  
     0.0  0.0  3.0;  
     3.0  0.0  0.0]  

E_rev_matrix = [0.0  0.0  0.0;
                0.0  0.0  0.0;
                0.0  0.0  0.0]

pulse_stimulus = 5.0 * (t > 0.0) * (t < 3.0)

# Map this temporary kick onto Neuron 1
stimulus_exprs = [pulse_stimulus, nothing, nothing]
println("Compiling 3-Neuron dense network with native symbolic driving...")
@named dense_ring = build_dense_network(neurons, W, E_rev_matrix; stimulus_exprs=stimulus_exprs)
dense_compiled = mtkcompile(dense_ring)
println("Success! Dense system is perfectly balanced.")


# ==============================================================================
# 3. Build the Heterogeneous Ring Network using AlphaSynapse
# ==============================================================================
# Create a matrix of component factory closures mapping to AlphaSynapse
SynM = Matrix{Union{Function, Nothing}}(nothing, 3, 3)
SynM[1, 2] = (; kwargs...) -> AlphaSynapse(; g_max=0.5, τ=5.0, E_rev=0.0, v_th=-56.0, w=0.1, kwargs...)
SynM[2, 3] = (; kwargs...) -> AlphaSynapse(; g_max=0.5, τ=5.0, E_rev=0.0, v_th=-56.0, w=0.1, kwargs...)
SynM[3, 1] = (; kwargs...) -> AlphaSynapse(; g_max=0.5, τ=5.0, E_rev=0.0, v_th=-56.0, w=0.1, kwargs...)

println("Compiling 3-Neuron heterogeneous network via AlphaSynapse components...")
@named heterogeneous_ring = build_heterogeneous_network(neurons, SynM; stimulus_exprs=stimulus_exprs)
compiled_net = mtkcompile(heterogeneous_ring)
println("Success! Heterogeneous system is perfectly balanced.")


# ==============================================================================
# 4. Simulation and Diagnostics
# ==============================================================================
prob = ODEProblem(compiled_net, [], (0.0, 30.0))
sol = solve(prob, Rosenbrock23())

p1 = plot(sol, idxs=[
    compiled_net.n1.p.v,          # True pre-synaptic voltage
    compiled_net.syn_1_to_2.V_pre # Voltage received by the synapse input port
], title="Diagnostic 1: Pre-Synaptic Coupling", xlabel="Time", ylabel="Voltage (mV)")

p2 = plot(sol, idxs=[
    compiled_net.syn_1_to_2.V_pre, 
    compiled_net.syn_1_to_2.s
], title="Diagnostic 2: Gating Variable Check", xlabel="Time", ylabel="Value")

# Main Voltage Trace
p3 = plot(sol, idxs=[compiled_net.n1.V, compiled_net.n2.V, compiled_net.n3.V], 
          title="Voltage Trace", xlabel="Time", ylabel="Voltage (mV)")

display(p1)
display(p2)
display(p3)
