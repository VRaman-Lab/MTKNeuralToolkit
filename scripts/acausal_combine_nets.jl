using MTKNeuralToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named, System, SymbolicT
using OrdinaryDiffEq
using Plots
using ModelingToolkit

# =============================================================================
# Define Channel Dynamics
# =============================================================================

# Standard Hodgkin-Huxley alpha/beta rates as lambda functions of voltage `v`
hh_na_m = v -> (
    0.182 * (v + 35.0) / (1.0 - exp(-(v + 35.0) / 9.0)),
    -0.124 * (v + 35.0) / (1.0 - exp((v + 35.0) / 9.0))
)
hh_na_h = v -> (
    0.25 * exp(-(v + 90.0) / 12.0),
    0.25 * (exp((v + 62.0) / 6.0)) / exp(-(v + 90.0) / 12.0)
)
hh_k_n = v -> (
    0.02 * (v - 25.0) / (1.0 - exp(-(v - 25.0) / 9.0)),
    -0.002 * (v - 25.0) / (1.0 - exp((v - 25.0) / 9.0))
)

# =============================================================================
# 2. Mixed Network Test Script
# =============================================================================

# 1. Build a population of 3 Hodgkin-Huxley style neurons
neurons = System[]
for i in 1:3
    @named soma = Capacitor(C = 1.0)

    channels = System[]
    
    # Sodium channel (m^3 * h^1)
    push!(channels, GenericChannel(
        name=Symbol(:na_, i), 
        g=120.0, 
        E_rev=50.0, 
        gates=[
            GateSpec(:m, 3, 0.0, hh_na_m),
            GateSpec(:h, 1, 1.0, hh_na_h)
        ]
    ))
    
    # Potassium channel (n^4)
    push!(channels, GenericChannel(
        name=Symbol(:k_, i), 
        g=36.0, 
        E_rev=-77.0, 
        gates=[
            GateSpec(:n, 4, 0.0, hh_k_n)
        ]
    ))
    
    # Leak channel (no gates)
    push!(channels, GenericChannel(
        name=Symbol(:l_, i), 
        g=0.3, 
        E_rev=-54.4, 
        gates=GateSpec[]
    ))

    nrn = build_compartment(soma, channels; name = Symbol(:nrn_, i))
    push!(neurons, nrn)
end

# Connection A: Event-driven Chemical Synapse (Neuron 1 -> Neuron 2)
event_synapse_conn = (
    neurons[1],
    neurons[2],
    (; name) -> ChemicalSynapse(name=name, g_max=0.5, τ=5.0, v_th=-20.0, w=0.1, E_rev=0.0)
)

# Connection B: Acausal Gap Junction (Neuron 2 <-> Neuron 3)
gap_junction_conn = (
    neurons[2],
    neurons[3],
    (; name) -> GapJunction(name=name, R=0.5)
)

# Combine them into a single flat edge list
all_connections = [event_synapse_conn, gap_junction_conn]

# 3. Setup a driving stimulus on the first neuron
@named stim = Blocks.Sine(frequency = 0.05, amplitude = 15.0)
drivers = [(1, stim)] # Target the first neuron by index

# 4. Build the complete mixed network using the explicit acausal builder
println("Building mixed acausal network...")
@named net = build_electrical_network(neurons, all_connections; drivers=drivers)

# Note: conservative=true and simplify=false drastically speed up compilation for circuits!
net_compiled = mtkcompile(net; conservative=true, simplify=false)

# 5. Simulate the network
println("Compiling and solving...")
prob = ODEProblem(net_compiled, Pair[], (0.0, 50.0))
sol = solve(prob, Rosenbrock23())

println("Simulation complete! Check `sol` for voltage traces of nrn_1, nrn_2, and nrn_3.")
plot(sol, idxs=[neurons[1].V, neurons[2].V, neurons[3].V],
            label=["Neuron 1" "Neuron 2" "Neuron 3"], ylabel="Voltage (mV)", xlabel="Time (ms)")
