using MTKNeuralToolkit
using ModelingToolkitStandardLibrary: Blocks
using ModelingToolkitStandardLibrary.Electrical: Resistor
using ModelingToolkit: mtkcompile, @named, System, SymbolicT
using OrdinaryDiffEq
using Plots

# =============================================================================
# 1. Define Channel Dynamics (Standard HH rates)
# =============================================================================

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
# 2. Build Compartments (Applying @named to all inner components)
# =============================================================================

# --- Neuron 1: Soma (Active) ---
@named soma1_cap = Capacitor(C=1.0)
@named na1 = GenericChannel(g=120.0, E_rev=50.0, gates=[GateSpec(:m, 3, 0.0, hh_na_m), GateSpec(:h, 1, 1.0, hh_na_h)])
@named k1  = GenericChannel(g=36.0, E_rev=-77.0, gates=[GateSpec(:n, 4, 0.0, hh_k_n)])
@named l1  = GenericChannel(g=0.3, E_rev=-54.4, gates=GateSpec[])
@named soma1 = build_compartment(soma1_cap, [na1, k1, l1])

# --- Neuron 1: Dendrite (Passive, smaller capacitance) ---
@named dend1_cap = Capacitor(C=0.5)
@named l2 = GenericChannel(g=0.1, E_rev=-54.4, gates=GateSpec[])
@named dend1 = build_compartment(dend1_cap, [l2])

# --- Neuron 2: Soma (Active) ---
@named soma2_cap = Capacitor(C=1.0)
@named na2 = GenericChannel(g=120.0, E_rev=50.0, gates=[GateSpec(:m, 3, 0.0, hh_na_m), GateSpec(:h, 1, 1.0, hh_na_h)])
@named k2  = GenericChannel(g=36.0, E_rev=-77.0, gates=[GateSpec(:n, 4, 0.0, hh_k_n)])
@named l3  = GenericChannel(g=0.3, E_rev=-54.4, gates=GateSpec[])
@named soma2 = build_compartment(soma2_cap, [na2, k2, l3])

# The flat list of all physical compartments in the network
compartments = [soma1, dend1, soma2]

# =============================================================================
# 3. Define Connections
# =============================================================================

# Axial Connection: Soma1 (idx 1) <-> Dendrite1 (idx 2) using OnePort Resistor
axial_conns = [
    (1, 2, (; name) -> Resistor(R=0.5, name=name))
]

# Synaptic Connection: Dendrite1 (idx 2) -> Soma2 (idx 3)
synapse_conns = [
    (2, 3, (; name) -> ChemicalSynapse(name=name, g_max=0.5, τ=5.0, v_th=-20.0, w=0.1, E_rev=0.0))
]

# =============================================================================
# 4. Build and Compile Network
# =============================================================================

# Increased amplitude and added an offset of 10.0 to push Soma 1 past its firing threshold
@named stim = Blocks.Sine(frequency=0.05, amplitude=30.0, offset=10.0)
drivers = [(1, stim)] # Drive Soma1

println("Building multi-compartment network...")
@named net = build_electrical_network(compartments, axial_conns, synapse_conns; drivers=drivers)

println("Compiling (this may take a moment)...")
# Conservative tearing is much faster for circuits!
net_compiled = mtkcompile(net; conservative=true, simplify=false)

# =============================================================================
# 5. Simulate and Plot
# =============================================================================

println("Solving...")
prob = ODEProblem(net_compiled, Pair[], (0.0, 100.0))
sol = solve(prob, Rosenbrock23(), saveat=0.1)

println("Simulation complete! Plotting traces...")
plot(sol, idxs=[soma1.V, dend1.V, soma2.V],
            label=["Soma 1 (Driven)" "Dendrite 1 (Passive)" "Soma 2 (Post-synaptic)"], 
            ylabel="Voltage (mV)", xlabel="Time (ms)", lw=2)
