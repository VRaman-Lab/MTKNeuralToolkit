# # Example 7: Calcium Dynamics & Nernst Potentials
# 
# This example introduces the Calcium dynamics machinery of MTKNeuralToolkit. 
# Real neurons maintain low intracellular Calcium, but voltage-gated Calcium 
# channels (CaV) allow Calcium ions to flow in during spikes. This influx is 
# tracked by a `CalciumTracker` pool, which slowly decays back to baseline.
# 
# Crucially, `CaVChannel` does not use a fixed reversal potential. It 
# dynamically calculates the Calcium Nernst potential based on the current 
# intracellular Calcium concentration. We also include a Calcium-activated 
# potassium channel (KCa) which uses the Calcium pool to modulate its gating.

using MTKNeuralToolkit
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq
using Plots

# ------------------------------------------------------------------------------
# 1. Define Gating Dynamics
# ------------------------------------------------------------------------------
# We use standard HH gates for Na and K (fast spiking)
hh_na_m = v -> (
    0.1 .* (v .+ 40.0) ./ (1.0 .- exp.(-(v .+ 40.0) ./ 10.0)),
    4.0 .* exp.(-(v .+ 65.0) ./ 18.0)
)
hh_na_h = v -> (
    0.07 .* exp.(-(v .+ 65.0) ./ 20.0),
    1.0 ./ (1.0 .+ exp.(-(v .+ 35.0) ./ 10.0))
)
sodium_gates = [GateSpec(:m, 3, 0.052, hh_na_m), GateSpec(:h, 1, 0.596, hh_na_h)]

hh_k_n = v -> (
    0.01 .* (v .+ 55.0) ./ (1.0 .- exp.(-(v .+ 55.0) ./ 10.0)),
    0.125 .* exp.(-(v .+ 65.0) ./ 80.0)
)
potassium_gates = [GateSpec(:n, 4, 0.317, hh_k_n)]

# Define a simple CaV channel gate using InfTau
CaV_m_inf(v) = 1.0 ./ (1.0 .+ exp.(-(v .+ 20.0) ./ 5.0))
CaV_tau_m(v) = 5.0 .+ 10.0 ./ (1.0 .+ exp.((v .+ 20.0) ./ 10.0))
CaV_dynamics = InfTau(CaV_m_inf, CaV_tau_m)
cav_gates = [GateSpec(:mCaV, 3, 0.0, CaV_dynamics)]

# Define a KCa channel gate. It depends on BOTH voltage and Calcium!
# Note the two arguments: (v, ca). We use the InfTauCa helper for this.
KCa_m_inf(v, ca) = (ca ./ (ca .+ 3.0)) ./ (1.0 .+ exp.(-(v .+ 20.0) ./ 5.0))
KCa_tau_m(v) = 20
KCa_dynamics = InfTauCa(KCa_m_inf, KCa_tau_m)
kca_gates = [GateSpec(:mKCa, 4, 0.0, KCa_dynamics)]

# ------------------------------------------------------------------------------
# 2. Build the Compartment
# ------------------------------------------------------------------------------
top = Scalar()

# Standard channels
@named cap = Capacitor(topology=top, C=1.0)
@named na  = GenericChannel(topology=top, g=100.0, E_rev=50.0,  gates=sodium_gates)
@named k   = GenericChannel(topology=top, g=36.0,  E_rev=-77.0, gates=potassium_gates)
@named leak= GenericChannel(topology=top, g=0.3,   E_rev=-54.4, gates=GateSpec[])

# Calcium channel. Note we don't pass E_rev! We pass Ca_out and nernst_factor.
# nernst_factor = (R*T)/(z*F) ~ 13.0 at room temp for natural log.
# The conversion_factor scales the current into a Ca2+ concentration rate.
@named cav = CaVChannel(topology=top, g=2.0, gates=cav_gates, Ca_out=3000.0, 
                        nernst_factor=13.0, conversion_factor=0.047)

# KCa channel
@named kca = KCaChannel(topology=top, g=5.0, E_rev=-80.0, gates=kca_gates)

# Calcium Tracker: sets the baseline intracellular Ca and the decay rate
ion_config = CalciumTracker(decay=200.0, Ca_init=0.05)

neuron = build_compartment(cap, [na, k, leak, cav, kca]; 
                           name=:neuron, 
                           V_init=-65.0, 
                           topology=top, 
                           ion_config=ion_config)

# ------------------------------------------------------------------------------
# 3. Build and Simulate the Network
# ------------------------------------------------------------------------------
drivers = [(1, 15.0)] # Current to elicit spikes and Calcium transients
net = build_acausal_network([neuron]; drivers=drivers, name=:ca_neuron)

println("Compiling Calcium neuron...")
sys = mtkcompile(net.sys)
prob = ODEProblem(sys, [], (0.0, 800.0), jac=true, sparse=true)

println("Solving...")
sol = solve(prob, Rosenbrock23(), reltol=1e-4, abstol=1e-4)

# ------------------------------------------------------------------------------
# 4. Plot the Results
# ------------------------------------------------------------------------------
# Let's look at the voltage, the dynamic Calcium concentration, and the KCa current.
p1 = plot(sol, idxs=[sys.neuron.cap.v], title="Membrane Potential", ylabel="V (mV)", legend=false)

# Access the internal calcium pool. The CalciumTracker creates a system named 
# `<compartment_name>_ca_pool`, which contains the `Ca` variable.
p2 = plot(sol, idxs=[sys.neuron.neuron_ca_pool.Ca], title="Intracellular Calcium", ylabel="[Ca] (uM)", legend=false)

p3 = plot(sol, idxs=[sys.neuron.kca.i], title="KCa Current", ylabel="I (mA)", xlabel="Time (ms)", legend=false)

plot(p1, p2, p3, layout=(3,1), size=(800, 900))
