# ==============================================================================
# Example 1: Building a Single-Compartment Hodgkin-Huxley Neuron from Scratch
# ==============================================================================
# 
# This example introduces the core acausal electrical primitives of 
# MTKNeuralToolkit. It demonstrates how to define standard ion channel 
# gating dynamics from first principles and assemble them into a single 
# point-neuron compartment.
#
# To run this, you will need to activate the examples environment: launch julia from the terminal (in root directory of package) as `julia --project=examples`


using MTKNeuralToolkit
using ModelingToolkit: mtkcompile, @named, t_nounits as t
using OrdinaryDiffEq
using Plots




# ------------------------------------------------------------------------------
# 1. Define Ion Channel Gating Dynamics
# ------------------------------------------------------------------------------
# We use the standard Hodgkin-Huxley formulations (Dayan & Abbott) where 
# V_rest = -65 mV.

hh_na_m = v -> (
    0.1 .* (v .+ 40.0) ./ (1.0 .- exp.(-(v .+ 40.0) ./ 10.0)),  # alpha_m
    4.0 .* exp.(-(v .+ 65.0) ./ 18.0)                           # beta_m
)

hh_na_h = v -> (
    0.07 .* exp.(-(v .+ 65.0) ./ 20.0),                         # alpha_h
    1.0 ./ (1.0 .+ exp.(-(v .+ 35.0) ./ 10.0))                  # beta_h
)

# ------------------------------------------------------------------------------
# Defining Gating Dynamics: Alpha/Beta vs. Infinity/Tau
# ------------------------------------------------------------------------------

# Method 1: Alpha/Beta Formulation
hh_k_n_ab = v -> (
    0.01 .* (v .+ 55.0) ./ (1.0 .- exp.(-(v .+ 55.0) ./ 10.0)), # alpha_n
    0.125 .* exp.(-(v .+ 65.0) ./ 80.0)                         # beta_n
)

# Method 2: Infinity/Tau Formulation (using the InfTau helper)
# MTKNeuralToolkit provides the `InfTau` helper to automatically convert these 
# into the alpha/beta formulation used internally:
#   alpha = inf / tau
#   beta  = (1 - inf) / tau
#
# Here, we define n_inf and tau_n mathematically equivalent to Method 1:
n_alpha(v) = 0.01 .* (v .+ 55.0) ./ (1.0 .- exp.(-(v .+ 55.0) ./ 10.0))
n_beta(v)  = 0.125 .* exp.(-(v .+ 65.0) ./ 80.0)
n_inf(v)   = n_alpha(v) ./ (n_alpha(v) .+ n_beta(v))
tau_n(v)   = 1.0 ./ (n_alpha(v) .+ n_beta(v))
hh_k_n_inftau = InfTau(n_inf, tau_n)

# ------------------------------------------------------------------------------
# 2. Define Gate Specifications
# ------------------------------------------------------------------------------
# At V = -65 mV, the steady-state values are:
# m = 0.052, h = 0.596, n = 0.317
sodium_gates = [
    GateSpec(:m, 3, 0.052, hh_na_m), 
    GateSpec(:h, 1, 0.596, hh_na_h)
]

# We use the InfTau formulation here to demonstrate it, but we could just as 
# easily swap it for `hh_k_n_ab` (the alpha/beta version) and get the exact same result.
# NOTE: We name the K gate `n_gate` to avoid a namespace clash with the negative 
# electrical pin `n` in MTK's OnePort!
potassium_gates = [
    GateSpec(:n_gate, 4, 0.317, hh_k_n_inftau)
]


# ------------------------------------------------------------------------------
# 3. Build the Electrical Components
# ------------------------------------------------------------------------------
# We instantiate the capacitor and our ion channels.
# `GenericChannel` automatically hooks up the acausal OnePort and generates 
# the correct ODEs for the gating variables under the hood.

top = Scalar() # Define a scalar (single-neuron) topology

@named soma_cap = Capacitor(topology=top, C=1.0)

@named na_ch = GenericChannel(topology=top, g=120.0, E_rev=50.0,  gates=sodium_gates)
@named k_ch  = GenericChannel(topology=top, g=36.0,  E_rev=-77.0, gates=potassium_gates)
@named leak  = GenericChannel(topology=top, g=0.3,   E_rev=-54.4, gates=GateSpec[]) # No gates = pure leak

# ------------------------------------------------------------------------------
# 4. Assemble the Compartment
# ------------------------------------------------------------------------------
# `build_compartment` connects all the positive pins to the membrane voltage,
# grounds the negative pins, and injects the currents. It returns a `Compartment`
# struct containing the MTK System and exposed interfaces.

channels = [na_ch, k_ch, leak]

soma = build_compartment(soma_cap, channels; 
                         name=:soma, 
                         V_init=-65.0, 
                         topology=top)

# ------------------------------------------------------------------------------
# 5. Build and Simulate the Network
# ------------------------------------------------------------------------------
# A single neuron is just a network with one node. We drive it with a constant
# 10.0 mA current injection and solve.

# NOTE: The first time you run this, Julia will JIT-compile the MTK symbolic 
# system and the differential equation solver. This "Time To First Plot" (TTFP) 
# can take 10-30 seconds. Subsequent runs (or changing parameters and re-running) 
# will be nearly instantaneous!

drivers = [(1, 10.0)] # (Index 1, Current 10.0)

net = build_acausal_network([soma]; drivers=drivers, name=:single_neuron)

println("Compiling single compartment...")
sys = mtkcompile(net.sys)
prob = ODEProblem(sys, [], (0.0, 100.0))

println("Solving...")
sol = solve(prob, Rosenbrock23(), reltol=1e-4, abstol=1e-4)


# ------------------------------------------------------------------------------
# 6. Plot the Results
# ------------------------------------------------------------------------------
# Because MTKNeuralToolkit is built on ModelingToolkit, EVERY variable in the 
# system (gating states, alpha/beta rates, local channel currents) is a named 
# symbolic variable. You don't need custom logging callbacks to observe them; 
# you just access them via the compiled system's namespace.

# 6a. Plot the main membrane voltage
p1 = plot(sol, idxs=[sys.soma.soma_cap.v], 
          title="Example 1: Membrane Potential", 
          ylabel="V (mV)", legend=false)

# 6b. Plot the internal HH gating variables (m, h, n)
# We reach into the sodium channel (na_ch) and potassium channel (k_ch) 
# components we created in Step 3.
p2 = plot(sol, idxs=[sys.soma.na_ch.m, sys.soma.na_ch.h, sys.soma.k_ch.n_gate], 
          title="Gating Variables", 
          labels=["m (Na+)" "h (Na+)" "n (K+)"], 
          ylabel="Fraction open", legend=:right)

# 6c. Plot the individual currents flowing through each channel
p3 = plot(sol, idxs=[sys.soma.na_ch.i, sys.soma.k_ch.i, sys.soma.leak.i], 
          title="Channel Currents", 
          labels=["I_Na" "I_K" "I_Leak"], 
          ylabel="Current (mA)", xlabel="Time (ms)", legend=:right)

# Increase canvas height to fit all three plots comfortably
plot(p1, p2, p3, layout=(3,1), size=(800, 900))

