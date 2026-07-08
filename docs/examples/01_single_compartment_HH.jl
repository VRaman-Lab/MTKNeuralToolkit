# # Example 1: Building a Single-Compartment Hodgkin-Huxley Neuron from Scratch
# 
# This example introduces the core acausal electrical primitives of MTKNeuralToolkit. We'll use a GateSpec helper to quickly make customisable ion channels and hook them up to run a hodgkin huxley neuron. Note that the next example builds the ion channels from first principles, which is barely any harder.
#

using MTKNeuralToolkit
using ModelingToolkit: mtkcompile, @named, t_nounits as t
using OrdinaryDiffEq
using Plots

# ## 1. Define Ion Channel Gating Dynamics
# We use the standard Hodgkin-Huxley formulations (Dayan & Abbott) where 
# V_rest = -65 mV. Here, we return the alpha and beta rates as a tuple.

hh_na_m = v -> (
    0.1 .* (v .+ 40.0) ./ (1.0 .- exp.(-(v .+ 40.0) ./ 10.0)),  #alpha_m
    4.0 .* exp.(-(v .+ 65.0) ./ 18.0)                           #beta_m
)

hh_na_h = v -> (
    0.07 .* exp.(-(v .+ 65.0) ./ 20.0),                         #alpha_h
    1.0 ./ (1.0 .+ exp.(-(v .+ 35.0) ./ 10.0))                 #beta_h
)

# ### Defining Gating Dynamics: Alpha/Beta vs. Infinity/Tau
# There are two common ways to define gating dynamics. 
# 
# #### Method 1: Alpha/Beta Formulation

hh_k_n_ab = v -> (
    0.01 .* (v .+ 55.0) ./ (1.0 .- exp.(-(v .+ 55.0) ./ 10.0)), #alpha_n
    0.125 .* exp.(-(v .+ 65.0) ./ 80.0)                        #beta_n
)

# #### Method 2: Infinity/Tau Formulation (using the InfTau helper)
# Instead of writing out the alpha/beta functions, you might only have the steady-state 
# (`inf`) and time constant (`tau`) curves. MTKNeuralToolkit provides the `InfTau` helper 
# to convert these into the internal alpha/beta formulation:
#   alpha = inf / tau
#   beta  = (1 - inf) / tau
#
# Here, we define `n_inf` and `tau_n` mathematically equivalent to Method 1 so you can 
# see how it maps over:

n_alpha(v) = 0.01 .* (v .+ 55.0) ./ (1.0 .- exp.(-(v .+ 55.0) ./ 10.0))
n_beta(v)  = 0.125 .* exp.(-(v .+ 65.0) ./ 80.0)
n_inf(v)   = n_alpha(v) ./ (n_alpha(v) .+ n_beta(v))
tau_n(v)   = 1.0 ./ (n_alpha(v) .+ n_beta(v))
hh_k_n_inftau = InfTau(n_inf, tau_n)

# ## 2. Define Gate Specifications
# At V = -65 mV, the steady-state values for the gates are:
# m = 0.052, h = 0.596, n = 0.317

sodium_gates = [
    GateSpec(:m, 3, 0.052, hh_na_m), 
    GateSpec(:h, 1, 0.596, hh_na_h)
]

# We're using the InfTau formulation (`hh_k_n_inftau`) here to show how it works, 
# but swapping it for `hh_k_n_ab` (the alpha/beta version) gives you the exact same 
# dynamics.
# 
# NOTE: We name the K gate `n_gate` instead of just `n` to avoid a namespace clash 
# with the negative electrical pin `n` in MTK's OnePort.

potassium_gates = [
    GateSpec(:n_gate, 4, 0.317, hh_k_n_inftau)
]

# ## 3. Build the Electrical Components
# First, we instantiate the capacitor and our ion channels. 
# When you pass a `GateSpec` array to `GenericChannel`, it creates the ODEs for 
# the gating variables and connects the acausal OnePort for you.

top = Scalar() #Define a single-neuron topology

@named soma_cap = Capacitor(topology=top, C=1.0)
@named na_ch = GenericChannel(topology=top, g=120.0, E_rev=50.0,  gates=sodium_gates)
@named k_ch  = GenericChannel(topology=top, g=36.0,  E_rev=-77.0, gates=potassium_gates)
@named leak  = GenericChannel(topology=top, g=0.3,   E_rev=-54.4, gates=GateSpec[]) #No gates = pure leak

# ## 4. Assemble the Compartment
# `build_compartment` connects all the positive pins  to the membrane voltage, grounds the negative pins, and wires up the injected currents. 
# It returns a `Compartment` struct containing the MTK System and its exposed interfaces.

channels = [na_ch, k_ch, leak]

soma = build_compartment(soma_cap, channels; 
                         name=:soma, 
                         V_init=-65.0, 
                         topology=top)

# ## 5. Build and Simulate the Network
# A single neuron is just a network with one node. We'll drive it with a constant 
# 10.0 mA current injection and solve it. Later examples make the simple extension of driving with time-varying currents.
#
# Note on compilation: The first time you run this, Julia has to JIT-compile the 
# symbolic system and the differential equation solver. This "Time To First Plot" (TTFP) 
# might take 10-30 seconds. Subsequent runs, or tweaking parameters and re-running, 
# will be much faster.

drivers = [(1, 10.0)] #(Index 1, Current 10.0)

net = build_acausal_network([soma]; drivers=drivers, name=:single_neuron)

println("Compiling single compartment...")
sys = mtkcompile(net.sys)
prob = ODEProblem(sys, [], (0.0, 100.0))

println("Solving...")
sol = solve(prob, Rosenbrock23(), reltol=1e-4, abstol=1e-4)

# ## 6. Plot the Results
# Since MTKNeuralToolkit is built on ModelingToolkit, every variable in the system 
# (gating states, alpha/beta rates, local channel currents) is a named symbolic 
# variable. You don't need custom logging callbacks to observe them; you just access 
# them via the compiled system's namespace.

# ### 6a. Plot the main membrane voltage
p1 = plot(sol, idxs=[sys.soma.soma_cap.v], 
          title="Example 1: Membrane Potential", 
          ylabel="V (mV)", legend=false)

# ### 6b. Plot the internal HH gating variables (m, h, n)
# Here we reach into the sodium channel (`na_ch`) and potassium channel (`k_ch`) 
# components we created back in Step 3.

p2 = plot(sol, idxs=[sys.soma.na_ch.m, sys.soma.na_ch.h, sys.soma.k_ch.n_gate], 
          title="Gating Variables", 
          labels=["m (Na+)" "h (Na+)" "n (K+)"], 
          ylabel="Fraction open", legend=:right)

# ### 6c. Plot the individual currents flowing through each channel
p3 = plot(sol, idxs=[sys.soma.na_ch.i, sys.soma.k_ch.i, sys.soma.leak.i], 
          title="Channel Currents", 
          labels=["I_Na" "I_K" "I_Leak"], 
          ylabel="Current (mA)", xlabel="Time (ms)", legend=:right)

plot(p1, p2, p3, layout=(3,1), size=(800, 900))
