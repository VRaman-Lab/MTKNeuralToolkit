# # Example 4: Vectorized Populations (E/I Network)
# 
# Let's scale up to population-level models. Instead of building thousands of 
# individual scalar systems, we can use the `Vectorized` topology. A single 
# `GenericChannel` with a vectorized topology expands to N elements. We wire dense 
# excitatory/inhibitory populations together using a weight matrix and the 
# `build_synapse_block` helper.
#
# The main benefit of the `Vectorized` topology is that it saves compile time. 
# Compiling 1,000 scalar systems requires MTK to process 1,000 separate symbolic 
# graphs. Using `Vectorized(1000)` collapses the entire population into a single 
# system that uses array operations.
#
# !!! note
#     An annoyance is the need to build both scalar and vector definitions for the components. It would be nice (and maybe exists?) for MTK to have some automatic way of turning a scalar component into a vectorised one? I spent some time attempting this with postwalk, and got it running. But it was messy and dependeng on the MTK internal API so decided to demand a separate set of equations for a vectorised component.

using MTKNeuralToolkit
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq
using Plots

# ------------------------------------------------------------------------------
# 1. Define Shared Gating Dynamics
# ------------------------------------------------------------------------------
# These functions use broadcasting (.) so they work identically for both scalar 
# and vectorized topologies.

hh_na_m = v -> (
    0.182 .* (v .+ 35.0) ./ (1.0 .- exp.(-(v .+ 35.0) ./ 9.0)),
    -0.124 .* (v .+ 35.0) ./ (1.0 .- exp.((v .+ 35.0) ./ 9.0))
)
hh_na_h = v -> (
    0.25 .* exp.(-(v .+ 90.0) ./ 12.0),
    0.25 .* (exp.((v .+ 62.0) ./ 6.0)) ./ exp.(-(v .+ 90.0) ./ 12.0)
)
sodium_gates = [GateSpec(:m, 3, 0.0, hh_na_m), GateSpec(:h, 1, 0.0, hh_na_h)]

hh_k_n = v -> (
    0.02 .* (v .- 25.0) ./ (1.0 .- exp.(-(v .- 25.0) ./ 9.0)),
    -0.002 .* (v .- 25.0) ./ (1.0 .- exp.((v .- 25.0) ./ 9.0))
)
potassium_gates = [GateSpec(:n, 4, 0.0, hh_k_n)]

# ------------------------------------------------------------------------------
# 2. Define Topologies and Build Populations
# ------------------------------------------------------------------------------
N_E = 15 
N_I = 5  
top_E = Vectorized(N_E)
top_I = Vectorized(N_I)

function build_population(name::Symbol, top)
    #Let's make the sodium conductance heterogeneous across the population
    gNa_heterogeneous = collect(range(119.0, 121.0, length=top.N)) 
    
    @named cap = Capacitor(topology=top, C=1.0)
    @named na  = GenericChannel(topology=top, g=gNa_heterogeneous, E_rev=50.0,  gates=sodium_gates)
    @named k   = GenericChannel(topology=top, g=36.0,  E_rev=-77.0, gates=potassium_gates)
    @named leak= GenericChannel(topology=top, g=0.3,   E_rev=-54.4, gates=GateSpec[])
    
    return build_compartment(cap, [na, k, leak]; name=name, V_init=-65.0, topology=top)
end

pop_E = build_population(:pop_E, top_E)
pop_I = build_population(:pop_I, top_I)

# ------------------------------------------------------------------------------
# 3. Define Connectivity Matrices
# ------------------------------------------------------------------------------
# The weight matrix W maps presynaptic populations to postsynaptic populations.
# Dimensions must be (N_post, N_pre).

W_EE = 0.5 .* rand(N_E, N_E)   #E -> E
W_EI = 1.0 .* rand(N_I, N_E)   #E -> I
W_IE = 2.0 .* rand(N_E, N_I)   #I -> E
W_II = 1.0 .* rand(N_I, N_I)   #I -> I

# ------------------------------------------------------------------------------
# 4. Build Synapse Blocks
# ------------------------------------------------------------------------------
# `build_synapse_block` sets up the vectorized synapse matrices and creates the 
# SynapseSpecs for the network builder.

syn_EE = build_synapse_block(pop_E, pop_E, W_EE; name=:syn_EE, E_rev=0.0)
syn_EI = build_synapse_block(pop_E, pop_I, W_EI; name=:syn_EI, E_rev=0.0)
syn_IE = build_synapse_block(pop_I, pop_E, W_IE; name=:syn_IE, E_rev=-80.0)
syn_II = build_synapse_block(pop_I, pop_I, W_II; name=:syn_II, E_rev=-80.0)

synapse_specs = [syn_EE, syn_EI, syn_IE, syn_II]

# ------------------------------------------------------------------------------
# 5. Driving Stimuli & Network Assembly
# ------------------------------------------------------------------------------
# Give the excitatory population a constant current kick to start the activity
drivers = [(1, 25.0)]

net = build_acausal_network([pop_E, pop_I]; synapse_specs=synapse_specs, drivers=drivers)

println("Compiling vectorized network...")
sys = mtkcompile(net.sys)

# Because the underlying equations are array-based, the Jacobian of this system 
# is highly sparse: most neurons only affect their own gating variables, with 
# off-diagonal elements coming only from the synapse blocks.
# 
# Passing `jac=true, sparse=true` tells the solver to compute an analytical 
# Jacobian and leverage sparse linear algebra. This speeds simulation, 
# as the solver only computes non-zero derivatives instead of a dense matrix. 
# Sparse Jacobians are also essential for fast automatic differentiation (AD), 
# which we will need later for parameter estimation.

prob = ODEProblem(sys, [], (0.0, 100.0), jac=true, sparse=true)

# Let's visualize the sparsity pattern using Plots.spy. You'll see a heavy 
# block-diagonal structure (intrinsic dynamics) with off-diagonal bands 
# representing the synaptic couplings between the E and I populations.
println("Plotting Jacobian sparsity pattern...")
p_jac = spy(prob.f.jac_prototype, 
            title="Jacobian Sparsity Pattern", 
            legend=false)

println("Solving...")
sol = solve(prob, Rosenbrock23())

# ------------------------------------------------------------------------------
# 6. Plot the Results
# ------------------------------------------------------------------------------
# We splat the voltage array (...) to plot all individual elements in the population. Couldn't find a way to add numberings with MTK but I assume it exists...?
p1 = plot(sol, idxs=[sys.pop_E.cap.v...], 
          title="Excitatory Population", legend=false, ylabel="V (mV)")
p2 = plot(sol, idxs=[sys.pop_I.cap.v...], 
          title="Inhibitory Population", legend=false, ylabel="V (mV)", xlabel="Time (ms)")

# Combine the simulation plots with the sparsity plot
plot(p1, p2, p_jac, layout=(3,1), size=(800, 800))
