using MTKNeuralToolkit
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq
using Plots

# === Define shared gates ===
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

# === Build Excitatory Population (N=30) ===
N_E = 30
@named cap_E = Capacitor(N=N_E, C=1.0)
@named na_E = GenericChannel(N=N_E, g=120.0, E_rev=50.0, gates=sodium_gates)
@named k_E  = GenericChannel(N=N_E, g=36.0, E_rev=-77.0, gates=potassium_gates)
@named leak_E = GenericChannel(N=N_E, g=0.3, E_rev=-54.4, gates=GateSpec[])

pop_E = build_compartment(cap_E, [na_E, k_E, leak_E]; name=:pop_E, V_init=-65.0,topology=Vectorized(30))

# === Build Inhibitory Population (N=10) ===
N_I = 10
@named cap_I = Capacitor(N=N_I, C=1.0)
@named na_I = GenericChannel(N=N_I, g=120.0, E_rev=50.0, gates=sodium_gates)
@named k_I  = GenericChannel(N=N_I, g=36.0, E_rev=-77.0, gates=potassium_gates)
@named leak_I = GenericChannel(N=N_I, g=0.3, E_rev=-54.4, gates=GateSpec[])

pop_I = build_compartment(cap_I, [na_I, k_I, leak_I]; name=:pop_I, V_init=-65.0, topology=Vectorized(N_I))

# === Define Connectivity Matrices ===
W_EE = 0.05 .* rand(N_E, N_E)   # E -> E
W_EI = 0.1  .* rand(N_I, N_E)   # E -> I
W_IE = 0.2  .* rand(N_E, N_I)   # I -> E
W_II = 0.1  .* rand(N_I, N_I)   # I -> I

# === Build Synapse Blocks ===
syn_EE = build_synapse_block(pop_E, pop_E, W_EE; name=:syn_EE, E_rev=0.0)
syn_EI = build_synapse_block(pop_E, pop_I, W_EI; name=:syn_EI, E_rev=0.0)
syn_IE = build_synapse_block(pop_I, pop_E, W_IE; name=:syn_IE, E_rev=-80.0)
syn_II = build_synapse_block(pop_I, pop_I, W_II; name=:syn_II, E_rev=-80.0)

synapse_specs = [syn_EE, syn_EI, syn_IE, syn_II]

# === Drivers ===
# Drive only the Excitatory population (index 1) with a constant 10.0
# build_acausal_network will broadcast the scalar to the whole N_E population
drivers = [(1, 15.0)]

# === Build Network ===
net = build_acausal_network([pop_E, pop_I]; synapse_specs=synapse_specs, drivers=drivers)

@time net_compiled = mtkcompile(net.sys)
prob = ODEProblem(net_compiled, [], (0.0, 100.0))
@time sol = solve(prob, Rosenbrock23())

# === Plot ===
# Note the fix: cap_E and cap_I instead of soma
p1 = plot(sol, idxs=[net_compiled.pop_E.cap_E.v...], title="Excitatory Population", legend=false)
p2 = plot(sol, idxs=[net_compiled.pop_I.cap_I.v...], title="Inhibitory Population", legend=false)

plot(p1, p2, layout=(2,1), size=(800,500))
