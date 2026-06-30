using MTKNeuralToolkit
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq
using Plots
using Random

# === Shared gate definitions ===
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

# === Population A: 20 excitatory neurons ===
Na = 20
@named somaA = Capacitor(N=Na, C=1.0)
@named naA   = GenericChannel(N=Na, g=120.0, E_rev=50.0,  gates=sodium_gates)
@named kA    = GenericChannel(N=Na, g=36.0,  E_rev=-77.0, gates=potassium_gates)
@named leakA = GenericChannel(N=Na, g=0.3,   E_rev=-54.4, gates=GateSpec[])

popA = build_compartment(somaA, [naA, kA, leakA]; name=:popA, V_init=-65.0, N=Na)

# === Population B: 15 neurons (no external drive — relies on A) ===
Nb = 15
@named somaB = Capacitor(N=Nb, C=1.0)
@named naB   = GenericChannel(N=Nb, g=120.0, E_rev=50.0,  gates=sodium_gates)
@named kB    = GenericChannel(N=Nb, g=36.0,  E_rev=-77.0, gates=potassium_gates)
@named leakB = GenericChannel(N=Nb, g=0.3,   E_rev=-54.4, gates=GateSpec[])

popB = build_compartment(somaB, [naB, kB, leakB]; name=:popB, V_init=-65.0, N=Nb)

# === Connectivity matrices ===
Random.seed!(42)

# Intra-A: ring (neuron i → neuron i+1)
W_AA = zeros(Na, Na)
for i in 1:Na-1
    W_AA[i+1, i] = 1.0
end
W_AA[1, Na] = 1.0

# A → B: random sparse, each B neuron receives from ~3 A neurons
W_AB = zeros(Nb, Na)
for j in 1:Nb
    pres = randperm(Na)[1:3]  # 3 random pre neurons
    for p in pres
        W_AB[j, p] = 1.0
    end
end

# === Synapse blocks ===
syn_intra_A = build_synapse_block(popA, popA, W_AA;
                                   name=:syn_AA, g_max=2.0, τ=5.0,
                                   E_rev=0.0, V_th=-20.0, slope=2.0)

syn_A_to_B = build_synapse_block(popA, popB, W_AB;
                                  name=:syn_AB, g_max=3.0, τ=8.0,
                                  E_rev=0.0, V_th=-20.0, slope=2.0)

# === Drivers: only population A gets external drive ===
drivers = [(1, collect(Float64, 1:Na))]  # popA (compartment 1) gets graded current

# === Build one network with all compartments ===
net = build_acausal_network([popA, popB];
                            synapse_specs=[syn_intra_A, syn_A_to_B],
                            drivers=drivers)

net_compiled = mtkcompile(net.sys)
prob = ODEProblem(net_compiled, [], (0.0, 50.0))
sol = solve(prob, Rosenbrock23())

# === Plot ===
p1 = plot(sol, idxs=[net_compiled.popA.somaA.v...],
          title="Population A (driven)", xlabel="Time", ylabel="V (mV)")

p2 = plot(sol, idxs=[net_compiled.popB.somaB.v...],
          title="Population B (driven only by A→B synapses)", xlabel="Time", ylabel="V (mV)")

p3 = plot(sol, idxs=[net_compiled.syn_AB.s[1:5]...],
          title="A→B synapse gating states (first 5)", xlabel="Time", ylabel="s")

plot(p1, p2, p3, layout=(3,1), size=(800,750))
