using MTKNeuralToolkit
using ModelingToolkit: mtkcompile, @named, connect, Equation, SymbolicT, t_nounits as t, System
using OrdinaryDiffEq
using Plots

N = 30

# === Build vectorized HH compartment ===
@named soma = Capacitor(N=N, C=1.0)

hh_na_m = v -> (
    0.182 .* (v .+ 35.0) ./ (1.0 .- exp.(-(v .+ 35.0) ./ 9.0)),
    -0.124 .* (v .+ 35.0) ./ (1.0 .- exp.((v .+ 35.0) ./ 9.0))
)
hh_na_h = v -> (
    0.25 .* exp.(-(v .+ 90.0) ./ 12.0),
    0.25 .* (exp.((v .+ 62.0) ./ 6.0)) ./ exp.(-(v .+ 90.0) ./ 12.0)
)
sodium_gates = [
    GateSpec(:m, 3, 0.0, hh_na_m),
    GateSpec(:h, 1, 0.0, hh_na_h)
]

hh_k_n = v -> (
    0.02 .* (v .- 25.0) ./ (1.0 .- exp.(-(v .- 25.0) ./ 9.0)),
    -0.002 .* (v .- 25.0) ./ (1.0 .- exp.((v .- 25.0) ./ 9.0))
)
potassium_gates = [
    GateSpec(:n, 4, 0.0, hh_k_n)
]

@named sodium_channel = GenericChannel(N=N, g=120.0, E_rev=50.0, gates=sodium_gates)
@named potassium_channel = GenericChannel(N=N, g=36.0, E_rev=-77.0, gates=potassium_gates)
@named leak_channel = GenericChannel(N=N, g=0.3, E_rev=-54.4, gates=GateSpec[])

hh = build_compartment(soma, [sodium_channel, potassium_channel, leak_channel];
                        name=:hh, V_init=-65.0, N=N)

# === Synapses with convergence and chains ===
#  1→2, 4→2    convergent excitatory onto neuron 2
#  1→3, 5→3    convergent mixed (exc + inh) onto neuron 3
#  3→7         chain: 3 is a post-synaptic target AND a pre-synaptic source
#  2→8         chain: 2 is a post-synaptic target AND a pre-synaptic source

@named syn_1to2 = ExpSynapse(g_max=2.0,  τ=5.0,  E_rev=0.0,   V_th=-20.0, slope=2.0)
@named syn_4to2 = ExpSynapse(g_max=1.5,  τ=5.0,  E_rev=0.0,   V_th=-20.0, slope=2.0)
@named syn_1to3 = ExpSynapse(g_max=1.0,  τ=8.0,  E_rev=0.0,   V_th=-20.0, slope=2.0)
@named syn_5to3 = ExpSynapse(g_max=1.2,  τ=8.0,  E_rev=-80.0, V_th=-20.0, slope=2.0)  # inhibitory
@named syn_3to7 = ExpSynapse(g_max=2.0,  τ=5.0,  E_rev=0.0,   V_th=-20.0, slope=2.0)
@named syn_2to8 = ExpSynapse(g_max=1.5,  τ=5.0,  E_rev=0.0,   V_th=-20.0, slope=2.0)

synapse_list = [
    (pre=1, post=2, syn=syn_1to2),
    (pre=4, post=2, syn=syn_4to2),
    (pre=1, post=3, syn=syn_1to3),
    (pre=5, post=3, syn=syn_5to3),
    (pre=3, post=7, syn=syn_3to7),
    (pre=2, post=8, syn=syn_2to8),
]

# === Build network ===
eqs = Equation[]
all_systems = System[hh.sys]

@named gnd = Ground(N=N)
push!(all_systems, gnd)
push!(eqs, connect(gnd.g, hh.interfaces.n_pin))

for i in 1:N
    push!(eqs, hh.interfaces.I_ext[i] ~ Float64(i))
end

# Wire synapses — pre-collect by target to handle convergence correctly
syn_by_target = Dict{Int, Vector{SymbolicT}}()

for conn in synapse_list
    push!(all_systems, conn.syn)
    push!(eqs, conn.syn.V_pre  ~ hh.interfaces.V[conn.pre])
    push!(eqs, conn.syn.V_post ~ hh.interfaces.V[conn.post])

    target = conn.post
    haskey(syn_by_target, target) || (syn_by_target[target] = SymbolicT[])
    push!(syn_by_target[target], conn.syn.I_syn)
end

# One equation per target — sum if convergent, alias if single
for (post_idx, currents) in syn_by_target
    if length(currents) == 1
        push!(eqs, hh.interfaces.I_syn[post_idx] ~ currents[1])
    else
        push!(eqs, hh.interfaces.I_syn[post_idx] ~ sum(currents))
    end
end

# Ground non-synapsed I_syn
for i in 1:N
    if !haskey(syn_by_target, i)
        push!(eqs, hh.interfaces.I_syn[i] ~ 0.0)
    end
end

push!(eqs, hh.interfaces.p_pin.i ~ zeros(Float64, N))

# === Build and solve ===
@named net = System(eqs, t, SymbolicT[], SymbolicT[]; systems=all_systems, name=:net)
net_compiled = mtkcompile(net)
prob = ODEProblem(net_compiled, [], (0.0, 50.0))
sol = solve(prob, Rosenbrock23())

# === Plot ===
p1 = plot(sol, idxs=[net_compiled.hh.soma.v...],
          title="All N=$N Neurons", xlabel="Time", ylabel="V (mV)")

p2 = plot(sol, idxs=[net_compiled.hh.soma.v[1],
                      net_compiled.hh.soma.v[4],
                      net_compiled.hh.soma.v[2]],
          label=["Pre 1" "Pre 4" "Post 2"],
          title="Convergent exc: 1→2 + 4→2", xlabel="Time", ylabel="V (mV)")

p3 = plot(sol, idxs=[net_compiled.hh.soma.v[1],
                      net_compiled.hh.soma.v[5],
                      net_compiled.hh.soma.v[3]],
          label=["Pre 1 (exc)" "Pre 5 (inh)" "Post 3"],
          title="Convergent mixed: 1→3 + 5→3", xlabel="Time", ylabel="V (mV)")

p4 = plot(sol, idxs=[net_compiled.hh.soma.v[2], net_compiled.hh.soma.v[8],
                      net_compiled.hh.soma.v[3], net_compiled.hh.soma.v[7]],
          label=["Neuron 2" "Neuron 8" "Neuron 3" "Neuron 7"],
          title="Chains: 2→8, 3→7", xlabel="Time", ylabel="V (mV)")

plot(p1, p2, p3, p4, layout=(4,1), size=(800,1000))

# Overlay individual synapse currents with their computed sum
plot(sol, idxs=[net_compiled.syn_1to2.I_syn, net_compiled.syn_4to2.I_syn],
     label=["syn 1→2" "syn 4→2"], title="Convergent currents on neuron 2",
     xlabel="Time", ylabel="I_syn")

# Add the computed sum as a dashed line
ts = 0.0:0.1:50.0
i_sum = [sol(t; idxs=net_compiled.syn_1to2.I_syn) + sol(t; idxs=net_compiled.syn_4to2.I_syn) 
         for t in ts]
plot!(ts, i_sum, ls=:dash, label="sum", lw=2)

