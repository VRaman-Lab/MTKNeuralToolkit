using MTKNeuralToolkit
using ModelingToolkit: mtkcompile, @named, System, t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using Plots

# 1. Define Channel Dynamics
hh_na_m = v -> (0.182 * (v + 35.0) / (1.0 - exp(-(v + 35.0) / 9.0)), -0.124 * (v + 35.0) / (1.0 - exp((v + 35.0) / 9.0)))
hh_na_h = v -> (0.25 * exp(-(v + 90.0) / 12.0), 0.25 * (exp((v + 62.0) / 6.0)) / exp(-(v + 90.0) / 12.0))
hh_k_n = v -> (0.02 * (v - 25.0) / (1.0 - exp(-(v - 25.0) / 9.0)), -0.002 * (v - 25.0) / (1.0 - exp((v - 25.0) / 9.0)))

# 2. Build Compartments
@named soma1_cap = Capacitor(C=1.0)
@named na1 = GenericChannel(g=120.0, E_rev=50.0, gates=[GateSpec(:m, 3, 0.0, hh_na_m), GateSpec(:h, 1, 1.0, hh_na_h)])
@named k1  = GenericChannel(g=36.0, E_rev=-77.0, gates=[GateSpec(:n, 4, 0.0, hh_k_n)])
@named l1  = GenericChannel(g=0.3, E_rev=-54.4, gates=GateSpec[])
soma_comp = build_floating_compartment(soma1_cap, [na1, k1, l1], name=:soma)

@named dend1_cap = Capacitor(C=0.5)
@named l2 = GenericChannel(g=0.1, E_rev=-54.4, gates=GateSpec[])
dend_comp = build_floating_compartment(dend1_cap, [l2], name=:dend)

# 3. Build Cell
axial_conns = [(1, 2, 0.5)]
cell = build_cell([soma_comp, dend_comp], axial_conns; drivers=[], ground_undriven=false, name=:hh_cell)


# 4. Build Network with Synapses
# Connect Cell 1, Compartment 1 (Soma) -> Cell 2, Compartment 1 (Soma)
synapses = [(1, 1, 2, 1, (; name) -> AlphaSynapse(name=name, g_max=5.0))]

println("Building Network...")
# ground_inputs=false exposes the un-synapsed I_ext variables as MTK inputs
net = build_network(cell, 5; synapse_connections=synapses, ground_inputs=false, name=:pop)

# Put this right before: net_compiled = mtkcompile(net.sys)
println("Checking for nothing or type issues in equations:")
for eq in equations(net.sys)
    if occursin("nothing", string(eq))
        println("FOUND NOTHING: ", eq)
    end
end

println("\nNetwork Unknowns (first 5):")
for u in unknowns(net.sys)[1:min(5, end)]
    println(u, " -> Type: ", typeof(u))
end

println("\nNetwork Parameters (first 5):")
for p in parameters(net.sys)[1:min(5, end)]
    println(p, " -> Type: ", typeof(p))
end



# 5. Compile the Network ONCE
println("Compiling Network...")
net_compiled = mtkcompile(net.sys, inputs=net.inputs)

# 6. Setup Inputs and ODE Problem
println("Setting up inputs...")
u0 = Dict()

# Initialize all exposed inputs to 0.0
for input_var in net.inputs
    u0[input_var] = 0.0
end

# Inject a 10.0 current into Cell 1's Soma to trigger an action potential
cell1_soma_input = net.nodes[(net.nodes.cell_idx .== 1) .& (net.nodes.comp_idx .== 1), :I_ext][1]
u0[cell1_soma_input] = 10.0

println("Setting up ODE Problem...")
prob = ODEProblem(net_compiled, u0, (0.0, 50.0))

println("Solving...")
sol = solve(prob, Rosenbrock23(), saveat=0.01)

# 7. Plot
println("Plotting...")
v_cell1_soma = net.nodes[(net.nodes.cell_idx .== 1) .& (net.nodes.comp_idx .== 1), :V][1]
v_cell2_soma = net.nodes[(net.nodes.cell_idx .== 2) .& (net.nodes.comp_idx .== 1), :V][1]
v_cell3_soma = net.nodes[(net.nodes.cell_idx .== 3) .& (net.nodes.comp_idx .== 1), :V][1]

plot(sol, idxs=[v_cell1_soma, v_cell2_soma, v_cell3_soma],
            label=["Cell 1 Soma (Pre)" "Cell 2 Soma (Post)" "Cell 3 Soma (Control)"], 
            ylabel="Voltage (mV)", xlabel="Time (ms)", lw=2, title="Synaptic Transmission")
