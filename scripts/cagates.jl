using MTKNeuralToolkit
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq, Plots

top = Scalar()

# === Standard T-Type Calcium (CaT) Gates ===
ca_t_m = v -> (
    0.055 .* (v .+ 27.0) ./ (1.0 .- exp.(-(v .+ 27.0) ./ 3.8)),
    0.94 .* exp.(-(v .+ 27.0) ./ 17.0)
)
ca_gates = [GateSpec(:m, 2, 0.0, ca_t_m)]

# === Calcium-activated Potassium (KCa) Gate ===
kca_m = (v, Ca) -> (0.1 .* Ca, 0.1)
kca_gates = [GateSpec(:m, 1, 0.0, kca_m)]

# === Build the Compartments ===
@named cap = Capacitor(topology=top, C=1.0)
# Added conversion_factor=5.0
@named cav = CaVChannel(topology=top, g=2.0, gates=ca_gates, conversion_factor=5.0, 
                        Ca_out=3000.0, nernst_factor=13.0)
@named kca = KCaChannel(topology=top, g=8.0, E_rev=-80.0, gates=kca_gates)
@named leak = GenericChannel(topology=top, g=0.3, E_rev=-54.4, gates=GateSpec[])

# Updated to decay=50.0, and lowered Ca_init to 0.05
cell = build_compartment(cap, [cav, kca, leak]; 
                         name=:cell, V_init=-65.0, topology=top, 
                         ion_config=CalciumTracker(decay=50.0, Ca_init=0.05))

# Increased driver to 10.0
drivers = [(1, 20.0)]
net = build_acausal_network([cell]; drivers=drivers)
net_compiled = mtkcompile(net.sys)

prob = ODEProblem(net_compiled, [], (0.0, 200.0))
sol = solve(prob, Rosenbrock23())

# === Plot ===
p1 = plot(sol, idxs=net_compiled.cell.cap.v, title="Membrane Potential", legend=false)
p2 = plot(sol, idxs=net_compiled.cell.cell_ca_pool.Ca, title="Calcium Concentration", legend=false)

plot(p1, p2, layout=(2,1), size=(800,500))
