using ModelingToolkit: mtkcompile, @named, System, t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using Plots
using ModelingToolkitStandardLibrary.Blocks: Constant

# 1. Define Channel Dynamics
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

# 2. Build Compartments
@named soma1_cap = Capacitor(C=1.0)
@named na1 = GenericChannel(g=120.0, E_rev=50.0, gates=[GateSpec(:m, 3, 0.0, hh_na_m), GateSpec(:h, 1, 0.0, hh_na_h)])
@named k1  = GenericChannel(g=36.0, E_rev=-77.0, gates=[GateSpec(:n, 4, 0.0, hh_k_n)])
@named l1  = GenericChannel(g=0.3, E_rev=-54.4, gates=GateSpec[])
# build_compartment now returns a Compartment struct
soma_comp = build_compartment(soma1_cap, [na1, k1, l1], name=:soma, V_init=-65.0)

@named dend1_cap = Capacitor(C=0.5)
@named l2 = GenericChannel(g=0.1, E_rev=-54.4, gates=GateSpec[])
dend_comp = build_compartment(dend1_cap, [l2], name=:dend, V_init=-65.0)

# 3. Build Cell
axial_conns = [(1, 2, 0.5)]
@named stim = Constant(k=10.0)
# build_cell now returns a Cell struct
cell = build_cell([soma_comp, dend_comp], axial_conns; drivers=[(1, stim)], name=:hh_cell)

# 4. Compile and Simulate
println("Compiling cell...")
cell_compiled = mtkcompile(cell.sys)

# Set initial conditions cleanly using native MTK symbolic access.
# The GenericChannels already set the defaults for m, h, and n via GateSpec,
# so we only need to explicitly set the membrane voltages.
u0 = [
    cell_compiled.soma.soma1_cap.v => -65.0,
    cell_compiled.dend.dend1_cap.v => -65.0
]

println("Setting up ODE Problem...")
prob = ODEProblem(cell_compiled, u0, (0.0, 50.0), fully_determined=true)

println("Solving...")
sol = solve(prob, Rosenbrock23(), saveat=0.01)

# 5. Plot
println("Plotting...")
# Plot using the clean, native MTK hierarchical variable access!
plot(sol, idxs=[cell_compiled.soma.soma1_cap.v, cell_compiled.dend.dend1_cap.v],
            label=["Soma" "Dendrite"], 
            ylabel="Voltage (mV)", xlabel="Time (ms)", lw=2)
