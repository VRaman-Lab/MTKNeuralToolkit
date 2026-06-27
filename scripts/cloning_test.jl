using MTKNeuralToolkit
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
@named na1 = GenericChannel(g=120.0, E_rev=50.0, gates=[GateSpec(:m, 3, 0.0, hh_na_m), GateSpec(:h, 1, 1.0, hh_na_h)])
@named k1  = GenericChannel(g=36.0, E_rev=-77.0, gates=[GateSpec(:n, 4, 0.0, hh_k_n)])
@named l1  = GenericChannel(g=0.3, E_rev=-54.4, gates=GateSpec[])
soma_comp = build_floating_compartment(soma1_cap, [na1, k1, l1], name=:soma)

@named dend1_cap = Capacitor(C=0.5)
@named l2 = GenericChannel(g=0.1, E_rev=-54.4, gates=GateSpec[])
dend_comp = build_floating_compartment(dend1_cap, [l2], name=:dend)

# 3. Build Cell
axial_conns = [(1, 2, 0.5)]
@named stim = Constant(k=10.0)
cell = build_cell([soma_comp, dend_comp], axial_conns; drivers=[(1, stim)], name=:hh_cell)

# 4. Compile and Simulate
println("Compiling cell...")
cell_compiled = mtkcompile(cell.sys)

u0 = Dict()
for st in unknowns(cell_compiled)
    name_str = string(st)
    # Only set the actual capacitor voltage, skip the alias 'V'
    if occursin("cap₊v", name_str) 
        u0[st] = -65.0
    elseif occursin("h", name_str)
        u0[st] = 1.0
    elseif occursin("m", name_str) || occursin("n", name_str)
        u0[st] = 0.0
    end
end


println("Setting up ODE Problem...")
prob = ODEProblem(cell_compiled, [],(0.0, 50.0), fully_determined=true)

println("Solving...")
sol = solve(prob, Rosenbrock23(), saveat=0.01)

# 5. Plot
println("Plotting...")
# Notice the clean, native MTK variable access!
plot(sol, idxs=[cell_compiled.soma.V, cell_compiled.dend.V],
            label=["Soma" "Dendrite"], 
            ylabel="Voltage (mV)", xlabel="Time (ms)", lw=2)







# using MTKNeuralToolkit
# using ModelingToolkit: mtkcompile, @named, System, t_nounits as t, D_nounits as D
# using OrdinaryDiffEq
# using Plots
# using ModelingToolkitStandardLibrary.Blocks: Constant, Sine

# # =============================================================================
# # 1. Define Channel Dynamics
# # =============================================================================

# hh_na_m = v -> (
#     0.182 * (v + 35.0) / (1.0 - exp(-(v + 35.0) / 9.0)),
#     -0.124 * (v + 35.0) / (1.0 - exp((v + 35.0) / 9.0))
# )
# hh_na_h = v -> (
#     0.25 * exp(-(v + 90.0) / 12.0),
#     0.25 * (exp((v + 62.0) / 6.0)) / exp(-(v + 90.0) / 12.0)
# )
# hh_k_n = v -> (
#     0.02 * (v - 25.0) / (1.0 - exp(-(v - 25.0) / 9.0)),
#     -0.002 * (v - 25.0) / (1.0 - exp((v - 25.0) / 9.0))
# )

# # =============================================================================
# # 2. Build Scalar Compartments (using build_floating_compartment)
# # =============================================================================

# @named soma1_cap = Capacitor(C=1.0)
# @named na1 = GenericChannel(g=120.0, E_rev=50.0, gates=[GateSpec(:m, 3, 0.0, hh_na_m), GateSpec(:h, 1, 1.0, hh_na_h)])
# @named k1  = GenericChannel(g=36.0, E_rev=-77.0, gates=[GateSpec(:n, 4, 0.0, hh_k_n)])
# soma_scalar, soma_ifaces = build_floating_compartment(soma1_cap, [na1, k1, l1])

# @named l1  = GenericChannel(g=0.3, E_rev=-54.4, gates=GateSpec[])

# @named dend1_cap = Capacitor(C=0.5)
# @named l2 = GenericChannel(g=0.1, E_rev=-54.4, gates=GateSpec[])
# dend_scalar, dend_ifaces = build_floating_compartment(dend1_cap, [l2])





# # Axial connection: Soma (1) to Dendrite (2)
# axial_conns = [(1, 2, 0.5)]

# drivers = [(1, (; name) -> Constant(k=10.0, name=name))]

# # Or drive a specific clone:
# # drivers = [(1, 1, (; name) -> Sine(frequency=0.1, name=name))]

# println("Vectorizing and connecting...")
# @named pop1 = vectorize_and_connect([(soma_scalar, soma_ifaces), (dend_scalar, dend_ifaces)], axial_conns, 5; drivers=drivers)

# # =============================================================================
# # 3. Vectorize and Connect (Cloning to N=5 for testing)
# # =============================================================================

# println("Compiling massive cloned system...")
# pop1_compiled = mtkcompile(pop1)

# # =============================================================================
# # 4. Setup Simulation
# # =============================================================================

# u0 = Dict()
# for st in unknowns(pop1_compiled)
#     name_str = string(st)
#     if occursin("_v_", name_str) 
#         u0[st] = -65.0
#     elseif occursin("_h_", name_str)
#         u0[st] = 1.0
#     else
#         u0[st] = 0.0
#     end
# end

# println("Setting up ODE Problem...")
# # Omit the parameter dict so MTK uses the original defaults (C=1.0, g=120, etc.)
# prob = ODEProblem(pop1_compiled, u0, (0.0, 50.0))

# println("Solving...")
# sol = solve(prob, Rosenbrock23(), saveat=0.1)

# # =============================================================================
# # 5. Plot
# # =============================================================================

# println("Plotting...")
# plot(sol, idxs=[pop1_compiled.c_1_v_1, pop1_compiled.c_2_v_1],
#             label=["Soma 1" "Dendrite 1"], 
#             ylabel="Voltage (mV)", xlabel="Time (ms)", lw=2)
