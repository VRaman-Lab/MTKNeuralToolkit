using MTKNeuralToolkit
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq
using Plots

# === Scalar gate definitions (same functions, works on scalars too) ===
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

# === Build two scalar compartments: soma + dendrite ===
@named soma_cap = Capacitor(C=1.0)
@named dend_cap = Capacitor(C=0.5)

@named soma_na   = GenericChannel(g=120.0, E_rev=50.0,  gates=sodium_gates)
@named soma_k    = GenericChannel(g=36.0,  E_rev=-77.0, gates=potassium_gates)
@named soma_leak = GenericChannel(g=0.3,   E_rev=-54.4, gates=GateSpec[])

# Dendrite has fewer channels (reduced density)
@named dend_na   = GenericChannel(g=5.0,   E_rev=50.0,  gates=sodium_gates)
@named dend_k    = GenericChannel(g=1.0,   E_rev=-77.0, gates=potassium_gates)
@named dend_leak = GenericChannel(g=0.1,   E_rev=-54.4, gates=GateSpec[])

soma = build_compartment(soma_cap, [soma_na, soma_k, soma_leak];
                          name=:soma, V_init=-65.0)
dend = build_compartment(dend_cap, [dend_na, dend_k, dend_leak];
                          name=:dend, V_init=-65.0)

# === Build cell with GapJunction axial connection ===
# axial_connections: [(pre_idx, post_idx, R)]
# R=10.0 is the axial resistance between soma and dendrite
axial = [(1, 2, 10.0)]

# Constant 10 mA current injection on soma
drivers = [(1, 10.0)]

cell = build_cell([soma, dend], axial; drivers=drivers, name=:cell)


cell_compiled = mtkcompile(cell.sys)
prob = ODEProblem(cell_compiled, [], (0.0, 100.0))
sol = solve(prob, Rosenbrock23())

# === Plot ===
p1 = plot(sol, idxs=[cell_compiled.soma.soma_cap.v, cell_compiled.dend.dend_cap.v],
          label=["Soma" "Dendrite"],
          title="2-compartment cell (GapJunction axial, R=10)",
          xlabel="Time", ylabel="V (mV)")

p2 = plot(sol, idxs=[cell_compiled.gj_1.v1, cell_compiled.gj_1.v2],
          label=["V_pre (soma side)" "V_post (dend side)"],
          title="GapJunction voltages", xlabel="Time", ylabel="V (mV)")

plot(p1, p2, layout=(2,1), size=(800,500))
