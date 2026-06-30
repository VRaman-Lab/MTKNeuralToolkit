using MTKNeuralToolkit
using ModelingToolkit
using ModelingToolkitStandardLibrary.Blocks
using ModelingToolkitStandardLibrary.Electrical.Analog: Basic
using ModelingToolkitStandardLibrary.Mechanical.Rotational
using OrdinaryDiffEq, Plots

# =============================================================================
# 1. Build the Biological Neuron
# =============================================================================
# A standard LIF compartment. Its membrane voltage will physically drive the motor!
@named neuron = build_compartment(SpikingCapacitor(C=1.0, V_th=-45.0, name=:soma), []; name=:neuron)

# =============================================================================
# 2. Build the Physical Motor and Mechanical Load
# =============================================================================
# EMF converts electrical current into mechanical torque (and back-EMF)
@named emf = Basic.EMF(k=1.0)
@named load = Rotational.Inertia(J=1.0) # Mechanical flywheel
@named ground = Rotational.Fixed()      # Mechanical ground

# =============================================================================
# 3. Build the Control System (Blocks)
# =============================================================================
@named setpoint = Blocks.Sine(frequency=0.1, amplitude=5.0) # Target angle
@named sensor = Rotational.AngleSensor()                   # Measures load angle
@named add = Blocks.Add(k1=1.0, k2=-1.0)                   # Calculates error
@named pi_controller = Blocks.PI(k=10.0, Ti=0.5)           # PI Controller

# =============================================================================
# 4. Connect the Multi-Physics Domains
# =============================================================================
eqs = [
    # --- Electrical Domain (Neuron -> Motor) ---
    connect(neuron.p, emf.p),        # Neuron positive terminal to motor armature
    connect(neuron.n, emf.n),        # Neuron ground to motor ground

    # --- Mechanical Domain (Motor -> Load) ---
    connect(emf.flange, load.flange_a),
    connect(load.flange_b, ground.flange),

    # --- Control Domain (Sensor -> PI -> Neuron) ---
    connect(load.flange_a, sensor.flange),            # Sensor measures load angle
    connect(setpoint.output, add.input1),             # Setpoint to error block
    connect(sensor.output, add.input2),               # Measured angle to error block
    connect(add.output, pi_controller.input),         # Error to PI controller
    connect(pi_controller.output, neuron.injector.I)  # PI output drives neuron current
]

all_systems = [neuron, emf, load, ground, sensor, setpoint, add, pi_controller]

@named neuro_mech_sys = System(eqs, t; systems=all_systems)
sys_compiled = mtkcompile(neuro_mech_sys)

# =============================================================================
# 5. Simulate and Plot
# =============================================================================
prob = ODEProblem(sys_compiled, [], (0.0, 100.0))
sol = solve(prob, Rosenbrock23(); reltol=1e-3, abstol=1e-3)

p1 = plot(sol, idxs=[sys_compiled.neuron.V], title="Neuron Membrane Potential (Volts)", label="V_neuron", ylabel="V")
p2 = plot(sol, idxs=[sys_compiled.load.phi], title="Mechanical Load Angle (Radians)", label="Load Angle", ylabel="rad")
p3 = plot(sol, idxs=[sys_compiled.setpoint.output.u], title="Control Setpoint", label="Setpoint", ylabel="rad")

plot(p1, p2, p3, layout=(3,1), size=(800, 600), xlabel="Time (s)")
