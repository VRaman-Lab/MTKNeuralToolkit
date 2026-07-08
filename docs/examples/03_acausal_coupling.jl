# # Example 3: Acausal Coupling (Tapered Multi-Compartment Cable)
# 
# Let's look at the acausal side of ModelingToolkit. Unlike directed chemical 
# synapses, gap junctions (electrical couplings) are bidirectional. We'll build 
# a 5-compartment passive dendrite that tapers in size down its length, and inject 
# current at the soma to observe classic cable-theory voltage attenuation.

using MTKNeuralToolkit
using ModelingToolkit: mtkcompile, @named
using ModelingToolkitStandardLibrary.Blocks: Sine
using OrdinaryDiffEq
using Plots

# ------------------------------------------------------------------------------
# 1. Build Passive Compartments with Heterogeneous Geometries
# ------------------------------------------------------------------------------
top = Scalar()

# Define a tapering geometry: the soma is large, and distal dendrites are small. 
# We'll keep standard membrane capacitance (1.0 uF/cm^2) and just change the area.
areas = [0.0628, 0.0314, 0.0157, 0.0078, 0.0039] #cm^2

function build_passive_compartment(name::Symbol, area::Float64)
    geom = Geometry(area=area, C_m=1.0) #Geometry struct handles biophysical scaling
    @named cap  = Capacitor(topology=top, C=1.0, geometry=geom)
    @named leak = GenericChannel(topology=top, g=0.3, E_rev=-65.0, gates=GateSpec[], geometry=geom)
    
    return build_compartment(cap, [leak]; name=name, V_init=-65.0, topology=top)
end

# Create a chain of compartments with decreasing area
N = 5
cable = [build_passive_compartment(Symbol(:comp, i), areas[i]) for i in 1:N]

# ------------------------------------------------------------------------------
# 2. Connect Compartments with Gap Junctions
# ------------------------------------------------------------------------------
# The axial resistance between two compartments is R = (R_i * L) / A. 
# For simplicity, let's say internal resistivity R_i * length L = 1.0. 
# Smaller areas mean higher axial resistance, which causes more voltage attenuation.
# We'll calculate R based on the average area of the two connected compartments.

coupling_specs = CouplingSpec[]
for i in 1:(N-1)
    avg_area = (areas[i] + areas[i+1]) / 2.0
    R_axial = 1.0 / avg_area 
    
    #Note: you must give unique names to systems created in a loop.
    gj = GapJunction(R=R_axial; name=Symbol(:gj_, i))  
    push!(coupling_specs, CouplingSpec(cable[i], cable[i+1], gj))
end

# ------------------------------------------------------------------------------
# 3. Driving Stimuli
# ------------------------------------------------------------------------------
# Inject a slow sinusoidal current only into the first compartment (the soma)
@named current_driver = Sine(amplitude=5.0, frequency=0.05, offset=5.0)

drivers = [(1, current_driver)] 

# ------------------------------------------------------------------------------
# 4. Build and Simulate the Network
# ------------------------------------------------------------------------------
net = build_acausal_network(cable; 
                            coupling_specs=coupling_specs, 
                            drivers=drivers, 
                            name=:cable_net)

println("Compiling acausal cable network...")
sys = mtkcompile(net.sys)
prob = ODEProblem(sys, [], (0.0, 200.0))

println("Solving...")
sol = solve(prob, Rosenbrock23())

# ------------------------------------------------------------------------------
# 5. Plot the Results
# ------------------------------------------------------------------------------
p = plot(title="Example 3: Tapered Multi-Compartment Cable", 
         xlabel="Time (ms)", ylabel="V (mV)")

for i in 1:N
    comp_sys = getproperty(sys, Symbol(:comp, i))
    plot!(p, sol, idxs=[getproperty(comp_sys, :cap).v], label="Comp $i (A=$(areas[i]))")
end
p
