using MTKNeuralToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named, System
using ModelingToolkit: t_nounits as t, D_nounits as D, SymbolicT, Equation, Num
using OrdinaryDiffEq

N = 40
println("Building Fully Vectorized HH Network (N=$N)...")

# Synapse Matrices
W = fill(0.1, N, N); [W[i,i] = 0.0 for i in 1:N]
tau_mat = fill(5.0, N, N)
gmax_mat = fill(0.5 / N, N, N)

# Instantiate Vectorized Components
@named vec_neurons = VectorizedHHNeuron(N=N)
@named exc_syn_conn = VectorizedAlphaSynapse(N=N, W=W, tau=tau_mat, g_max=gmax_mat)

# Stimulus
@named stim = Blocks.Sine(frequency = 0.05, amplitude = 15.0)
drivers = [(1, stim)]

# Build and Compile
println("Compiling Network...")
t_c = @elapsed begin
    @named net = build_fully_vectorized_network(vec_neurons, [exc_syn_conn]; drivers=drivers)
    net_compiled = mtkcompile(net)
end
println("Compile time: $(round(t_c, digits=2)) seconds")

# Solve
println("Solving...")
prob = ODEProblem(net_compiled, [], (0.0, 50.0), fully_determined=true)
sol = solve(prob, Tsit5(); reltol=1e-3, abstol=1e-3)

println("Simulation finished successfully!")
println("Retcode: ", sol.retcode)
println("Final Voltage (V[1]): ", sol[vec_neurons.V[1]][end])
